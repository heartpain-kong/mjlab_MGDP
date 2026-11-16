"""Opt-in skill acquisition experiment; baseline wall settings stay independent."""
from dataclasses import dataclass,replace
import math
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import StrictStonesTerrainCfg

@dataclass(kw_only=True)
class SkillStonesTerrainCfg(StrictStonesTerrainCfg):
  counts: tuple[int,...]=(2,3,4,5,6,6,6,6)
  def function(self,difficulty,spec,rng):
    from mjlab.terrains.terrain_generator import TerrainOutput
    stage=min(7,int(min(max(difficulty,0.),1.)*8));count=self.counts[stage]
    result=super().function(difficulty,spec,rng);keep=[]
    for item in result.geometries:
      if '_stone_' in item.geom.name and int(item.geom.name.rsplit('_',1)[-1])>=count:
        spec.delete(item.geom);continue
      if '_stones_exit_' in item.geom.name:item.geom.pos[0]-=(6-count)*.6
      keep.append(item)
    return TerrainOutput(origin=result.origin,geometries=keep)

def skill_contract(cfg):
  return getattr(cfg.commands.get('twist'),'skill_training_profile',{}) or None

def configure_action_scale(cfg,value):
  from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommandCfg
  if not isinstance(cfg.commands.get('twist'),BodyAwarePassageCommandCfg):
    raise ValueError('动作映射实验仅适用于比赛尺寸强化课程')
  if isinstance(value,bool) or not math.isfinite(value) or not .2<=value<=.5:
    raise ValueError('强化课程动作映射实验范围为0.2到0.5')
  cfg.actions['joint_pos'].scale=float(value)

def configure_skill_training(cfg):
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommandCfg
  from src.tasks.velocity.mdp.skill_training import WallSkillHints,StoneTouchdown,HeadingProgress,turn_aware_tracking
  from src.tasks.velocity.config.lgdog1.env_cfgs import FOOT_SITES
  command=cfg.commands.get('twist')
  if not isinstance(command,BodyAwarePassageCommandCfg):raise ValueError('技能引导仅适用于比赛尺寸强化课程')
  if skill_contract(cfg):return
  command.slalom_use_lateral=True;command.skill_training_profile={'version':1}
  command.ranges.lin_vel_y=(-.35,.35)
  # Shift lanes between poles only after the offset rear of the trunk clears.
  command.waypoints['slalom_brick']=((0.,0.),(.8,.02),(2.05,.02),(2.05,-1.22),
    (2.95,-1.22),(2.95,.02),(3.85,.02),(3.85,-1.22),(4.85,-1.22),
    (4.85,-.08),(3.5,-.08),(3.5,.75),(4.75,.75),(4.75,1.7),(4.1,1.95),
    (4.1,2.4),(5.4,2.4),(6.,1.4),(6.,.15))
  g=cfg.scene.terrain.terrain_generator;old=g.sub_terrains['stepping_stones']
  g.sub_terrains['stepping_stones']=SkillStonesTerrainCfg(size=old.size,proportion=old.proportion)
  cfg.terminations['stone_ground_contact'].params['exit_by_level']=tuple(5.9-(6-n)*.6 for n in SkillStonesTerrainCfg().counts)
  cfg.rewards['wall_skill_hints']=RewardTermCfg(func=WallSkillHints,weight=1.,params={'asset_cfg':SceneEntityCfg('robot',site_names=FOOT_SITES)})
  cfg.rewards['stone_touchdown']=RewardTermCfg(func=StoneTouchdown,weight=.25,params={'sensor_name':'stone_support_contact','margin_m':.01,'floor_tolerance_m':.008,'min_force_n':1.})
  cfg.rewards['slalom_heading_progress']=RewardTermCfg(func=HeadingProgress,weight=.5)
  tracking=cfg.rewards['track_linear_velocity']
  # The existing wall wrapper remains in place, and invokes this function.
  if 'base_func' in tracking.params:
    tracking.params={**tracking.params,'base_func':turn_aware_tracking,'original_tracking':tracking.params['base_func']}
  else:raise ValueError('技能引导需要启用高墙正奖励门控')
  for term in cfg.rewards.values():
    if 'progress_reference_mps' in term.params:term.params['progress_reference_mps']=.15
  waiting=cfg.rewards['wall_waiting']
  if waiting.weight!=0:waiting.weight=-.5
  waiting.params.update(grace_s=1.5,ramp_s=2.,max_rate=1.)
  stall=cfg.terminations.get('no_passage_progress')
  if stall:stall.params['timeout_s']=6.

def validate_skill_resume(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_skill_training')
  if saved!=skill_contract(cfg):raise ValueError('技能引导配置不同；请用--warm-start-checkpoint迁移策略，新建实验')

def restore_skill_for_play(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_skill_training')
  if saved is not None:
    if saved!={'version':1}:raise ValueError('不支持的技能引导版本')
    configure_skill_training(cfg)
  validate_skill_resume(cfg,checkpoint)
