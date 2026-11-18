"""Opt-in skill acquisition experiment; baseline wall settings stay independent."""
from dataclasses import dataclass,replace
import math
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import StrictStonesTerrainCfg
from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import UnifiedWallTerrainCfg

@dataclass(kw_only=True)
class SpacedTwoWallTerrainCfg(UnifiedWallTerrainCfg):
  counts: tuple[int,...]=(2,)*8
  wall_centres: tuple[float,...]=(2.3,4.7,7.1)
  def wall_parameters(self,difficulty):
    _,height=super().wall_parameters(difficulty)
    return self.wall_centres[:2],height

@dataclass(kw_only=True)
class SkillStonesTerrainCfg(StrictStonesTerrainCfg):
  counts: tuple[int,...]=(2,3,4,5,6,6,6,6)
  entry_extensions: tuple[float,...]=(0.,)*8
  def function(self,difficulty,spec,rng):
    from mjlab.terrains.terrain_generator import TerrainOutput
    stage=min(7,int(min(max(difficulty,0.),1.)*8));count=self.counts[stage]
    import mujoco
    from mjlab.terrains.terrain_generator import TerrainGeometry
    result=super().function(difficulty,spec,rng);keep=[]
    for item in result.geometries:
      if '_stone_' in item.geom.name and int(item.geom.name.rsplit('_',1)[-1])>=count:
        spec.delete(item.geom);continue
      if '_stones_exit_' in item.geom.name:item.geom.pos[0]-=(6-count)*.6
      keep.append(item)
      extension=self.entry_extensions[stage]
      if '_stones_entry_' in item.geom.name and extension>0:
        # Only the staggered right row gets an entrance tongue. Stage six and
        # above keep the full, unassisted competition entrance.
        geom=spec.body('terrain').add_geom(name=item.geom.name+'_right_tongue',
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=(2.2+extension/2,item.geom.pos[1]-.2,self.heights[stage]/2),
          size=(extension/2,self.widths[stage]/2,self.heights[stage]/2),
          rgba=(.35,.62,.20,1.),contype=1,conaffinity=1)
        keep.append(TerrainGeometry(geom=geom))
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

def configure_skill_training(cfg,version=2):
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommandCfg
  from src.tasks.velocity.mdp.skill_training import WallSkillHints,StoneTouchdown,HeadingProgress,turn_aware_tracking,SkillPhaseReward,FrontFootWallApproach
  from src.tasks.velocity.config.lgdog1.env_cfgs import FOOT_SITES
  command=cfg.commands.get('twist')
  if not isinstance(command,BodyAwarePassageCommandCfg):raise ValueError('技能引导仅适用于比赛尺寸强化课程')
  if version not in (1,2,3,4,5):raise ValueError('不支持的技能引导版本')
  if skill_contract(cfg):
    if skill_contract(cfg)!={'version':version}:raise ValueError('不能原地切换技能引导版本')
    return
  # ★ 2026-11-19：绕桩改为“走-停-转”模式，让策略真的学会转弯（此前横移模式
  #   下策略既不会转也不会侧移，11 m 处被判 left_lane）。转向时线速度置零，
  #   由 heading_error 生成 yaw 指令；杆高已抬到 20-40 cm，注意尾部扫杆风险。
  command.slalom_use_lateral=False;command.skill_training_profile={'version':version}
  # 绕桩单独加速升级：连续 1 次成功即可升档（其它地形仍用全局 streak）。
  command.terrain_level_up_streak={'slalom_brick':1}
  command.ranges.lin_vel_y=(-.35,.35)
  # Shift lanes between poles only after the offset rear of the trunk clears.
  command.waypoints['slalom_brick']=((0.,0.),(.8,.02),(2.05,.02),(2.05,-1.22),
    (2.95,-1.22),(2.95,.02),(3.85,.02),(3.85,-1.22),(4.85,-1.22),
    (4.85,-.08),(3.5,-.08),(3.5,.75),(4.75,.75),(4.75,1.7),(4.1,1.95),
    (4.1,2.4),(5.4,2.4),(6.,1.4),(6.,.15))
  g=cfg.scene.terrain.terrain_generator;old=g.sub_terrains['stepping_stones']
  g.sub_terrains['stepping_stones']=SkillStonesTerrainCfg(size=old.size,proportion=old.proportion,
    entry_extensions=(.24,.22,.18,.135,.09,0.,0.,0.) if version>=2 else (0.,)*8)
  cfg.terminations['stone_ground_contact'].params['exit_by_level']=tuple(5.9-(6-n)*.6 for n in SkillStonesTerrainCfg().counts)
  cfg.rewards['wall_skill_hints']=RewardTermCfg(func=WallSkillHints,weight=1.,params={'asset_cfg':SceneEntityCfg('robot',site_names=FOOT_SITES)})
  cfg.rewards['stone_touchdown']=RewardTermCfg(func=StoneTouchdown,weight=.25,params={'sensor_name':'stone_support_contact','margin_m':.01,'floor_tolerance_m':.008,'min_force_n':1.})
  cfg.rewards['slalom_heading_progress']=RewardTermCfg(func=HeadingProgress,weight=.5)
  tracking=cfg.rewards['track_linear_velocity']
  # The existing wall wrapper remains in place, and invokes this function.
  if 'base_func' in tracking.params:
    tracking.params={**tracking.params,'base_func':turn_aware_tracking,'original_tracking':tracking.params['base_func']}
  else:raise ValueError('技能引导需要启用高墙正奖励门控')
  from src.wall_reward_settings import load_wall_reward_settings
  from src.config.mgdp_effective_rules import resolve_wall_rules
  effective=resolve_wall_rules(load_wall_reward_settings(),version)['settings']
  for term in cfg.rewards.values():
    if 'progress_reference_mps' in term.params:
      term.params['progress_reference_mps']=effective['progress_gate']['progress_reference_mps']
  if version>=2:
    # Wrap exactly the existing positive base shapers; finite event rewards
    # and penalties are kept independent. Each wrapper owns its progress state.
    for term in cfg.rewards.values():
      if 'base_func' in term.params and term.weight>0:
        term.params={**term.params,'wrapped_func':term.func}
        term.func=SkillPhaseReward
    cfg.rewards['front_wall_approach']=RewardTermCfg(func=FrontFootWallApproach,weight=1.,
      params={'asset_cfg':SceneEntityCfg('robot',site_names=FOOT_SITES)})
  waiting=cfg.rewards['wall_waiting']
  if waiting.weight!=0:waiting.weight=effective['wall_waiting']['weight']
  waiting.params.update({k:effective['wall_waiting'][k] for k in ('grace_s','ramp_s','max_rate')})
  stall=cfg.terminations.get('no_passage_progress')
  if stall:stall.params['timeout_s']=effective['no_passage_progress']['timeout_s']
  if version>=3:
    from mjlab.managers.termination_manager import TerminationTermCfg
    from src.tasks.velocity.mdp.obstacle_rules import BeamGroundContact,WallVelocityMask,WallDistanceProgress
    cfg.rewards['passage_progress'].func=WallDistanceProgress
    cfg.terminations['beam_ground_contact']=TerminationTermCfg(func=BeamGroundContact,
      params={'sensor_name':'stone_support_contact','entry_x':2.53,'exit_x':5.47,
              'floor_tolerance_m':.008,'min_force_n':1.})
    for name in ('track_linear_velocity','track_angular_velocity'):
      term=cfg.rewards[name]
      term.params={**term.params,'velocity_base_func':term.func}
      term.func=WallVelocityMask
  if version>=4:
    old=g.sub_terrains['high_wall']
    g.sub_terrains['high_wall']=SpacedTwoWallTerrainCfg(size=old.size,proportion=old.proportion,heights=old.heights)
    command.obstacle_speed_limits=(.35,.6)
    command.ranges.lin_vel_y=(-.6,.6)
  if version==5:
    from src.config.mgdp_stages import staged_routes
    command.staged_waypoints=staged_routes(command,g)

def validate_skill_resume(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_skill_training')
  if saved!=skill_contract(cfg):raise ValueError('技能引导配置不同；请用--warm-start-checkpoint迁移策略，新建实验')

def restore_skill_for_play(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_skill_training')
  if saved is not None:
    if saved not in ({'version':1},{'version':2},{'version':3},{'version':4},{'version':5}):raise ValueError('不支持的技能引导版本')
    configure_skill_training(cfg,version=saved['version'])
  validate_skill_resume(cfg,checkpoint)

def configure_skill_policy(agent_cfg,env_cfg):
  if skill_contract(env_cfg) in ({'version':2},{'version':3},{'version':4},{'version':5}):
    if agent_cfg.clip_actions!=1.:
      raise ValueError('技能实验V2要求归一化动作裁剪范围为±1')
    distribution={**agent_cfg.actor.distribution_cfg}
    if not .05<=distribution.get('init_std',.25)<=.35:
      raise ValueError('技能实验V2动作标准差须在0.05到0.35之间')
    distribution['class_name']='src.tasks.velocity.rl.skill_distribution:BoundedMeanGaussian'
    agent_cfg.actor.distribution_cfg=distribution
