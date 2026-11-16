"""Explicit opt-in to a changed foot-placement objective; keep old runs intact."""
import hashlib

GUIDANCE_VERSION=1


def guidance_contract(env_cfg):
  term=env_cfg.rewards.get('tire_hole_touchdown')
  if term is None:return None
  return {'version':GUIDANCE_VERSION,'weight':term.weight,'params':dict(term.params)}


def configure_tire_hole_guidance(env_cfg):
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import UnifiedObstacleTerrainCfg
  from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommandCfg
  from src.tasks.velocity.mdp.tire_training import TireHoleTouchdown,touchdown_metric
  terrain=env_cfg.scene.terrain
  generator=None if terrain is None else terrain.terrain_generator
  tire=None if generator is None else generator.sub_terrains.get('tire_array')
  command=env_cfg.commands.get('twist')
  if not isinstance(tire,UnifiedObstacleTerrainCfg) or not isinstance(command,BodyAwarePassageCommandCfg):
    raise ValueError('轮胎内孔落脚引导仅适用于比赛尺寸强化课程（含D435i20）')
  if guidance_contract(env_cfg) is not None:return
  env_cfg.rewards['tire_hole_touchdown']=RewardTermCfg(func=TireHoleTouchdown,weight=.25,
    params={'sensor_name':'stone_support_contact','margin_m':.01,
            'floor_tolerance_m':.008,'min_force_n':1.})
  for name,valid in (('tire_hole_new_landings_per_step',True),('tire_field_touchdowns_per_step',False)):
    env_cfg.metrics[name]=MetricsTermCfg(func=touchdown_metric,params={'valid':valid})
  contract=(command.wall_reward_contract_version,guidance_contract(env_cfg))
  command.wall_reward_contract_version=hashlib.sha256(repr(contract).encode()).hexdigest()


def validate_tire_guidance_resume(env_cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_tire_hole_guidance')
  if saved!=guidance_contract(env_cfg):
    raise ValueError('轮胎落脚引导与检查点不一致，不能完整续训；请匹配原设置，'
                     '或用--warm-start-checkpoint仅迁移策略后新建训练')


def restore_tire_guidance_for_play(env_cfg,checkpoint):
  if (checkpoint.get('infos') or {}).get('mgdp_tire_hole_guidance') is not None:
    configure_tire_hole_guidance(env_cfg)
  validate_tire_guidance_resume(env_cfg,checkpoint)
