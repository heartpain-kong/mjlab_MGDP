"""Explicit directed-passage cruise speed; independent of signed velocity ranges."""
import math


def validate_fixed_speed(value):
  if isinstance(value,bool):
    raise ValueError('固定巡航速度必须为0.05到3.0m/s的有限数')
  speed=float(value)
  if not math.isfinite(speed) or not .05 <= speed <= 3.:
    raise ValueError('固定巡航速度必须为0.05到3.0m/s的有限数')
  return speed


def configure_passage_fixed_speed(env_cfg,value):
  from src.tasks.velocity.mdp.passage import PassageCommandCfg
  command=env_cfg.commands.get('twist')
  if not isinstance(command,PassageCommandCfg):
    raise ValueError('--passage-fixed-speed仅适用于定向越障任务')
  speed=validate_fixed_speed(value)
  command.ranges.lin_vel_x=(speed,speed)
  curriculum=env_cfg.curriculum.get('passage')
  if curriculum is not None:
    curriculum.params={**curriculum.params,'fixed_speed':speed}
  return speed


def checkpoint_fixed_speed(checkpoint):
  state=(checkpoint.get('infos') or {}).get('mgdp_passage_curriculum') or {}
  value=(state.get('params') or {}).get('fixed_speed')
  return None if value is None else validate_fixed_speed(value)


def validate_passage_speed_resume(env_cfg,checkpoint):
  curriculum=env_cfg.curriculum.get('passage')
  if curriculum is None:return
  requested=curriculum.params.get('fixed_speed')
  saved=checkpoint_fixed_speed(checkpoint)
  if requested!=saved:
    raise ValueError('固定巡航速度或速度模式与检查点不同，不能完整续训；'
                     '请匹配原速度设置，或用--warm-start-checkpoint迁移策略后新建课程')
  state=(checkpoint.get('infos') or {}).get('mgdp_passage_curriculum') or {}
  previous=state.get('params') or {}
  defaults={'terrain_max_error':'speed','level_up_streak':1,'level_down_streak':1}
  if any(curriculum.params.get(key,default)!=previous.get(key,default)
         for key,default in defaults.items()):
    raise ValueError('地形升降档规则与检查点不同，不能完整续训；'
                     '请用--warm-start-checkpoint迁移策略后新建课程')


def restore_fixed_speed_for_play(env_cfg,checkpoint):
  speed=checkpoint_fixed_speed(checkpoint)
  if speed is not None:configure_passage_fixed_speed(env_cfg,speed)
  else:
    state=(checkpoint.get('infos') or {}).get('mgdp_passage_curriculum') or {}
    ranges=state.get('ranges')
    if ranges is not None:
      if len(ranges)!=2 or not all(math.isfinite(float(x)) for x in ranges) or not 0<=ranges[0]<=ranges[1]<=3:
        raise ValueError('检查点速度范围无效')
      env_cfg.commands['twist'].ranges.lin_vel_x=tuple(float(x) for x in ranges)
  return speed
