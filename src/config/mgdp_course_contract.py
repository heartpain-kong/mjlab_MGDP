"""Preserve action scale and stone termination across checkpoint playback."""


def current_contract(cfg):
  stone=cfg.terminations.get('stone_ground_contact')
  return {'action_scale':float(cfg.actions['joint_pos'].scale),
          'stone_min_feet':int(stone.params.get('min_feet',1)) if stone else None}


def checkpoint_contract(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_course_contract')
  if saved is not None:return saved
  # Earlier MGDP checkpoints did not store this field. The archived V6 and
  # pre-change strengthened task both use 0.25 and one-foot termination.
  stone=cfg.terminations.get('stone_ground_contact')
  return {'action_scale':.25,'stone_min_feet':1 if stone else None}


def validate_resume(cfg,checkpoint):
  if current_contract(cfg)!=checkpoint_contract(cfg,checkpoint):
    raise ValueError('动作映射或梅花桩触地规则与检查点不同，不能完整续训；'
                     '请使用 --warm-start-checkpoint 仅迁移策略并新建实验')


def restore_for_play(cfg,checkpoint):
  restore_obstacle_speed(cfg,checkpoint)
  saved=checkpoint_contract(cfg,checkpoint)
  scale=saved.get('action_scale')
  if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not 0<scale<=1:
    raise ValueError('检查点动作映射无效')
  cfg.actions['joint_pos'].scale=float(scale)
  stone=cfg.terminations.get('stone_ground_contact')
  if stone is not None:
    count=saved.get('stone_min_feet')
    if count not in (1,2):raise ValueError('检查点梅花桩规则无效')
    stone.params['min_feet']=int(count)


def rotation_stall_contract(cfg):
  term=cfg.terminations.get('no_passage_progress')
  if term is None or term.params.get('turn_timeout_s',0)<=0:return None
  return {key:term.params[key] for key in ('turn_timeout_s','turn_progress_rad','turn_min_command_radps')}


def validate_rotation_stall_resume(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_rotation_stall')
  if rotation_stall_contract(cfg)!=saved:
    raise ValueError('转向停滞终止规则与检查点不同，不能完整续训；'
                     '请使用 --warm-start-checkpoint 仅迁移策略并新建实验')


def obstacle_speed_contract(cfg):
  limits=getattr(cfg.commands.get('twist'),'obstacle_speed_limits',None)
  return None if limits is None else {'version':1,'limits':list(limits)}


def validate_obstacle_speed_resume(cfg,checkpoint):
  if obstacle_speed_contract(cfg)!=(checkpoint.get('infos') or {}).get('mgdp_obstacle_speed'):
    raise ValueError('障碍局部速度与检查点不同；请用--warm-start-checkpoint迁移策略，新建训练')


def restore_obstacle_speed(cfg,checkpoint):
  import math
  command=cfg.commands.get('twist')
  if not hasattr(command,'obstacle_speed_limits'):return
  saved=(checkpoint.get('infos') or {}).get('mgdp_obstacle_speed')
  if saved is None:
    command.obstacle_speed_limits=None
    return
  limits=saved.get('limits')
  if (saved.get('version')!=1 or not isinstance(limits,(list,tuple)) or len(limits)!=2
      or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not .05<=v<=2 for v in limits)):
    raise ValueError('检查点障碍速度无效')
  command.obstacle_speed_limits=tuple(limits)
