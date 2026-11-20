"""One stdlib-only rule resolver shared by training and the workbench."""
from copy import deepcopy


def resolve_wall_rules(settings, skill_version=None):
  result=deepcopy(settings)
  if not result['wall_waiting']['enabled']:result['wall_waiting']['weight']=0.
  if skill_version is not None:
    if skill_version not in (1,2,3,4,5):raise ValueError('不支持的技能规则版本')
    if not result['progress_gate']['enabled']:raise ValueError('技能引导需要启用高墙正奖励门控')
    result['progress_gate']['progress_reference_mps']=.15
    waiting=result['wall_waiting']
    if waiting['weight']!=0:waiting['weight']=-.5
    waiting.update(grace_s=1.5,ramp_s=2.,max_rate=1.)
    result['no_passage_progress']['timeout_s']=6.
  changes=[{'group':group,'key':key,'saved':settings[group][key],'effective':value,
            'source':f'技能规则V{skill_version}' if skill_version is not None else '启用开关'}
    for group,values in result.items() for key,value in values.items()
    if settings[group][key]!=value]
  return {'settings':result,'changes':changes,'skill_version':skill_version,
          'wall_velocity_reward':not(skill_version is not None and skill_version>=3),
          'max_waiting_cost_per_second':abs(result['wall_waiting']['weight'])*result['wall_waiting']['max_rate']
            if result['wall_waiting']['enabled'] else 0.}


def effective_rule_snapshot(cfg):
  """Read actual manager bindings AFTER all training overrides."""
  from src.wall_reward_settings import load_wall_reward_settings
  version=getattr(cfg.commands.get('twist'),'skill_training_profile',{}).get('version')
  data=resolve_wall_rules(load_wall_reward_settings(),version)
  settings=data['settings']
  pose=cfg.rewards.get('pose')
  if pose and 'progress_reference_mps' in pose.params:
    settings['progress_gate'].update(progress_reference_mps=pose.params['progress_reference_mps'],
      min_command_mps=pose.params['min_command_mps'])
  for group in ('wall_waiting','wall_crossing'):
    term=cfg.rewards.get(group)
    if term:
      for key in settings[group]:
        if key=='weight':settings[group][key]=term.weight
        elif key=='enabled':settings[group][key]=term.weight!=0
        elif key in term.params:settings[group][key]=term.params[key]
  for group in ('passage_progress','passage_success'):
    settings[group]['weight']=cfg.rewards[group].weight
  stall=cfg.terminations.get('no_passage_progress')
  settings['no_passage_progress']['enabled']=stall is not None
  if stall:
    settings['no_passage_progress'].update({k:v for k,v in stall.params.items() if k in settings['no_passage_progress']})
  data['max_waiting_cost_per_second']=abs(settings['wall_waiting']['weight'])*settings['wall_waiting']['max_rate'] if settings['wall_waiting']['enabled'] else 0.
  data['staged_waypoints']=getattr(cfg.commands['twist'],'staged_waypoints',{})
  data['obstacle_speed_limits']=cfg.commands['twist'].obstacle_speed_limits
  data['action_scale']=cfg.actions['joint_pos'].scale
  return data
