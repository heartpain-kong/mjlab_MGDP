"""Explicit actor-only height oracle for controlled perception diagnostics."""
from copy import deepcopy
from mjlab.managers.observation_manager import ObservationGroupCfg

def vision_mode(cfg):
  return 'height_oracle' if 'terrain_oracle' in cfg.observations else 'depth'

def configure_vision(cfg,agent_cfg,mode):
  if mode not in ('depth','height_oracle'):raise ValueError('未知地形输入模式')
  if mode=='depth':
    if vision_mode(cfg)!='depth':raise ValueError('不能原地切换地形输入模式')
    return
  from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommandCfg
  if not isinstance(cfg.commands.get('twist'),BodyAwarePassageCommandCfg):
    raise ValueError('地形真值对照仅适用于比赛尺寸强化课程')
  if vision_mode(cfg)=='depth':
    term=deepcopy(cfg.observations['critic'].terms['height_scan'])
    term.noise=None
    cfg.observations['terrain_oracle']=ObservationGroupCfg(terms={'height_scan':term},
      enable_corruption=False,concatenate_terms=True,history_length=1)
    del cfg.observations['perception']
  agent_cfg.obs_groups['actor']=['actor','terrain_oracle']

def validate_vision_resume(cfg,checkpoint):
  saved=(checkpoint.get('infos') or {}).get('mgdp_vision_mode','depth')
  if saved!=vision_mode(cfg):
    raise ValueError('深度策略与地形真值策略的输入不同，必须分别新建训练')

def restore_vision_for_play(cfg,agent_cfg,checkpoint):
  configure_vision(cfg,agent_cfg,(checkpoint.get('infos') or {}).get('mgdp_vision_mode','depth'))
