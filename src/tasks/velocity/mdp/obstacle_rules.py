"""Skill-v3 rules: valid beam support and distance-led wall rewards."""
from dataclasses import replace
import inspect
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .course_v3 import kind_columns
from .passage import ForwardProgress,failed,left_lane


class WallDistanceProgress(ForwardProgress):
  """Credit the full new wall distance, including a fast crossing step."""
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.columns=kind_columns(env,'high_wall')

  def __call__(self,env):
    previous=self.highwater.clone()
    base=super().__call__(env)
    distance=(self.highwater-previous).clamp_min(0)/env.step_dt
    valid=~left_lane(env)&~failed(env)
    return torch.where(self.columns[env.scene.terrain.terrain_types],distance*valid,base)


class BeamGroundContact(ManagerTermBase):
  """A loaded foot on the floor inside the beam's longitudinal span fails.

  Approach/exit ground and airborne feet remain valid. Inspect every recorded
  physics substep; a one-sided floor walk cannot count as a beam crossing.
  """
  def __init__(self,cfg,env):
    super().__init__(env)
    self.columns=kind_columns(env,'balance_beam')

  def __call__(self,env,sensor_name='stone_support_contact',entry_x=2.53,
               exit_x=5.47,floor_tolerance_m=.008,min_force_n=1.):
    data=env.scene[sensor_name].data
    p=data.pos_history-env.scene.terrain.env_origins[:,None,None,:]
    loaded=(data.found_history>0)&(data.force_history.norm(dim=-1)>=min_force_n)
    # Recorded contacts are relative to spawn x=1m, while geometry is cell-local.
    cell_x=p[:,:,:,0]+1.
    floor=(p[:,:,:,2]<=floor_tolerance_m)&(data.normal_history[:,:,:,2].abs()>.7)
    invalid=loaded&floor&(cell_x>entry_x)&(cell_x<exit_x)
    return self.columns[env.scene.terrain.terrain_types]&invalid.any(dim=(1,2))


class WallVelocityMask(ManagerTermBase):
  """Remove only wall velocity-tracking reward; keep other lessons unchanged."""
  def __init__(self,cfg,env):
    super().__init__(env)
    self.columns=kind_columns(env,'high_wall')
    self.positive_weight=cfg.weight>0
    base=cfg.params['velocity_base_func']
    params={k:v for k,v in cfg.params.items() if k!='velocity_base_func'}
    self.base=base(cfg=replace(cfg,func=base,params=params),env=env) if inspect.isclass(base) else base

  def reset(self,env_ids=None):
    if hasattr(self.base,'reset'):self.base.reset(env_ids=env_ids)

  def __call__(self,env,velocity_base_func,**params):
    value=self.base(env,**params)
    if not self.positive_weight:return value
    wall=self.columns[env.scene.terrain.terrain_types]
    return torch.where(wall&(value>0),torch.zeros_like(value),value)
