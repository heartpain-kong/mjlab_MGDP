"""Finite skill acquisition hints, used only by the opt-in skill experiment."""
import torch
from mjlab.managers.manager_base import ManagerTermBase
from . import passage
from .wall_training import WallGeometry
from .tire_training import TireHoleTouchdown
from .course_v3 import kind_columns

class WallSkillHints(WallGeometry):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.peak=torch.zeros(env.num_envs,4,3,device=env.device)
    self.paid=torch.zeros_like(self.peak,dtype=torch.bool)
    self.ready=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
  def reset(self,env_ids=None):
    ids=slice(None) if env_ids is None else env_ids
    self.peak[ids]=0;self.paid[ids]=False;self.ready[ids]=False
  def __call__(self,env,asset_cfg):
    feet=self.feet(env,asset_cfg);active=self.active_walls(env)[:,None,:]
    distance=self.centres[None,None,:]-feet[:,:,0,None]
    near=(distance>=-.08)&(distance<=.35)&active
    target=(self.heights[env.scene.terrain.terrain_levels]+.03)[:,None,None]
    height=(feet[:,:,2,None]/target).clamp(0,1)
    record=torch.where(near,torch.maximum(self.peak,height),self.peak)
    lift=(record-self.peak).sum(dim=(1,2));self.peak[:]=record
    root=env.command_manager.get_term('twist').local_xy()[:,0]
    clear=(feet[:,:,0,None]>=self.centres[None,None,:]+.07)&active
    clear&=root[:,None,None]>=self.centres[None,None,:]-.3
    crossed=(clear&~self.paid).sum(dim=(1,2))
    eligible=self.ready&~passage.failed(env)&~passage.left_lane(env)
    self.paid|=clear;self.ready[:]=True
    return (.25*lift+.25*crossed)*eligible/env.step_dt

class StoneTouchdown(TireHoleTouchdown):
  def __init__(self,cfg,env):
    super().__init__(cfg,env);self.columns=kind_columns(env,'stepping_stones')
    x=1.4+torch.arange(6,device=env.device)*.6
    stagger=torch.tensor([.3,0.,.3,0.],device=env.device)
    y=torch.tensor([-.2,.2,-.2,.2],device=env.device)
    self.centres=torch.stack((x[None,:]+stagger[:,None],y[:,None].expand(-1,6)),dim=-1)
    stone=env.scene.terrain.cfg.terrain_generator.sub_terrains['stepping_stones']
    self.heights=torch.tensor(stone.heights,device=env.device)
    from ..config.lgdog1.course_v4_terrain_cfg import TIRE_INNER_APOTHEM
    radii=TIRE_INNER_APOTHEM-self.safe_radius-cfg.params['margin_m']
    self.safe_width=torch.tensor(stone.widths,device=env.device)[:,None]/2-radii[None,:]-cfg.params['margin_m']
    self.counts=torch.tensor(stone.counts,device=env.device)
    self.paid=torch.zeros(env.num_envs,4,6,device=env.device,dtype=torch.bool)
  def __call__(self,env,sensor_name,margin_m=.01,floor_tolerance_m=.008,min_force_n=1.):
    data=env.scene[sensor_name].data;n=env.num_envs
    def history(value,vector=False):
      shape=(n,4,self.slots,-1,3) if vector else (n,4,self.slots,-1)
      return value.reshape(shape)[:,self.order].flip(3)
    pos=history(data.pos_history,True)-env.scene.terrain.env_origins[:,None,None,None,:]
    normal=history(data.normal_history,True)
    loaded=(history(data.found_history)>0)&(history(data.force_history,True).norm(dim=-1)>=min_force_n)
    contact=loaded.any(dim=2)
    touchdown=contact&~torch.cat((self.previous[:,:,None],contact[:,:,:-1]),dim=2)
    self.previous[:]=contact[:,:,-1];levels=env.scene.terrain.terrain_levels
    offset=(pos[:,:,:,:,None,:2]-self.centres[None,:,None,None,:,:]).abs().amax(dim=-1)
    inside=offset<=self.safe_width[levels,:,None,None,None]
    top=(pos[:,:,:,:,2]-self.heights[levels,None,None,None]).abs()<=floor_tolerance_m
    top&=normal[:,:,:,:,2].abs()>.7
    valid=(loaded&top&touchdown[:,:,None,:])[:,:,:,:,None]&inside
    eligible=self.columns[env.scene.terrain.terrain_types]&self.ready&~passage.failed(env)&~passage.left_lane(env)
    landed=valid.any(dim=(2,3))&eligible[:,None,None]
    landed&=torch.arange(6,device=env.device)[None,None,:]<self.counts[levels,None,None]
    new=landed&~self.paid;self.paid|=landed;self.ready[:]=True
    return new.sum(dim=(1,2)).float()/env.step_dt

class HeadingProgress(ManagerTermBase):
  def __init__(self,cfg,env):
    super().__init__(env);self.columns=kind_columns(env,'slalom_brick')
    self.best=torch.zeros(env.num_envs,device=env.device)
    self.waypoint=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
  def reset(self,env_ids=None):
    self.waypoint[slice(None) if env_ids is None else env_ids]=-1
  def __call__(self,env):
    cmd=env.command_manager.get_term('twist');error=cmd.heading_error.abs()
    best=torch.where(cmd.waypoint_index!=self.waypoint,error,self.best)
    delta=(best-error).clamp_min(0);self.best[:]=torch.minimum(best,error)
    self.waypoint[:]=cmd.waypoint_index
    valid=self.columns[env.scene.terrain.terrain_types]&~passage.failed(env)&~passage.left_lane(env)
    return delta*valid/env.step_dt

def turn_aware_tracking(env,original_tracking,**params):
  cmd=env.command_manager.get_term('twist');value=original_tracking(env,**params)
  turning=cmd.slalom_columns[env.scene.terrain.terrain_types]&(cmd.heading_error.abs()>.12)
  factor=(cmd.command[:,:2].norm(dim=1)/.3).clamp(.1,1.)
  return value*torch.where(turning,factor,torch.ones_like(factor))
