"""Optional tire-hole touchdown guidance; no actor inputs or terrain changes."""
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .course_v3 import kind_columns
from . import passage
from ..config.lgdog1.course_v4_terrain_cfg import (
  TIRE_CENTER_X,TIRE_SPACING,TIRE_COLUMNS,TIRE_INNER_APOTHEM,
)
from ..config.lgdog1.passage_terrain_cfg import START_X

FOOT_NAMES=('FR','FL','RR','RL')


def hole_centres(device='cpu'):
  """[leg, longitudinal hole, XY], relative to the tile spawn."""
  x=TIRE_CENTER_X-START_X+(torch.arange(TIRE_COLUMNS,device=device)-(TIRE_COLUMNS-1)/2)*TIRE_SPACING
  y=torch.tensor([-1.,1.,-1.,1.],device=device)*TIRE_SPACING/2
  return torch.stack((x.expand(4,-1),y[:,None].expand(-1,TIRE_COLUMNS)),dim=-1)


class TireHoleTouchdown(ManagerTermBase):
  """Pay once per leg/hole/episode on a loaded FLOOR touchdown inside its hole.

  Inspect every physics substep, including brief contacts. Tire tops/sides,
  wrong rows, airborne feet and repeated stance never earn this bonus.
  """
  def __init__(self,cfg,env):
    super().__init__(env)
    self.columns=kind_columns(env,'tire_array')
    self.centres=hole_centres(env.device)
    sensor=env.scene[cfg.params['sensor_name']]
    names=[s.primary_name.split('/')[-1] for s in sensor._slots if s.field_name=='found']
    expected=[n+'_calf1_collision' for n in FOOT_NAMES]
    if len(names)!=4 or set(names)!=set(expected):
      raise ValueError('Tire guidance requires four individually identified foot contacts')
    self.order=torch.tensor([names.index(n) for n in expected],device=env.device)
    self.slots=sensor.cfg.num_slots
    radii=[]
    for name in expected:
      g=env.sim.mj_model.geom('robot/'+name)
      if int(env.sim.mj_model.geom_type[g.id])!=2:
        raise ValueError('Tire guidance requires spherical foot collision geoms')
      radii.append(float(env.sim.mj_model.geom_size[g.id,0]))
    margin=cfg.params['margin_m']
    if not 0<=margin<TIRE_INNER_APOTHEM-max(radii):raise ValueError('Invalid tire-hole safety margin')
    self.safe_radius=TIRE_INNER_APOTHEM-torch.tensor(radii,device=env.device)-margin
    self.paid=torch.zeros(env.num_envs,4,TIRE_COLUMNS,device=env.device,dtype=torch.bool)
    self.previous=torch.zeros(env.num_envs,4,device=env.device,dtype=torch.bool)
    self.ready=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    self.last_valid=torch.zeros(env.num_envs,device=env.device)
    self.last_attempts=torch.zeros_like(self.last_valid)

  def reset(self,env_ids=None):
    ids=slice(None) if env_ids is None else env_ids
    self.paid[ids]=False;self.previous[ids]=False;self.ready[ids]=False
    self.last_valid[ids]=0;self.last_attempts[ids]=0

  def __call__(self,env,sensor_name,margin_m=.01,floor_tolerance_m=.008,min_force_n=1.):
    data=env.scene[sensor_name].data;n=env.num_envs
    # The history is newest first; use chronological order for touchdown edges.
    def history(value,vector=False):
      shape=(n,4,self.slots,-1,3) if vector else (n,4,self.slots,-1)
      return value.reshape(shape)[:,self.order].flip(3)
    pos=history(data.pos_history,True)-env.scene.terrain.env_origins[:,None,None,None,:]
    normal=history(data.normal_history,True)
    loaded=(history(data.found_history)>0)&(history(data.force_history,True).norm(dim=-1)>=min_force_n)
    contact=loaded.any(dim=2)
    previous=torch.cat((self.previous[:,:,None],contact[:,:,:-1]),dim=2)
    touchdown=contact&~previous
    self.previous[:]=contact[:,:,-1]
    distance=(pos[:,:,:,:,None,:2]-self.centres[None,:,None,None,:,:]).norm(dim=-1)
    floor=(pos[:,:,:,:,2].abs()<=floor_tolerance_m)&(normal[:,:,:,:,2].abs()>.7)
    inside=distance<=self.safe_radius[None,:,None,None,None]
    valid=(loaded&floor&touchdown[:,:,None,:])[:,:,:,:,None]&inside
    active=self.columns[env.scene.terrain.terrain_types]&self.ready
    eligible=active&~passage.failed(env)&~passage.left_lane(env)
    landed=valid.any(dim=(2,3))&eligible[:,None,None]
    new=landed&~self.paid
    self.paid|=landed
    self.last_valid[:]=new.sum(dim=(1,2))
    # Attempts in the longitudinal field, including missed holes and tire tops.
    field=(pos[:,:,:,:,0]>=self.centres[0,0,0]-.275)&(pos[:,:,:,:,0]<=self.centres[0,-1,0]+.275)
    self.last_attempts[:]=((loaded&field).any(dim=2)&touchdown&eligible[:,None,None]).sum(dim=(1,2))
    self.ready[:]=True
    return self.last_valid/env.step_dt


def touchdown_metric(env,valid=True):
  term=env.reward_manager.get_term_cfg('tire_hole_touchdown').func
  return term.last_valid if valid else term.last_attempts
