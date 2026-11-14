"""Body-aware slalom guidance and strict stepping-stone support for course V3."""
from dataclasses import dataclass,field
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor.contact_sensor import ContactSensor,ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply
from .passage import PassageCommand,PassageCommandCfg
from ..config.lgdog1.passage_terrain_cfg import START_X


SAFE_SLALOM_ROUTE=((0.,0.),(.65,-.6),(1.4,.02),(2.3,-1.22),(3.2,.02),(4.1,-1.22),
 (4.85,-1.22),(4.85,-.08),(3.5,-.08),(3.5,.3),(4.7,1.2),(4.1,1.95),
 (4.1,2.4),(5.4,2.4),(6.,1.4),(6.,.15))


def kind_columns(env,kind):
  g=env.scene.terrain.cfg.terrain_generator
  p=torch.tensor([t.proportion for t in g.sub_terrains.values()],device=env.device)
  kinds=torch.searchsorted((p/p.sum()).cumsum(0),torch.arange(g.num_cols,device=env.device)/g.num_cols+.001,right=True)
  return kinds==list(g.sub_terrains).index(kind)


class BodyAwarePassageCommand(PassageCommand):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.slalom_columns=kind_columns(env,'slalom_brick')
    self.lane_limits=torch.where(self.slalom_columns,
      torch.full((len(self.routes),),cfg.slalom_lane_half_width,device=self.device),
      torch.full((len(self.routes),),cfg.lane_half_width,device=self.device))
    self.radii=torch.where(self.slalom_columns,
      torch.full_like(self.lane_limits,cfg.slalom_waypoint_radius),
      torch.full_like(self.lane_limits,cfg.waypoint_radius))

  def route_success(self):
    types=self._env.scene.terrain.terrain_types
    final=self.routes[types,self.last_waypoint[types]]
    return ((self.waypoint_index==self.last_waypoint[types]) &
            ((self.local_xy()-final).norm(dim=-1)<=self.radii[types]))

  def _update_command(self):
    types=self._env.scene.terrain.terrain_types
    target,_,_,_=self.route_state()
    reached=(self.local_xy()-target).norm(dim=-1)<=self.radii[types]
    self.waypoint_index+=(reached & (self.waypoint_index<self.last_waypoint[types])).long()
    target,direction,_,_=self.route_state();offset=target-self.local_xy()
    distance=offset.norm(dim=-1).clamp_min(1e-6)
    velocity=offset/distance[:,None]*torch.minimum(self.speed,(1.5*distance).clamp_min(.12))[:,None]
    heading=self.robot.data.heading_w
    self.heading_target=torch.atan2(direction[:,1],direction[:,0])
    error=self.heading_target-heading;self.heading_error=torch.atan2(error.sin(),error.cos())
    # Stop translation to align in the clear turning zones; this avoids dragging
    # the offset rear half of the trunk across a post when the next segment turns.
    turning=self.slalom_columns[types] & (self.heading_error.abs()>self.cfg.slalom_heading_tolerance)
    velocity=torch.where(turning[:,None],torch.zeros_like(velocity),velocity)
    self.vel_command_b[:,0]=velocity[:,0]*heading.cos()+velocity[:,1]*heading.sin()
    self.vel_command_b[:,1]=-velocity[:,0]*heading.sin()+velocity[:,1]*heading.cos()
    self.vel_command_b[:,2]=(.8*self.heading_error).clamp(-.8,.8)


@dataclass(kw_only=True)
class BodyAwarePassageCommandCfg(PassageCommandCfg):
  slalom_lane_half_width: float=.08
  slalom_waypoint_radius: float=.06
  slalom_heading_tolerance: float=.12
  course_version: int=3
  # Part of the saved passage fingerprint: old critics/curriculum statistics
  # must not be fully resumed after changing the wall reward objective.
  wall_reward_contract_version: int|str=1

  def build(self,env):return BodyAwarePassageCommand(self,env)


class PositionHistoryContactSensor(ContactSensor):
  """Keep contact positions/normals/counts for EACH physics substep, too."""
  def initialize(self,mj_model,model,data,device):
    super().initialize(mj_model,model,data,device)
    shape=self._history_state['force'].shape
    self._history_state['pos']=torch.zeros(shape,device=device)
    self._history_state['normal']=torch.zeros(shape,device=device)
    self._history_state['found']=torch.zeros(shape[:-1],device=device)

  def _update_history(self):
    super()._update_history();current=self._extract_sensor_data()
    for name in ('pos','normal','found'):
      self._history_state[name]=self._history_state[name].roll(1,dims=2)
      self._history_state[name][:,:,0]=getattr(current,name)

  def _compute_data(self):
    result=super()._compute_data()
    for name in ('pos','normal','found'):
      setattr(result,name+'_history',self._history_state[name])
    return result


@dataclass(kw_only=True)
class PositionHistoryContactSensorCfg(ContactSensorCfg):
  def build(self):return PositionHistoryContactSensor(self)


class StoneGroundContact(ManagerTermBase):
  """Only the stone FIELD forbids floor contact; approach and exit remain safe."""
  def __init__(self,cfg,env):
    super().__init__(env);self.columns=kind_columns(env,'stepping_stones')

  def __call__(self,env,sensor_name='stone_support_contact',entry_end_x=2.2,exit_start_x=5.9,
               floor_tolerance_m=.008,min_force_n=1.):
    data=env.scene[sensor_name].data
    p=data.pos_history-env.scene.terrain.env_origins[:,None,None,:]
    loaded=(data.found_history>0)&(data.force_history.norm(dim=-1)>=min_force_n)
    floor=(p[:,:,:,2]<=floor_tolerance_m)&(data.normal_history[:,:,:,2].abs()>.7)
    # env_origins is the SPAWN (cell X=1m), not the lower-left cell corner.
    cell_x=p[:,:,:,0]+START_X
    field=(cell_x>entry_end_x)&(cell_x<exit_start_x)
    return self.columns[env.scene.terrain.terrain_types] & (loaded&floor&field).any(dim=(1,2))


def box_projection_distance(corners,points):
  """Exact distance from XY points to the convex XY projection of an oriented box.

  Projected 3D box edges include all boundary edges. Polygon face normals are
  perpendicular to the three projected box axes, so they also give an inside test.
  """
  vertices=corners[:,:,:2];center=vertices.mean(dim=1)
  axes=vertices[:,[4,2,1]]-vertices[:,0,None]
  normals=torch.stack((-axes[:,:,1],axes[:,:,0]),dim=-1)
  norm=normals.norm(dim=-1);normals=normals/norm.clamp_min(1e-9)[:,:,None]
  radius=torch.einsum('bvi,bai->bva',vertices-center[:,None],normals).abs().amax(dim=1)
  value=torch.einsum('bpi,bai->bpa',points-center[:,None],normals).abs()
  inside=((value<=radius[:,None]+1e-7)|(norm[:,None]<1e-8)).all(dim=-1)
  starts=[i for i in range(8) for bit in (1,2,4) if not i&bit]
  ends=[i^bit for i in range(8) for bit in (1,2,4) if not i&bit]
  a=vertices[:,starts];delta=vertices[:,ends]-a
  offset=points[:,:,None]-a[:,None]
  t=(offset*delta[:,None]).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)[:,None]
  closest=a[:,None]+t.clamp(0.,1.)[:,:,:,None]*delta[:,None]
  distance=(points[:,:,None]-closest).norm(dim=-1).amin(dim=-1)
  return torch.where(inside,torch.zeros_like(distance),distance)


def box_corners(position,quat,half_size):
  signs=torch.tensor([[(1 if i&4 else -1),(1 if i&2 else -1),(1 if i&1 else -1)] for i in range(8)],
                     device=position.device,dtype=position.dtype)
  local=(signs*half_size).expand(len(position),8,3)
  return position[:,None]+quat_apply(quat[:,None].expand(-1,8,-1),local)


class SlalomBodyProjection(ManagerTermBase):
  def __init__(self,cfg,env):
    super().__init__(env);self.columns=kind_columns(env,'slalom_brick')
    geom=env.sim.mj_model.geom('robot/trunk_collision')
    self.half_size=torch.tensor(geom.size,device=env.device,dtype=torch.float32)
    self.poles=torch.tensor([(1.4+i*.9,-.6) for i in range(4)]+[(4.1,.3),(4.1,1.2)],device=env.device)
    obstacle=env.scene.terrain.cfg.terrain_generator.sub_terrains['slalom_brick']
    radii=getattr(obstacle,'post_radii',None)
    self.radii=None if radii is None else torch.tensor(radii,device=env.device)

  def __call__(self,env,asset_cfg,post_radius_m=.05,margin_m=.03):
    robot=env.scene[asset_cfg.name].data
    position=robot.geom_pos_w[:,asset_cfg.geom_ids,:].squeeze(1)-env.scene.terrain.env_origins
    quat=robot.geom_quat_w[:,asset_cfg.geom_ids,:].squeeze(1)
    distance=box_projection_distance(box_corners(position,quat,self.half_size),self.poles.expand(env.num_envs,-1,-1))
    radius=post_radius_m if self.radii is None else self.radii[env.scene.terrain.terrain_levels,None]
    return self.columns[env.scene.terrain.terrain_types] & (distance<=radius+margin_m).any(dim=-1)
