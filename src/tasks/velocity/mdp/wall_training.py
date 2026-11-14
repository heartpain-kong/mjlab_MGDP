"""Auxiliary passage training: stagnation, one-time crossings and swing clearance.

These terms belong to the explicit WallTraining task, never the original course
used to evaluate a policy. All state and reductions remain on the device.
"""
from dataclasses import replace
import inspect
import torch
from mjlab.managers.manager_base import ManagerTermBase
from . import passage
from ..config.lgdog1.passage_terrain_cfg import (
  START_X, WALL_CENTRES, WALL_HALF_THICKNESS, WallSkillTerrainCfg,
)


class NoPassageProgress(ManagerTermBase):
  """Terminate a commanded-moving episode after 4s without 5cm NEW progress.

  A high-water mark prevents forward/backward stepping from refreshing the timer.
  Unlike a time limit, this is a failure and must not receive timeout bootstrap.
  """
  def __init__(self, cfg, env):
    super().__init__(env)
    self.highwater = torch.zeros(env.num_envs, device=env.device)
    self.anchor = torch.zeros_like(self.highwater)
    self.idle = torch.zeros_like(self.highwater)
    self.ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.ready[ids] = False
    self.idle[ids] = 0.

  def __call__(self, env, timeout_s=4., min_progress_m=.05, min_command_mps=.1):
    if timeout_s <= 0 or min_progress_m <= 0:
      raise ValueError("Stagnation window and minimum progress must be positive")
    command = env.command_manager.get_term("twist")
    progress = command.route_state()[3]
    self.highwater = torch.where(self.ready, torch.maximum(self.highwater, progress), progress)
    self.anchor = torch.where(self.ready, self.anchor, progress)
    moving = command.command[:, :2].norm(dim=1) >= min_command_mps
    advanced = self.highwater - self.anchor >= min_progress_m
    self.idle = torch.where(self.ready & moving & ~advanced,
                            self.idle + env.step_dt, torch.zeros_like(self.idle))
    self.anchor = torch.where(advanced | ~moving, self.highwater, self.anchor)
    self.ready[:] = True
    return (self.idle >= timeout_s - 1e-5) & moving & ~passage.passage_success(env)


class WallGeometry(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    wall = generator.sub_terrains.get("high_wall")
    if not isinstance(wall, WallSkillTerrainCfg) or generator.num_rows != 8:
      raise ValueError("WallTraining requires its eight-row auxiliary terrain")
    if tuple(generator.difficulty_range) != (0., 1.):
      raise ValueError("WallTraining requires difficulty_range=(0,1)")
    names = list(generator.sub_terrains)
    proportions = torch.tensor([c.proportion for c in generator.sub_terrains.values()], device=env.device)
    columns = torch.searchsorted((proportions/proportions.sum()).cumsum(0),
      torch.arange(generator.num_cols, device=env.device)/generator.num_cols+.001, right=True)
    self.wall_columns = columns == names.index("high_wall")
    self.centres = torch.tensor(WALL_CENTRES, device=env.device) - START_X
    self.counts = torch.tensor(wall.counts, device=env.device)
    self.heights = torch.tensor(wall.heights, device=env.device)
    self.indices = torch.arange(3, device=env.device)

  def active_walls(self, env):
    terrain = env.scene.terrain
    return (self.wall_columns[terrain.terrain_types, None] &
            (self.indices[None, :] < self.counts[terrain.terrain_levels, None]))

  def feet(self, env, asset_cfg):
    return (env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, :] -
            env.scene.terrain.env_origins[:, None, :])


class WallCrossingBonus(WallGeometry):
  """Pay once only after all four feet and the trunk have cleared a wall."""
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.paid = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.bool)
    self.ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.paid[ids] = False
    self.ready[ids] = False

  def __call__(self, env, asset_cfg, root_clearance_m=.35, foot_clearance_m=.03):
    command = env.command_manager.get_term("twist")
    rear = self.centres + WALL_HALF_THICKNESS
    foot_x = self.feet(env, asset_cfg)[:, :, 0].amin(dim=1)
    root_x = command.local_xy()[:, 0]
    clear = ((foot_x[:, None] >= rear + foot_clearance_m) &
             (root_x[:, None] >= rear + root_clearance_m) & self.active_walls(env))
    # Seed at the first actual pose after reset: no credit for spawning past a wall.
    self.paid |= clear & ~self.ready[:, None]
    valid = ~passage.failed(env) & ~passage.left_lane(env)
    new = clear & ~self.paid & self.ready[:, None] & valid[:, None]
    self.paid |= new
    self.ready[:] = True
    return new.sum(dim=1).float() / env.step_dt


class WallSwingClearance(WallGeometry):
  """Cost for a FORWARD swing foot below the nearby wall top + 3cm.

  Stance feet and feet far from the wall are excluded. No fixed jump-height
  target is imposed on the other lessons or on approach/exit pads.
  """
  def __call__(self, env, asset_cfg, sensor_name="feet_ground_contact",
               margin_m=.03, approach_m=.25, departure_m=.08):
    feet = self.feet(env, asset_cfg)
    speed = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, 0]
    in_air = env.scene[sensor_name].data.found == 0
    distance = self.centres[None, None, :] - feet[:, :, 0, None]
    near = ((distance >= -departure_m) & (distance <= approach_m) &
            self.active_walls(env)[:, None, :])
    target = self.heights[env.scene.terrain.terrain_levels, None] + margin_m
    deficit = ((target - feet[:, :, 2]).clamp(min=0) / .1).square().clamp(max=4.)
    mask = near.any(dim=-1) & in_air & (speed > .03)
    return (deficit * mask).mean(dim=1)


class WallProgressReward(WallGeometry):
  """Gate positive shaping by NEW distance; retain every negative value.

  One instance wraps each original term, so class-based posture rewards keep
  their setup and resolved entity parameters. No timer or shared GPU state is
  advanced multiple times by different reward terms.
  """
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.highwater=torch.zeros(env.num_envs,device=env.device)
    self.ready=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    self.positive_weight=cfg.weight>0
    base=cfg.params['base_func']
    params={k:v for k,v in cfg.params.items() if k not in
            ('base_func','progress_reference_mps','min_command_mps')}
    self.base=base(cfg=replace(cfg,func=base,params=params),env=env) if inspect.isclass(base) else base

  def reset(self,env_ids=None):
    ids=slice(None) if env_ids is None else env_ids
    self.ready[ids]=False
    if hasattr(self.base,'reset'):self.base.reset(env_ids=env_ids)

  def __call__(self,env,base_func,progress_reference_mps=.1,min_command_mps=.1,**params):
    del base_func
    if progress_reference_mps<=0:raise ValueError('Progress reference speed must be positive')
    value=self.base(env,**params)
    if not self.positive_weight:return value
    command=env.command_manager.get_term('twist');progress=command.route_state()[3]
    previous=torch.where(self.ready,self.highwater,progress)
    delta=(progress-previous).clamp_min(0.)
    self.highwater=torch.maximum(previous,progress);self.ready[:]=True
    moving=command.command[:,:2].norm(dim=1)>=min_command_mps
    gated=self.wall_columns[env.scene.terrain.terrain_types]&moving
    factor=(delta/(env.step_dt*progress_reference_mps)).clamp(max=1.)
    factor=torch.where(gated,factor,torch.ones_like(factor))
    return torch.where(value>0,value*factor,value)


class WallWaitingPenalty(WallGeometry):
  """Bounded time cost near the next uncleared wall, using record progress."""
  def __init__(self,cfg,env):
    if getattr(cfg,'weight',-1.)>0:
      raise ValueError('wall_waiting is a nonnegative cost and requires a nonpositive weight')
    super().__init__(cfg,env)
    self.cleared=torch.zeros((env.num_envs,3),device=env.device,dtype=torch.bool)
    self.highwater=torch.zeros(env.num_envs,device=env.device)
    self.anchor=torch.zeros_like(self.highwater);self.idle=torch.zeros_like(self.highwater)
    self.ready=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    self.previous_wall=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)

  def reset(self,env_ids=None):
    ids=slice(None) if env_ids is None else env_ids
    self.cleared[ids]=False;self.ready[ids]=False;self.idle[ids]=0.;self.previous_wall[ids]=-1

  def __call__(self,env,asset_cfg,approach_m=.75,grace_s=.75,ramp_s=1.5,
               max_rate=2.,min_progress_m=.02,min_command_mps=.1,
               root_clearance_m=.35,foot_clearance_m=.03):
    if approach_m<=0 or grace_s<0 or ramp_s<=0 or max_rate<0 or min_progress_m<=0:
      raise ValueError('Invalid wall waiting penalty parameters')
    command=env.command_manager.get_term('twist');root_x=command.local_xy()[:,0]
    foot_x=self.feet(env,asset_cfg)[:,:,0].amin(dim=1);rear=self.centres+WALL_HALF_THICKNESS
    clear=(root_x[:,None]>=rear+root_clearance_m)&(foot_x[:,None]>=rear+foot_clearance_m)
    self.cleared|=clear
    pending=self.active_walls(env)&~self.cleared
    wall=torch.where(pending,self.indices[None,:],3).amin(dim=1)
    safe=wall.clamp(max=2);near=pending.any(dim=1)&(root_x>=self.centres[safe]-approach_m)
    moving=command.command[:,:2].norm(dim=1)>=min_command_mps
    # The bottleneck also follows the hindmost foot, so stretching only the
    # front of the body cannot repeatedly refresh the waiting timer.
    progress=torch.minimum(root_x,foot_x+root_clearance_m-foot_clearance_m)
    changed=(wall!=self.previous_wall)|~self.ready
    previous=torch.where(changed,progress,self.highwater)
    self.highwater=torch.maximum(previous,progress)
    anchor=torch.where(changed,progress,self.anchor)
    advanced=self.highwater-anchor>=min_progress_m
    waiting=near&moving&~changed&~advanced
    self.idle=torch.where(waiting,self.idle+env.step_dt,torch.zeros_like(self.idle))
    self.anchor=torch.where(advanced|~near|~moving,self.highwater,anchor)
    self.previous_wall=wall;self.ready[:]=True
    return ((self.idle-grace_s)/ramp_s).clamp(0.,1.)*max_rate


def guard_wall_shaping_rewards(rewards,progress_reference_mps=.1,min_command_mps=.1):
  """Preserve profile weights/parameters, and wrap only positive base shaping."""
  shaping=('track_linear_velocity','track_angular_velocity','pose','foot_gait','feet_air_time')
  for name,term in rewards.items():
    if name in shaping:
      term.params={**term.params,'base_func':term.func,'progress_reference_mps':progress_reference_mps,
                   'min_command_mps':min_command_mps}
      term.func=WallProgressReward
