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
    self.turn_anchor = torch.zeros_like(self.highwater)
    self.turn_idle = torch.zeros_like(self.highwater)
    self.turn_ready = torch.zeros_like(self.ready)
    self.turn_waypoint = torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.ready[ids] = False
    self.idle[ids] = 0.
    self.turn_idle[ids] = 0.
    self.turn_ready[ids] = False
    self.turn_waypoint[ids] = -1

  def __call__(self, env, timeout_s=4., min_progress_m=.05, min_command_mps=.1,
               turn_timeout_s=0., turn_progress_rad=.05, turn_min_command_radps=.1,
               wall_grace_m=.9, wall_timeout_s=12.):
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
    # ★ 2026-11-19：墙前允许更长的停滞窗口（依次翻三墙时原地调整不算失败）。
    wall_timeout = torch.full_like(self.idle, float(timeout_s))
    if wall_timeout_s > 0 and wall_grace_m > 0:
      centres = getattr(command, 'wall_centres', None)
      local_xy = getattr(command, 'local_xy', None)
      if centres is not None and local_xy is not None and len(centres) > 0:
        x = local_xy()[:, 0]
        near = (x[:, None] - centres.to(x.device)[None, :]).abs().min(dim=1).values <= wall_grace_m
        wall_timeout = torch.where(near, torch.full_like(wall_timeout, float(wall_timeout_s)), wall_timeout)
    stalled=(self.idle >= wall_timeout - 1e-5) & moving
    if turn_timeout_s>0:
      if turn_progress_rad<=0 or turn_min_command_radps<=0:raise ValueError('Invalid rotation stall parameters')
      if not hasattr(command,'heading_error') or not hasattr(command,'waypoint_index'):
        raise ValueError('Rotation stall detection requires a directed passage command')
      turning=(~moving)&(command.command[:,2].abs()>=turn_min_command_radps)
      angle=command.heading_error.abs()
      changed=(command.waypoint_index!=self.turn_waypoint)|~self.turn_ready
      anchor=torch.where(changed,angle,self.turn_anchor)
      improved=(anchor-angle)>=turn_progress_rad
      self.turn_idle=torch.where(turning&~changed&~improved,self.turn_idle+env.step_dt,torch.zeros_like(self.turn_idle))
      self.turn_anchor=torch.where(changed|~turning|improved,angle,anchor)
      self.turn_waypoint[:]=command.waypoint_index
      self.turn_ready[:]=turning
      stalled|=turning&(self.turn_idle>=turn_timeout_s-1e-5)
    return stalled & ~passage.passage_success(env)


class WallGeometry(ManagerTermBase):
  """Wall centres/counts/heights of the high-wall lesson.

  ``LGDOG_WALL_LESSONS`` / the distillation single-lesson cfgs can trim the
  course so the high-wall lesson is absent.  Construction then must not fail:
  the geometry is emptied and every wall-specific term returns zero, while the
  rest of the task keeps training on the remaining lessons.
  """
  def __init__(self, cfg, env):
    super().__init__(env)
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    wall = generator.sub_terrains.get("high_wall")
    self.available = wall is not None
    self.indices = torch.arange(3, device=env.device)
    if wall is None:
      self.wall_columns = torch.zeros(generator.num_cols, dtype=torch.bool,
                                      device=env.device)
      # Empty geometry has no wall: centres are empty and the counts are zero.
      self.centres = torch.zeros(0, device=env.device)
      self.counts = torch.zeros((), dtype=torch.long, device=env.device)
      self.heights = torch.zeros(0, device=env.device)
      return
    if not isinstance(wall, WallSkillTerrainCfg) or generator.num_rows != 8:
      raise ValueError("WallTraining requires its eight-row auxiliary terrain")
    if tuple(generator.difficulty_range) != (0., 1.):
      raise ValueError("WallTraining requires difficulty_range=(0,1)")
    names = list(generator.sub_terrains)
    proportions = torch.tensor([c.proportion for c in generator.sub_terrains.values()], device=env.device)
    columns = torch.searchsorted((proportions/proportions.sum()).cumsum(0),
      torch.arange(generator.num_cols, device=env.device)/generator.num_cols+.001, right=True)
    self.wall_columns = columns == names.index("high_wall")
    self.centres = torch.tensor(getattr(wall,'wall_centres',WALL_CENTRES), device=env.device) - START_X
    self.counts = torch.tensor(wall.counts, device=env.device)
    self.heights = torch.tensor(wall.heights, device=env.device)

  def active_walls(self, env):
    terrain = env.scene.terrain
    if not self.available:
      return torch.zeros((len(terrain.terrain_types), len(self.indices)),
                         dtype=torch.bool, device=terrain.terrain_types.device)
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
    if not self.available:
      return torch.zeros(env.num_envs, device=env.device)
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
    if not self.available:
      return torch.zeros(env.num_envs, device=env.device)
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
    if not self.available:return value
    command=env.command_manager.get_term('twist');progress=command.route_state()[3]
    previous=torch.where(self.ready,self.highwater,progress)
    delta=(progress-previous).clamp_min(0.)
    self.highwater=torch.maximum(previous,progress);self.ready[:]=True
    moving=command.command[:,:2].norm(dim=1)>=min_command_mps
    gated=self.wall_columns[env.scene.terrain.terrain_types]&moving
    reference=progress_reference_mps
    if getattr(command.cfg,'obstacle_speed_limits',None) is not None:
      reference=torch.minimum(torch.full_like(delta,progress_reference_mps),
          command.command[:,:2].norm(dim=1).clamp_min(.001))
    factor=(delta/(env.step_dt*reference)).clamp(max=1.)
    factor=torch.where(gated,factor,torch.ones_like(factor))
    return torch.where(value>0,value*factor,value)


class WallApproachReward(WallGeometry):
  """★ 2026-11-20：世界系路点进展奖励（CMU Extreme Parkour 式速度投影）。

  旧实现按"到下一面墙的局部 x 距离"给正位移奖励：机器人横着走、甚至倒着走
  也能拿到分，训练诊断显示 91% 的回合在第一面墙前 1.25 m 停住。这里改为对
  "指向下一面未越过墙面中点"的速度投影做跟踪（墙面在世界 x 上固定，y 取车道
  中心 0）：

      d̂ = normalize((centre_x, 0) - local_xy)
      r = clamp(⟨v_world, d̂⟩, 0, v_cmd)          # m/s，奖励管理器再乘 dt

  只在下一面墙前方 ``max_distance_m`` 内生效；越过一面墙后自动改瞄下一面，三面
  全部越过（或当前地形等级没有剩余墙面 / 不在高墙列）返回 0。``v_cmd`` 是跟踪
  速度的上限，避免策略用超速冲刺刷分。
  """

  # 墙面中心仍在机器人"前方"的容差；越过 5 cm 后切换到下一面墙。
  CROSSED_TOLERANCE_M = .05

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    # 位置差分回退（速度接口不可用时）的上一帧局部 xy 与有效标志。
    self.previous_xy = torch.zeros((env.num_envs, 2), device=env.device)
    self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.previous_xy[ids] = self._env.command_manager.get_term("twist").local_xy()[ids]
    self.ready[ids] = False

  def _world_velocity(self, env, command, xy):
    """世界系平面速度；没有速度接口时用根位置（局部 xy）差分。"""
    entity = getattr(env.scene, 'robot', None)
    if entity is None:
      try:
        entity = env.scene['robot']
      except (KeyError, IndexError, TypeError):
        entity = None
    data = getattr(entity, 'data', None)
    velocity = getattr(data, 'root_link_lin_vel_w', None)
    if (torch.is_tensor(velocity) and velocity.dim() >= 2
        and velocity.shape[0] == xy.shape[0] and velocity.shape[1] >= 2):
      result = velocity[:, :2].to(device=xy.device, dtype=xy.dtype)
    else:
      # 局部 xy = 根世界位置 - env_origins；同一回合内平移量即世界系位移。
      step_dt = max(float(getattr(env, 'step_dt', 0.)), 1e-6)
      delta = xy - self.previous_xy.to(device=xy.device, dtype=xy.dtype)
      result = torch.where(self.ready.to(xy.device)[:, None], delta/step_dt,
                           torch.zeros_like(delta))
    self.previous_xy[:] = xy
    self.ready[:] = True
    return result

  def __call__(self, env, v_cmd=.6, max_distance_m=1.6, min_command_mps=.1):
    if not v_cmd > 0.:
      raise ValueError('v_cmd must be positive')
    if not max_distance_m > 0.:
      raise ValueError('max_distance_m must be positive')
    # 旧 reward 配置仍会传这个参数；新的速度投影不需要它。
    del min_command_mps
    if not self.available:
      return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_term("twist")
    xy = command.local_xy()
    zero = torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype)
    centres = getattr(command, 'wall_centres', None)
    if centres is None or len(centres) == 0:
      centres = self.centres
    if centres is None or len(centres) == 0:
      return zero
    centres = torch.as_tensor(centres, device=xy.device, dtype=xy.dtype).flatten()
    # 只考虑当前地形等级真正存在的墙面（非高墙列 active_walls 全 False）。
    active = self.active_walls(env)
    gaps = centres[None, :] - xy[:, 0:1]
    # 下一面未越过的墙：墙面中心还在机器人前方（含 5cm 容差）。
    candidates = active & (gaps > -self.CROSSED_TOLERANCE_M)
    # 指向墙面中点（车道中心 y=0）的单位向量。
    dx = gaps
    dy = -xy[:, 1:2]
    distance = (dx.square() + dy.square()).sqrt().clamp(min=1e-6)
    aimed = torch.where(candidates, distance, torch.full_like(distance, torch.inf))
    nearest = aimed.argmin(dim=1)
    nearest_distance = aimed.gather(1, nearest[:, None])[:, 0]
    direction = torch.stack((dx/distance, dy/distance), dim=-1)
    target_direction = torch.gather(
      direction, 1, nearest[:, None, None].expand(-1, 1, 2))[:, 0]
    world_velocity = self._world_velocity(env, command, xy)
    projection = (world_velocity*target_direction).sum(dim=-1)
    valid = candidates.any(dim=1) & (nearest_distance <= max_distance_m)
    return torch.where(valid, projection.clamp(min=0., max=float(v_cmd)), zero)


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
    if not self.available:
      return torch.zeros(env.num_envs, device=env.device)
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
