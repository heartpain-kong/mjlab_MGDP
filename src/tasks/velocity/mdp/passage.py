"""Directed passage objective and reset-time, outcome-gated curricula."""

from dataclasses import dataclass, asdict, field
import hashlib
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


DEFAULT_SPEED_LIMIT = 1.0


class PassageCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    generator = env.scene.terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    proportions = torch.tensor([c.proportion for c in generator.sub_terrains.values()],device=self.device)
    columns = torch.searchsorted((proportions/proportions.sum()).cumsum(0),
      torch.arange(generator.num_cols,device=self.device)/generator.num_cols+.001,right=True)
    routes = [cfg.waypoints.get(names[int(i)],((0.,0.),(cfg.goal_x,0.))) for i in columns]
    self.routes = torch.zeros((len(routes),max(map(len,routes)),2),device=self.device)
    self.last_waypoint = torch.tensor([len(r)-1 for r in routes],device=self.device)
    for i,route in enumerate(routes):
      if len(route)<2 or any(len(p)!=2 for p in route):
        raise ValueError("Passage routes need at least two xy points")
      self.routes[i,:len(route)] = torch.tensor(route,device=self.device)
      self.routes[i,len(route):] = self.routes[i,len(route)-1]
    lengths = torch.linalg.vector_norm(self.routes[:,1:]-self.routes[:,:-1],dim=-1)
    self.cumulative = torch.cat((torch.zeros((len(routes),1),device=self.device),lengths.cumsum(1)),1)
    self.waypoint_index = torch.ones(self.num_envs,device=self.device,dtype=torch.long)

  def route_state(self):
    types = self._env.scene.terrain.terrain_types
    index = self.waypoint_index
    start, target = self.routes[types,index-1], self.routes[types,index]
    delta = target-start
    length = torch.linalg.vector_norm(delta,dim=-1).clamp(min=1e-6)
    direction = delta/length[:,None]
    xy = self.local_xy()
    along = ((xy-start)*direction).sum(-1).clamp(min=0)
    projection = torch.minimum(along,length)
    lateral = torch.linalg.vector_norm(xy-start-projection[:,None]*direction,dim=-1)
    progress = self.cumulative[types,index-1]+projection
    return target, direction, lateral, progress

  def route_success(self):
    types = self._env.scene.terrain.terrain_types
    final = self.routes[types,self.last_waypoint[types]]
    return ((self.waypoint_index == self.last_waypoint[types]) &
            (torch.linalg.vector_norm(self.local_xy()-final,dim=-1) <= self.cfg.waypoint_radius))

  def local_xy(self):
    return self.robot.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2]

  def _resample_command(self, env_ids):
    self.speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.lin_vel_x)
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.heading_target[env_ids] = 0.
    self.waypoint_index[env_ids] = 1

  def _update_command(self):
    target, _, _, _ = self.route_state()
    types = self._env.scene.terrain.terrain_types
    reached = torch.linalg.vector_norm(self.local_xy()-target,dim=-1) <= self.cfg.waypoint_radius
    self.waypoint_index += (reached & (self.waypoint_index < self.last_waypoint[types])).long()
    target, direction, _, _ = self.route_state()
    offset = target-self.local_xy()
    distance = torch.linalg.vector_norm(offset,dim=-1).clamp(min=1e-6)
    velocity = offset/distance[:,None] * torch.minimum(self.speed,(1.5*distance).clamp(min=.12))[:,None]
    heading = self.robot.data.heading_w
    self.vel_command_b[:,0] = velocity[:,0]*heading.cos()+velocity[:,1]*heading.sin()
    self.vel_command_b[:,1] = -velocity[:,0]*heading.sin()+velocity[:,1]*heading.cos()
    self.heading_target = torch.atan2(direction[:,1],direction[:,0])
    error = self.heading_target-heading
    self.heading_error = torch.atan2(error.sin(),error.cos())
    self.vel_command_b[:,2] = (.8*self.heading_error).clamp(-.8,.8)


@dataclass(kw_only=True)
class PassageCommandCfg(UniformVelocityCommandCfg):
  goal_x: float = 5.3
  lane_half_width: float = .28
  waypoint_radius: float = .15
  waypoints: dict[str, tuple[tuple[float,float],...]] = field(default_factory=dict)

  def build(self, env):
    return PassageCommand(self, env)


def passage_success(env):
  term = env.command_manager.get_term("twist")
  if hasattr(term,"route_success"):
    return term.route_success()
  xy = term.local_xy()
  return (xy[:, 0] >= term.cfg.goal_x) & (xy[:, 1].abs() <= term.cfg.lane_half_width)


def left_lane(env):
  term = env.command_manager.get_term("twist")
  if hasattr(term,"route_state"):
    width=(term.lane_limits[env.scene.terrain.terrain_types]
           if hasattr(term,'lane_limits') else term.cfg.lane_half_width)
    return term.route_state()[2] > width
  xy = term.local_xy()
  return (xy[:, 1].abs() > term.cfg.lane_half_width) | (xy[:, 0] < -.5)


def failed(env):
  result = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for name, cfg in env.cfg.terminations.items():
    if cfg is not None and not cfg.time_out and name != "passage_success":
      result |= env.termination_manager.get_term(name)
  return result


def failure_cost(env):
  return failed(env).float()


def success_bonus(env):
  # Fixed terminal bonus after reward manager's dt scaling, paid once by done.
  return (passage_success(env) & ~failed(env)).float() / env.step_dt


class ForwardProgress(ManagerTermBase):
  """Reward only new route distance; backtracking cannot farm reward."""
  def __init__(self, cfg, env):
    super().__init__(env)
    self.highwater = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.highwater[ids] = self._env.command_manager.get_term("twist").local_xy()[ids, 0]

  def __call__(self, env):
    command = env.command_manager.get_term("twist")
    xy = command.local_xy()
    x = command.route_state()[3] if hasattr(command,"route_state") else xy[:,0].clamp(max=command.cfg.goal_x)
    progress = (x-self.highwater).clamp(min=0) / env.step_dt
    self.highwater = torch.maximum(self.highwater, x)
    return progress.clamp(max=1.) * (~left_lane(env) & ~failed(env)).float()


class PassageCurriculum(ManagerTermBase):
  """Advance terrain only on successful crossings; speed needs every column."""
  def __init__(self, cfg, env):
    super().__init__(env)
    self.terrain = env.scene.terrain
    self.command = env.command_manager.get_term("twist")
    cols = self.terrain.cfg.terrain_generator.num_cols
    self.count = torch.zeros(cols, device=self.device)
    self.success = torch.zeros_like(self.count)
    self.error = torch.zeros_like(self.count)
    self.streak = 0
    self.params = dict(cfg.params)
    self.column_names = list(self.terrain.cfg.terrain_generator.sub_terrains)
    generator = self.terrain.cfg.terrain_generator
    self.grid = (generator.num_rows, generator.num_cols)
    contract = (asdict(generator), env.step_dt, env.cfg.episode_length_s if hasattr(env.cfg,"episode_length_s") else None,
                asdict(self.command.cfg) if hasattr(self.command.cfg,"__dataclass_fields__") else
                (self.command.cfg.goal_x,self.command.cfg.lane_half_width))
    events = getattr(env.cfg, "events", {})
    if "himloco_pd_motor" in events:
      # Old mastery counts must not carry over to a different physics profile.
      randomization = tuple((name, events[name].params) for name in (
        "base_mass_inertia", "base_com", "foot_friction", "encoder_bias",
        "reset_robot_joints", "himloco_pd_motor", "himloco_push"))
      contract += (randomization, asdict(env.cfg.actions["joint_pos"]))
    self.fingerprint = hashlib.sha256(repr(contract).encode()).hexdigest()
    proportions = torch.tensor([getattr(c, "proportion", 1.)
      for c in generator.sub_terrains.values()], device=self.device)
    cumulative = (proportions/proportions.sum()).cumsum(0)
    self.kind_of_column = torch.searchsorted(cumulative,
      torch.arange(cols, device=self.device)/cols+.001, right=True)

  def __call__(self, env, env_ids, min_episodes=32, success_threshold=.7,
               max_error=.25, speed_limit=DEFAULT_SPEED_LIMIT):
    # Initial/randomized reset buffers are not completed episodes.
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=self.device)
    reset_mask = getattr(env, "reset_buf", None)
    ids = ids[:0] if reset_mask is None else ids[(env.episode_length_buf[ids] > 0) & reset_mask[ids]]
    # RSL-RL can randomize the first episode's length counter. Its artificial
    # timeout is not evidence that the robot failed a full passage attempt.
    actual_steps = env.metrics_manager._step_count
    ids = ids[actual_steps[ids] >= env.episode_length_buf[ids]]
    if len(ids):
      types = self.terrain.terrain_types[ids]
      passed = (passage_success(env) & ~failed(env))[ids]
      error_sum = env.metrics_manager._episode_sums["velocity_error_xy_mps"][ids]
      error = error_sum / env.episode_length_buf[ids].clamp(min=1)
      self.count.scatter_add_(0, types, torch.ones_like(error))
      self.success.scatter_add_(0, types, passed.float())
      self.error.scatter_add_(0, types, error)
      # Reaching the top must stay at the top, not wrap to a random row.
      level = self.terrain.terrain_levels[ids]
      last = self.terrain.max_terrain_level-1
      up = passed & (error <= max_error) & (level < last)
      down = ~passed
      self.terrain.update_env_origins(ids, up, down)
    rate = self.success / self.count.clamp(min=1)
    errors = self.error / self.count.clamp(min=1)
    represented = torch.bincount(self.kind_of_column, minlength=len(self.column_names)) > 0
    logs = {"speed_max": torch.tensor(self.command.cfg.ranges.lin_vel_x[1], device=self.device),
            "level_mean": self.terrain.terrain_levels.float().mean()}
    # Allocation is equal for the directed lessons; several columns may map to
    # one kind. Log reset-episode counts, rather than averaging per-step booleans.
    for i, name in enumerate(self.column_names):
      mask = self.kind_of_column == i
      n = self.count[mask].sum()
      logs[f"samples/{name}"] = n.clone()
      logs[f"success_rate/{name}"] = (self.success[mask].sum()/n.clamp(min=1)).clone()
    # Missing obstacle kinds/samples never count as passed.
    if represented.all() and (self.count >= min_episodes).all():
      good = (rate >= success_threshold).all() & (errors <= max_error).all()
      bad = (rate < .35).any() | (errors > .4).any()
      self.streak = self.streak + 1 if good else 0
      low, high = self.command.cfg.ranges.lin_vel_x
      if self.streak >= 3:
        high = min(speed_limit, high+.025)
        self.streak = 0
      elif bad:
        high = max(.35, high-.025)
      self.command.cfg.ranges.lin_vel_x = (low, high)
      self.count.zero_(); self.success.zero_(); self.error.zero_()
    return logs

  def state_dict(self):
    return {"version": 1, "params": self.params, "names": self.column_names, "grid": self.grid,
      "ranges": tuple(self.command.cfg.ranges.lin_vel_x), "streak": self.streak,
      "fingerprint": self.fingerprint,
      **{k: getattr(self,k).detach().cpu().clone() for k in ("count","success","error")},
      "levels": self.terrain.terrain_levels.detach().cpu().clone(),
      "types": self.terrain.terrain_types.detach().cpu().clone()}

  def load_state_dict(self, state):
    saved_params = state.get("params")
    # Increasing only the ceiling preserves the meaning of existing samples,
    # streaks and difficulty levels. Keep the remaining contract checks strict.
    legacy_ceiling = (self.params.get("speed_limit") == DEFAULT_SPEED_LIMIT
                      and saved_params == {**self.params, "speed_limit": .6})
    if (state.get("version") != 1 or (saved_params != self.params and not legacy_ceiling)
        or state.get("names") != self.column_names or tuple(state.get("grid", ())) != self.grid
        or state.get("fingerprint") != self.fingerprint):
      raise ValueError("Incompatible passage curriculum state")
    values = {}
    for key in ("count", "success", "error"):
      tensor = torch.as_tensor(state[key], device=self.device)
      if tensor.shape != self.count.shape or not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("Invalid passage statistics")
      values[key] = tensor
    if (values["success"] > values["count"]).any():
      raise ValueError("Invalid passage successes")
    levels = torch.as_tensor(state["levels"], device=self.device, dtype=torch.long)
    types = torch.as_tensor(state["types"], device=self.device, dtype=torch.long)
    if (levels.shape != self.terrain.terrain_levels.shape or types.shape != levels.shape
        or (levels < 0).any() or (levels >= self.terrain.max_terrain_level).any()
        or (types < 0).any() or (types >= len(self.count)).any()):
      raise ValueError("Passage resume requires the same environment/terrain grid")
    low, high = state["ranges"]
    saved_ceiling = saved_params.get("speed_limit", DEFAULT_SPEED_LIMIT)
    if not .25 <= low <= high <= saved_ceiling or not 0 <= int(state["streak"]) < 3:
      raise ValueError("Invalid passage speed range")
    for key, value in values.items():
      getattr(self,key).copy_(value)
    self.command.cfg.ranges.lin_vel_x = (low, high)
    self.streak = int(state["streak"])
    self.terrain.terrain_levels.copy_(levels); self.terrain.terrain_types.copy_(types)
    self.terrain.env_origins[:] = self.terrain.terrain_origins[levels, types]
