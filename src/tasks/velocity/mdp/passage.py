"""Directed passage objective and reset-time, outcome-gated curricula."""

from dataclasses import dataclass, asdict
import hashlib
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


class PassageCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.speed = torch.zeros(self.num_envs, device=self.device)

  def local_xy(self):
    return self.robot.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2]

  def _resample_command(self, env_ids):
    self.speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.ranges.lin_vel_x)
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.heading_target[env_ids] = 0.

  def _update_command(self):
    heading = self.robot.data.heading_w
    # Goal direction is world +X; convert the gentle centering velocity to body.
    vy = (-.5*self.local_xy()[:, 1]).clamp(-.1, .1)
    self.vel_command_b[:, 0] = self.speed*heading.cos() + vy*heading.sin()
    self.vel_command_b[:, 1] = -self.speed*heading.sin() + vy*heading.cos()
    self.heading_error = torch.atan2(-heading.sin(), heading.cos())
    self.vel_command_b[:, 2] = (.8*self.heading_error).clamp(-.4, .4)


@dataclass(kw_only=True)
class PassageCommandCfg(UniformVelocityCommandCfg):
  goal_x: float = 5.3
  lane_half_width: float = .28

  def build(self, env):
    return PassageCommand(self, env)


def passage_success(env):
  term = env.command_manager.get_term("twist")
  xy = term.local_xy()
  return (xy[:, 0] >= term.cfg.goal_x) & (xy[:, 1].abs() <= term.cfg.lane_half_width)


def left_lane(env):
  term = env.command_manager.get_term("twist")
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
  """Reward only newly attained +X positions; backtracking cannot farm reward."""
  def __init__(self, cfg, env):
    super().__init__(env)
    self.highwater = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.highwater[ids] = self._env.command_manager.get_term("twist").local_xy()[ids, 0]

  def __call__(self, env):
    command = env.command_manager.get_term("twist")
    xy = command.local_xy()
    x = xy[:, 0].clamp(max=command.cfg.goal_x)
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
    contract = (asdict(generator), env.step_dt, self.command.cfg.goal_x,
                self.command.cfg.lane_half_width)
    self.fingerprint = hashlib.sha256(repr(contract).encode()).hexdigest()
    proportions = torch.tensor([getattr(c, "proportion", 1.)
      for c in generator.sub_terrains.values()], device=self.device)
    cumulative = (proportions/proportions.sum()).cumsum(0)
    self.kind_of_column = torch.searchsorted(cumulative,
      torch.arange(cols, device=self.device)/cols+.001, right=True)

  def __call__(self, env, env_ids, min_episodes=32, success_threshold=.7,
               max_error=.25, speed_limit=.6):
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
    if (state.get("version") != 1 or state.get("params") != self.params
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
    if not .25 <= low <= high <= self.params.get("speed_limit", .6) or not 0 <= int(state["streak"]) < 3:
      raise ValueError("Invalid passage speed range")
    for key, value in values.items():
      getattr(self,key).copy_(value)
    self.command.cfg.ranges.lin_vel_x = (low, high)
    self.streak = int(state["streak"])
    self.terrain.terrain_levels.copy_(levels); self.terrain.terrain_types.copy_(types)
    self.terrain.env_origins[:] = self.terrain.terrain_origins[levels, types]
