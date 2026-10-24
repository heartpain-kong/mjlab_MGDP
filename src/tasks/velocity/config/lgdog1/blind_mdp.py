"""LgDog blind-terrain training terms; no terrain data enters the actor."""

from __future__ import annotations

import math

import torch


def clearance_cost(clearance, speed_xy, contact, valid, active, target_height):
  """Swing-only clearance cost; invalid ray hits contribute zero, never NaN."""
  usable = valid & ~contact & active[:, None]
  error = torch.where(usable, clearance - target_height, 0.0)
  return (error.abs() * speed_xy).sum(dim=1)


def foot_clearance_relative(env, sensor_names):
  """Vertical foot-site clearance, using one terrain-only ray per foot."""
  heights, valid = [], []
  for name in sensor_names:
    data = env.scene[name].data
    height = data.pos_w[:, 2] - data.hit_pos_w[:, 0, 2]
    hit = (data.distances[:, 0] >= 0) & torch.isfinite(height)
    heights.append(torch.where(hit, height, 0.0))
    valid.append(hit)
  return torch.stack(heights, dim=1), torch.stack(valid, dim=1)


def feet_clearance_terrain(
  env, sensor_names, asset_cfg, target_height=0.10,
  sensor_name="feet_ground_contact", command_name="twist", command_threshold=0.1,
):
  clearance, valid = foot_clearance_relative(env, sensor_names)
  robot = env.scene[asset_cfg.name]
  speed = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1)
  contact = env.scene[sensor_name].data.found > 0
  command = env.command_manager.get_command(command_name)
  active = command[:, :2].norm(dim=-1) + command[:, 2].abs() > command_threshold
  env.extras["log"]["Metrics/foot_clearance_ray_valid"] = valid.float().mean()
  return clearance_cost(clearance, speed, contact, valid, active, target_height)


def relative_foot_height(env, sensor_names):
  """Critic-only foot height; replace absolute world elevation."""
  return foot_clearance_relative(env, sensor_names)[0]


def commanded_distance(env, command_name="twist"):
  command = env.command_manager.get_command(command_name)
  speed = command[:, :2].norm(dim=-1)
  return torch.where(speed > 0.1, speed * env.step_dt, 0.0)


def tracking_progress(env, command_name="twist"):
  """Travel along each step's command, capped at requested distance.

  Sideways drift, walking backwards and overspeed cannot earn extra progress.
  Both vectors use the body frame, matching the existing tracking reward.
  """
  command = env.command_manager.get_command(command_name)[:, :2]
  speed = command.norm(dim=-1)
  velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  along = (velocity * command).sum(dim=-1) / speed.clamp_min(1e-6)
  along = torch.minimum(along.clamp_min(0.0), speed)
  return torch.where(speed > 0.1, along * env.step_dt, 0.0)


def velocity_error_xy(env, command_name="twist"):
  command = env.command_manager.get_command(command_name)
  return (command[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2]).norm(dim=-1)


def world_distance(env):
  return env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1) * env.step_dt


def outside_spawn_platform(env, radius=1.75):
  """Seconds spent beyond the central 3 m stair platform plus 0.25 m margin."""
  xy = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return (xy.abs().amax(dim=1) > radius).float() * env.step_dt


def terrain_decisions(
  expected, progress, exposure, elapsed, failed, episode_s,
  velocity_error=None, max_velocity_error=0.8,
):
  """Episode-level hysteresis with a velocity-quality gate.

  ``velocity_error`` is optional for compatibility with the generic unit
  tests and older callers.  LgDog blind training supplies the episode mean
  XY velocity error so a run cannot promote terrain solely by travelling far
  while tracking the command poorly.
  """
  eligible = expected >= 1.0
  ratio = progress / expected.clamp_min(1e-6)
  if velocity_error is None:
    tracking_ok = torch.ones_like(eligible, dtype=torch.bool)
  else:
    tracking_ok = velocity_error <= max_velocity_error
  up = (eligible & ~failed & (elapsed >= 0.9 * episode_s)
        & tracking_ok & (ratio >= 0.75) & (progress >= 2.0) & (exposure >= 1.0))
  down = eligible & (failed | (ratio < 0.35)) & ~up
  return up, down, ratio


def terrain_levels_tracking(env, env_ids, max_velocity_error=None):
  """Read completed-episode sums BEFORE mjlab 1.2 MetricsManager.reset()."""
  sums = env.metrics_manager._episode_sums
  expected = sums["commanded_distance"][env_ids]
  progress = sums["tracking_progress"][env_ids]
  exposure = sums["outside_spawn_platform"][env_ids]
  failed = env.termination_manager.terminated[env_ids]
  elapsed = env.episode_length_buf[env_ids] * env.step_dt
  # MetricsManager accumulates this term once per control step.  Divide by the
  # number of control steps to obtain the episode mean in m/s before applying
  # the promotion gate.  Dividing by elapsed seconds would multiply a per-step
  # m/s sum by 1 / step_dt (50x at the current 0.02 s control period).
  # Older snapshots may not contain the metric, so keep a safe fallback.
  velocity_error = None
  if "velocity_error_xy_mps" in sums:
    episode_steps = (elapsed / env.step_dt).clamp_min(1.0)
    velocity_error = sums["velocity_error_xy_mps"][env_ids] / episode_steps
  gate_error = velocity_error if max_velocity_error is not None else None
  up, down, ratio = terrain_decisions(
    expected, progress, exposure, elapsed, failed, env.max_episode_length_s,
    velocity_error=gate_error,
    max_velocity_error=0.8 if max_velocity_error is None else max_velocity_error,
  )
  terrain = env.scene.terrain
  terrain.update_env_origins(env_ids, up, down)
  eligible = expected >= 1.0
  count = eligible.sum().clamp_min(1)
  completed = elapsed > 0
  return {
    "level_mean": terrain.terrain_levels.float().mean(),
    "promote_fraction": up.float().sum() / count,
    "demote_fraction": down.float().sum() / count,
    "tracking_ratio": torch.where(eligible, ratio, 0.0).sum() / count,
    "velocity_error_xy_mps": (
      torch.where(eligible, velocity_error, torch.zeros_like(expected)).sum() / count
      if velocity_error is not None else torch.tensor(0.0, device=expected.device)
    ),
    "reset_batch_failure_fraction": (failed & completed).float().sum() / completed.sum().clamp_min(1),
    "episode_commanded_m": expected.mean(),
    "episode_progress_m": progress.mean(),
  }


class sustained_leg_contact:
  """Allow brief light leg scrapes; terminate sustained or severe impacts.

  Counter tracks the same geom across CONTROL steps. Force history still catches
  a severe impact during any physics substep. Trunk contact is a separate term.
  """

  def __init__(self, cfg, env):
    self.count = None
    self.steps = max(1, math.ceil(cfg.params.get("duration_s", 0.1) / env.step_dt))

  def reset(self, env_ids=None):
    if self.count is not None:
      self.count[slice(None) if env_ids is None else env_ids] = 0

  def __call__(self, env, sensor_name, force_threshold=10.0,
               severe_threshold=80.0, duration_s=0.1):
    data = env.scene[sensor_name].data
    # The configured sensor has history_length=4, one control interval.
    force = data.force_history.norm(dim=-1).amax(dim=-1)
    if self.count is None:
      self.count = torch.zeros_like(force, dtype=torch.long)
    self.count = torch.where(force > force_threshold, self.count + 1, 0)
    return ((self.count >= self.steps) | (force > severe_threshold)).any(dim=1)
