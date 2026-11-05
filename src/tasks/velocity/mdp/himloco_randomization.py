"""HIMLoco released-code domain randomization adapted to mjlab/LgDog.

Reference: InternRobotics/HIMLoco, ef289acaa62795009363b7b819c9186690630441.
Torque gain is implemented explicitly; the released code samples a motor
factor but does not multiply it into _compute_torques.
"""

import math

import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel


def _ids(env, env_ids):
  return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids


def _record(env, name, ids, value):
  if not hasattr(env, "_himloco_samples"):
    env._himloco_samples = {}
  if name not in env._himloco_samples:
    env._himloco_samples[name] = torch.zeros(
      (env.num_envs, *value.shape[1:]), device=env.device)
  env._himloco_samples[name][ids] = value


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def payload_mass_inertia(env, env_ids, payload_mass_range, asset_cfg):
  """Uniform additive payload; density-scale inertia to retain physical validity."""
  ids = _ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  bodies = asset.indexing.body_ids[asset_cfg.body_ids]
  nominal = env.sim.get_default_field("body_mass")[bodies]
  if float(nominal.min()) + payload_mass_range[0] <= 0:
    raise ValueError("Payload range would make a selected body mass non-positive")
  payload = torch.empty((len(ids), len(bodies)), device=env.device).uniform_(*payload_mass_range)
  mass = nominal[None, :] + payload
  factor = mass / nominal[None, :]
  env.sim.model.body_mass[ids[:, None], bodies] = mass
  env.sim.model.body_inertia[ids[:, None], bodies] = (
    env.sim.get_default_field("body_inertia")[bodies][None, :, :] * factor[:, :, None])
  _record(env, "payload_kg", ids, payload)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_motor_strength(env, env_ids, kp_range, kd_range, motor_strength_range, asset_cfg):
  """Sample scalar Kp/Kd/motor factors shared across each robot's joints.

  Scaling both position-actuator gains by motor strength scales unsaturated
  PD torque. The actuator force limits remain unchanged.
  """
  ids = _ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  controls = asset.indexing.ctrl_ids
  assert controls is not None
  kp = torch.empty((len(ids), 1), device=env.device).uniform_(*kp_range)
  kd = torch.empty_like(kp).uniform_(*kd_range)
  strength = torch.empty_like(kp).uniform_(*motor_strength_range)
  gain = env.sim.get_default_field("actuator_gainprm")[controls]
  bias = env.sim.get_default_field("actuator_biasprm")[controls]
  env.sim.model.actuator_gainprm[ids[:, None], controls, 0] = gain[None, :, 0] * kp * strength
  env.sim.model.actuator_biasprm[ids[:, None], controls, 1] = bias[None, :, 1] * kp * strength
  env.sim.model.actuator_biasprm[ids[:, None], controls, 2] = bias[None, :, 2] * kd * strength
  for name, value in (("kp_factor", kp), ("kd_factor", kd), ("motor_strength", strength)):
    _record(env, name, ids, value)


def initial_joint_positions(env, env_ids, position_range, asset_cfg):
  ids = _ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  joints = asset_cfg.joint_ids
  nominal = asset.data.default_joint_pos[ids][:, joints]
  positions = nominal * torch.empty_like(nominal).uniform_(*position_range)
  limits = asset.data.soft_joint_pos_limits[ids][:, joints]
  positions = positions.clamp(limits[..., 0], limits[..., 1])
  asset.write_joint_state_to_sim(
    positions, torch.zeros_like(positions), env_ids=ids, joint_ids=joints)


def push_xy_velocity(env, env_ids, interval_s, max_velocity, asset_cfg):
  """Match HIMLoco's global periodic absolute XY velocity replacement."""
  period = max(1, math.ceil(interval_s / env.step_dt))
  if env.common_step_counter <= 0 or env.common_step_counter % period:
    return
  ids = _ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  velocity = asset.data.root_link_vel_w[ids].clone()
  velocity[:, :2] = torch.empty_like(velocity[:, :2]).uniform_(-max_velocity, max_velocity)
  asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)
  _record(env, "push_velocity", ids, velocity)
