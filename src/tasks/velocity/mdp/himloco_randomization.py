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


def _scale_dof_field(env, ids, dofs, field, factors):
  """Write ``default[field][dofs] * factors`` back into a (possibly expanded) field.

  Expanded 1-D DOF fields have shape ``(num_envs, nv)``; a single-world sim skips
  ``expand_model_fields`` and keeps the bare ``(nv,)`` shape.
  """
  default = env.sim.get_default_field(field)[dofs]
  values = default[None, :] * factors
  model_field = getattr(env.sim.model, field)
  if model_field.dim() == 2:
    model_field[ids[:, None], dofs] = values
  elif model_field.dim() == 1:
    if len(ids) != 1:
      raise ValueError(
        f"'{field}' is not expanded per world but {len(ids)} env ids were given; "
        "register the actuator_reality event so expand_model_fields allocates it")
    model_field[dofs] = values[0]
  else:
    raise ValueError(f"Unexpected '{field}' shape {tuple(model_field.shape)}")


@requires_model_fields("dof_frictionloss", "dof_damping")
def joint_friction_damping_strength(
    env, env_ids, frictionloss_scale_range, damping_scale_range, asset_cfg):
  """Multiplicatively randomize joint friction loss and viscous damping.

  Independent factors are sampled per environment and joint, then applied to the
  compile-time joint defaults (never to the current values), so repeated calls do
  not compound.  The sampled factors are recorded for diagnostics.
  """
  ids = _ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  dofs = asset.indexing.joint_v_adr[asset_cfg.joint_ids]
  shape = (len(ids), len(dofs))
  friction = torch.empty(shape, device=env.device).uniform_(*frictionloss_scale_range)
  damping = torch.empty(shape, device=env.device).uniform_(*damping_scale_range)
  _scale_dof_field(env, ids, dofs, "dof_frictionloss", friction)
  _scale_dof_field(env, ids, dofs, "dof_damping", damping)
  _record(env, "frictionloss_scale", ids, friction)
  _record(env, "damping_scale", ids, damping)


@requires_model_fields("site_pos", "site_quat")
def depth_camera_pose(env, env_ids, asset_cfg, pos_noise_m=.01, pitch_noise_deg=5.,
                      yaw_roll_noise_deg=2.):
  """按 WMP 的做法随机深度相机安装位姿。

  WMP：位置 ±1 cm、俯仰 15~25°、yaw/roll ±2°。这里以名义姿态为基准，每个 episode
  叠加位置 ±pos_noise_m、俯仰 ±pitch_noise_deg、yaw/roll ±yaw_roll_noise_deg，
  直接写 per-world 的 site_pos/site_quat，射线方向随之改变。
  """
  import math
  import torch
  site_ids = asset_cfg.site_ids
  if site_ids is None or len(site_ids) == 0:
    raise ValueError("depth_camera_pose needs a site asset_cfg")
  site_ids = torch.as_tensor(site_ids, device=env.device, dtype=torch.long)
  base = getattr(env, "_depth_camera_base", None)
  if base is None:
    base = (env.sim.model.site_pos[:, site_ids].clone(),
            env.sim.model.site_quat[:, site_ids].clone())
    env._depth_camera_base = base
  ids = _ids(env, env_ids)
  n = ids.numel()
  dev = env.device
  dpos = (torch.rand(n, 1, 3, device=dev) * 2 - 1) * float(pos_noise_m)
  def small(deg, axis):
    a = (torch.rand(n, 1, device=dev) * 2 - 1) * math.radians(deg) * .5
    q = torch.zeros(n, 1, 4, device=dev)
    q[..., 0] = torch.cos(a)
    q[..., axis] = torch.sin(a)
    return q
  q = small(yaw_roll_noise_deg, 3) * small(pitch_noise_deg, 2) * small(yaw_roll_noise_deg, 1)
  bq = base[1][ids]
  w = q[..., 0] * bq[..., 0] - q[..., 1] * bq[..., 1] - q[..., 2] * bq[..., 2] - q[..., 3] * bq[..., 3]
  x = q[..., 0] * bq[..., 1] + q[..., 1] * bq[..., 0] + q[..., 2] * bq[..., 3] - q[..., 3] * bq[..., 2]
  y = q[..., 0] * bq[..., 2] - q[..., 1] * bq[..., 3] + q[..., 2] * bq[..., 0] + q[..., 3] * bq[..., 1]
  z = q[..., 0] * bq[..., 3] + q[..., 1] * bq[..., 2] - q[..., 2] * bq[..., 1] + q[..., 3] * bq[..., 0]
  env.sim.model.site_pos[ids[:, None], site_ids] = base[0][ids] + dpos
  env.sim.model.site_quat[ids[:, None], site_ids] = torch.stack([w, x, y, z], dim=-1)


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
