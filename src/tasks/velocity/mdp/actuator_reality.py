"""Actuator reality shaping for zero-shot sim-to-real (paper #5).

The kinematic PD servo that MuJoCo models ignores the actuator non-idealities
that dominate the sim-to-real gap on a geared joint: transport delay, the
torque-speed envelope, backlash/static friction dead bands and the hardware's
actual friction/damping.  This module adds them on top of the released HIMLoco
profile in four independently switchable pieces:

* control delay (``action_delay_steps``) -- the policy target enters a per-env
  ring buffer and leaves it ``k`` physics steps later, ``k`` drawn from the
  configured range once per episode;
* torque-speed envelope (``torque_speed_envelope``) -- the PD torque saturates
  at ``effort_limit * torque_speed_scale(|qd|)``: full torque below the motor
  base speed and a linear fall to ``min_torque_factor`` at the no-load speed;
* backlash dead band (``target_deadband``, rad) -- target deviations from the
  current joint position smaller than the dead band are ignored;
* joint friction/damping multiplicative randomization -- implemented in
  :func:`src.tasks.velocity.mdp.himloco_randomization.joint_friction_damping_strength`
  and wired through the unified :func:`actuator_reality` event below.

Everything is opt-in.  ``ActuatorRealityProfile.enabled`` defaults to ``False``
in ``domain_rand_cfg``; when disabled the task keeps the exact HIMLoco action
term and event list, so the released profile is bit-for-bit unchanged.

Reference: Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning,
arXiv:2607.02205.

Why the envelope clamps the *target* instead of ``actuator_forcerange``: mjlab
drives joints with MuJoCo ``<position>`` actuators, whose force is
``kp * (ctrl - q) - kd * qd`` clipped to the nominal effort range
(``mjlab/utils/spec.py:create_position_actuator``).  Solving that same equation
for the ctrl that produces the derated torque reproduces the envelope exactly
without touching a per-world model field, so the feature only relies on
``actuator_gainprm``/``actuator_biasprm``, which ``himloco_pd_motor`` already
expands per world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from src.tasks.velocity.mdp.himloco_actions import (
  HIMLocoJointPositionAction,
  HIMLocoJointPositionActionCfg,
)
from src.tasks.velocity.mdp.himloco_randomization import (
  joint_friction_damping_strength,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Motor no-load speeds measured at the joint output (24 V, see the header of
# ``src/assets/robots/lgdog1/LgDog1.urdf``): hip/thigh are direct drive, calf
# has a further 1:2 reduction.
LGDOG1_NO_LOAD_SPEEDS: dict[str, float] = {
  r".*_hip_joint": 18.85,
  r".*_thigh_joint": 18.85,
  r".*_calf_joint": 9.425,
}


def torque_speed_scale(
  velocity: torch.Tensor,
  base_speed: torch.Tensor | float,
  no_load_speed: torch.Tensor | float,
  min_scale: torch.Tensor | float = 0.0,
) -> torch.Tensor:
  """Linear torque derating above the motor base speed.

  ``|qd| <= base_speed`` keeps full torque.  Between the base speed and the
  no-load speed the available torque falls linearly, reaching
  ``min_scale * effort_limit`` at the no-load speed and staying there above it.
  DC motors have no torque right at the no-load speed, so ``min_scale = 0`` is
  the physically default value.

  Args:
    velocity: Joint velocity, any shape.
    base_speed: Speed where derating starts (the "base speed" of the motor,
      normally a fraction of ``no_load_speed``). Broadcastable to ``velocity``.
    no_load_speed: Speed where the torque reaches ``min_scale``.
    min_scale: Residual torque fraction beyond the no-load speed, in [0, 1].

  Returns:
    Torque scale in ``[min_scale, 1]``, same shape as ``velocity``.
  """
  speed = velocity.abs()
  base = torch.as_tensor(base_speed, dtype=speed.dtype, device=speed.device)
  no_load = torch.as_tensor(no_load_speed, dtype=speed.dtype, device=speed.device)
  floor = torch.as_tensor(min_scale, dtype=speed.dtype, device=speed.device)
  span = (no_load - base).clamp_min(1e-6)
  fall = (speed - base).clamp_min(0.0) * (1.0 - floor) / span
  return (1.0 - fall).clamp(min=floor, max=1.0)


def resolve_no_load_speeds(
  target_names: list[str] | tuple[str, ...],
  speed_patterns: dict[str, float],
  default_speed: float,
) -> torch.Tensor:
  """Resolve per-actuator no-load speeds from regex -> speed patterns."""
  if default_speed <= 0:
    raise ValueError(f"default_no_load_speed must be positive, got {default_speed}")
  speeds = torch.full((len(target_names),), float(default_speed), dtype=torch.float)
  for pattern, speed in speed_patterns.items():
    if speed <= 0:
      raise ValueError(f"No-load speed for {pattern!r} must be positive, got {speed}")
    for index, name in enumerate(target_names):
      if re.fullmatch(pattern, name):
        speeds[index] = float(speed)
  return speeds


def resolve_actuator_ids(entity: Entity, target_names: list[str] | tuple[str, ...]) -> torch.Tensor:
  """Model-global actuator ids ordered like the action's joint targets.

  mjlab's built-in position actuators carry the joint name, so the mapping is by
  name rather than by position.
  """
  actuator_names = list(entity.actuator_names)
  local_ids: list[int] = []
  for name in target_names:
    if name not in actuator_names:
      raise ValueError(
        f"No actuator named {name!r}; available actuators: {actuator_names}")
    local_ids.append(actuator_names.index(name))
  local = torch.tensor(local_ids, dtype=torch.long, device=entity.indexing.ctrl_ids.device)
  return entity.indexing.ctrl_ids[local]


class ActionDelayBuffer:
  """Per-environment ring buffer that delays action targets by physics steps.

  ``capacity = max_delay + 1`` slots are pushed once per substep; ``delayed(k)``
  returns the value pushed ``k`` substeps ago.  Every environment advances its
  cursor in lockstep (mjlab steps all worlds together), so a single write cursor
  is enough; the read slot is per-environment because each env samples its own
  delay.
  """

  def __init__(self, num_envs: int, max_delay: int, dim: int, device: str):
    if max_delay < 0:
      raise ValueError(f"max_delay must be non-negative, got {max_delay}")
    self.capacity = max_delay + 1
    self.buffer = torch.zeros((num_envs, self.capacity, dim), device=device)
    self.index = 0

  def reset(self, value: torch.Tensor | float, env_ids: torch.Tensor | slice | None = None) -> None:
    """Fill every slot of the selected environments with ``value``."""
    ids = slice(None) if env_ids is None else env_ids
    if torch.is_tensor(value):
      value = value.unsqueeze(1)  # broadcast over the ring capacity
    self.buffer[ids] = value

  def push(self, value: torch.Tensor) -> None:
    """Write the current target and advance the shared cursor."""
    self.buffer[:, self.index] = value
    self.index = (self.index + 1) % self.capacity

  def delayed(self, delay: torch.Tensor | int) -> torch.Tensor:
    """Return the target pushed ``delay`` steps ago for every environment."""
    slots = (self.index - 1 - torch.as_tensor(
      delay, dtype=torch.long, device=self.buffer.device)) % self.capacity
    if slots.ndim == 0:
      slots = slots.expand(self.buffer.shape[0])
    envs = torch.arange(self.buffer.shape[0], device=self.buffer.device)
    return self.buffer[envs, slots]


@dataclass(kw_only=True)
class ActuatorRealityJointPositionActionCfg(HIMLocoJointPositionActionCfg):
  """HIMLoco joint-position action with optional actuator reality shaping."""

  action_delay_steps: tuple[int, int] = (0, 0)
  """Inclusive range the transport delay k (physics steps) is sampled from.
  ``(0, 0)`` disables the delay.  It stacks on top of the HIMLoco substep hold
  (``max_delay_steps``); set that to 0 for a single delay channel."""

  torque_speed_envelope: bool = False
  """Whether PD output is limited by the motor torque-speed envelope."""

  base_speed_fraction: float = 0.5
  """Fraction of the no-load speed at which torque derating starts."""

  min_torque_factor: float = 0.0
  """Residual torque fraction beyond the no-load speed, in [0, 1]."""

  default_no_load_speed: float = 18.85
  """No-load speed used for joints that no pattern in ``no_load_speeds`` matches."""

  no_load_speeds: dict[str, float] = field(default_factory=dict)
  """Joint-name regex -> motor no-load speed mapped to the joint output."""

  target_deadband: float = 0.0
  """Joint-side dead band in rad; target errors smaller than this are ignored.
  ``0`` disables it."""

  def build(self, env: ManagerBasedRlEnv) -> ActuatorRealityJointPositionAction:
    return ActuatorRealityJointPositionAction(self, env)


class ActuatorRealityJointPositionAction(HIMLocoJointPositionAction):
  """Joint-position action with delay, torque-speed envelope and dead band.

  ``apply_actions`` mirrors :meth:`HIMLocoJointPositionAction.apply_actions`
  because that method has no hook between target computation and the entity
  write; keep the two in sync if the released action changes.
  """

  cfg: ActuatorRealityJointPositionActionCfg

  def __init__(self, cfg: ActuatorRealityJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    low, high = (int(step) for step in cfg.action_delay_steps)
    if not 0 <= low <= high:
      raise ValueError(
        f"action_delay_steps must satisfy 0 <= low <= high, got {cfg.action_delay_steps}")
    self._delay_range = (low, high)
    self.transport_delay_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device)
    self._delay_buffer = (
      ActionDelayBuffer(self.num_envs, high, self.action_dim, self.device)
      if high > 0 else None)
    self._deadband = float(cfg.target_deadband)
    if self._deadband < 0:
      raise ValueError(f"target_deadband must be non-negative, got {cfg.target_deadband}")
    self._envelope_enabled = bool(cfg.torque_speed_envelope)
    if self._envelope_enabled:
      if not 0.0 < float(cfg.base_speed_fraction) <= 1.0:
        raise ValueError(
          "base_speed_fraction must be in (0, 1], got "
          f"{cfg.base_speed_fraction}")
      if not 0.0 <= float(cfg.min_torque_factor) <= 1.0:
        raise ValueError(
          "min_torque_factor must be in [0, 1], got "
          f"{cfg.min_torque_factor}")
      self._actuator_ids = resolve_actuator_ids(self._entity, self._target_names)
      no_load = resolve_no_load_speeds(
        self._target_names, cfg.no_load_speeds, cfg.default_no_load_speed)
      self._no_load_speed = no_load.to(self.device)
      self._base_speed = float(cfg.base_speed_fraction) * self._no_load_speed
      self._min_torque_factor = float(cfg.min_torque_factor)
      effort = env.sim.get_default_field("actuator_forcerange")[self._actuator_ids]
      self._effort_limit = effort.abs().amax(dim=-1).to(self.device)
    # No RNG here: the first ``reset`` samples the per-episode delay, so building
    # the action term does not shift the global RNG stream.
    self._fill_delay_buffer(slice(None))

  # Actuator reality pipeline.

  def _sample_delay(self, ids: torch.Tensor | slice) -> None:
    low, high = self._delay_range
    if high == 0:
      self.transport_delay_steps[ids] = 0
      return
    count = self.num_envs if isinstance(ids, slice) else len(ids)
    self.transport_delay_steps[ids] = torch.randint(
      low, high + 1, (count,), device=self.device)

  def _reset_reality_state(self, ids: torch.Tensor | slice) -> None:
    """Resample this episode's delay and refill the ring buffer with the target."""
    self._sample_delay(ids)
    self._fill_delay_buffer(ids)

  def _fill_delay_buffer(self, ids: torch.Tensor | slice) -> None:
    """Fill every ring slot with the current target so a fresh episode is sane."""
    if self._delay_buffer is None:
      return
    bias = self._entity.data.encoder_bias[ids][:, self._target_ids]
    if torch.is_tensor(self._offset):
      default = self._offset[ids]
    else:
      default = torch.full_like(bias, float(self._offset))
    self._delay_buffer.reset(default - bias, ids)

  def _envelope_targets(self, targets: torch.Tensor) -> torch.Tensor:
    """Rewrite the PD target so the produced torque respects the envelope."""
    current = self._entity.data.joint_pos[:, self._target_ids]
    velocity = self._entity.data.joint_vel[:, self._target_ids]
    scale = torque_speed_scale(
      velocity, self._base_speed, self._no_load_speed, self._min_torque_factor)
    limit = self._effort_limit * scale
    gain = self._env.sim.model.actuator_gainprm
    bias = self._env.sim.model.actuator_biasprm
    if gain.dim() == 3:
      kp = gain[:, self._actuator_ids, 0]
      kd = -bias[:, self._actuator_ids, 2]
    else:
      # ``sim.expand_model_fields`` is skipped for a single world.
      kp = gain[self._actuator_ids, 0].unsqueeze(0).expand(self.num_envs, -1)
      kd = -bias[self._actuator_ids, 2].unsqueeze(0).expand(self.num_envs, -1)
    torque = kp * (targets - current) - kd * velocity
    limited = torch.clamp(torque, -limit, limit)
    # kp*(ctrl-q) - kd*qd = limited  =>  ctrl = q + (limited + kd*qd) / kp
    return current + (limited + kd * velocity) / kp.clamp_min(1e-6)

  def _shape_targets(self, targets: torch.Tensor) -> torch.Tensor:
    """Delay -> dead band -> torque-speed envelope, in that physical order."""
    if self._delay_buffer is not None:
      self._delay_buffer.push(targets)
      targets = self._delay_buffer.delayed(self.transport_delay_steps)
    if self._deadband > 0:
      current = self._entity.data.joint_pos[:, self._target_ids]
      targets = torch.where(
        (targets - current).abs() < self._deadband, current, targets)
    if self._envelope_enabled:
      targets = self._envelope_targets(targets)
    return targets

  # Action term interface.

  def apply_actions(self) -> None:
    # Mirrors HIMLocoJointPositionAction.apply_actions (no hook available there).
    targets = torch.where(
      (self.physics_phase < self.delay_steps)[:, None],
      self.previous_targets, self._processed_actions)
    targets = targets - self._entity.data.encoder_bias[:, self._target_ids]
    targets = self._shape_targets(targets)
    self._entity.set_joint_position_target(targets, joint_ids=self._target_ids)
    if self.physics_phase == 0:
      self.last_force_world[:] = quat_apply(
        self._entity.data.root_link_quat_w, self.local_force[:, 0])[:, None, :]
    else:
      self.last_force_world.zero_()
    # MuJoCo retains xfrc_applied. Clear it after one physics substep.
    self._entity.write_external_wrench_to_sim(
      self.last_force_world, torch.zeros_like(self.last_force_world),
      body_ids=self.force_body_ids)
    self.physics_phase += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    super().reset(ids)
    self._reset_reality_state(ids)


@requires_model_fields("dof_frictionloss", "dof_damping")
def actuator_reality(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  frictionloss_scale_range: tuple[float, float],
  damping_scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Unified ``actuator_reality`` event: multiplicative friction/damping DR.

  The event also declares ``dof_frictionloss``/``dof_damping`` as per-world model
  fields, which is what allows every environment to carry its own factors.  The
  delay, envelope and dead band settings live on the action term cfg because they
  are applied per physics substep inside ``apply_actions``.
  """
  joint_friction_damping_strength(
    env, env_ids, frictionloss_scale_range, damping_scale_range, asset_cfg)
