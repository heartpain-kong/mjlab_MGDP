"""HIMLoco released-code full-strength profile for race MGDP."""

from dataclasses import dataclass, asdict, field

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.actuator_reality import (
  ActuatorRealityJointPositionActionCfg, LGDOG1_NO_LOAD_SPEEDS, actuator_reality,
)
from src.tasks.velocity.mdp.himloco_randomization import (
  depth_camera_pose,
  payload_mass_inertia, pd_motor_strength, initial_joint_positions, push_xy_velocity,
)
from src.tasks.velocity.mdp.himloco_actions import HIMLocoJointPositionActionCfg


@dataclass(frozen=True)
class ActuatorRealityProfile:
  """Opt-in actuator reality shaping (paper #5, arXiv:2607.02205).

  All four non-idealities are switched on together by ``enabled``.  The default
  of ``False`` keeps the released HIMLoco profile unchanged (same action term,
  same event list, same random draws); enable it deliberately to train against
  actuator delay, a torque-speed envelope, a servo dead band and randomized
  joint friction/damping.

  Note: ``action_delay_steps`` stacks on top of the HIMLoco substep hold
  (``max_delay_steps = decimation - 1``); set that to 0 in
  ``configure_race_himloco_randomization`` if a single delay channel is wanted.
  """

  enabled: bool = False
  action_delay_steps: tuple[int, int] = (0, 2)
  """Inclusive transport-delay range k in physics steps, sampled per episode."""
  torque_speed_envelope: bool = True
  base_speed_fraction: float = .5
  """Fraction of the URDF no-load speed at which derating starts."""
  min_torque_factor: float = 0.
  no_load_speeds: dict[str, float] = field(
    default_factory=lambda: dict(LGDOG1_NO_LOAD_SPEEDS))
  target_deadband: float = .01
  """Joint-side dead band in rad."""
  frictionloss_scale_range: tuple[float, float] = (.8, 1.2)
  damping_scale_range: tuple[float, float] = (.8, 1.2)


@dataclass(frozen=True)
class HIMLocoDomainRand:
  payload_mass_range: tuple[float, float] = (-1., 2.)
  com_range: tuple[float, float] = (-.05, .05)
  friction_range: tuple[float, float] = (.2, 1.25)
  kp_range: tuple[float, float] = (.9, 1.1)
  kd_range: tuple[float, float] = (.9, 1.1)
  motor_strength_range: tuple[float, float] = (.9, 1.1)
  initial_joint_pos_range: tuple[float, float] = (.5, 1.5)
  disturbance_range: tuple[float, float] = (-30., 30.)
  disturbance_interval: int = 8
  push_interval_s: float = 16.
  max_push_vel_xy: float = 1.
  actuator_reality: ActuatorRealityProfile = field(default_factory=ActuatorRealityProfile)


HIMLOCO_DR = HIMLocoDomainRand()
HIMLOCO_REFERENCE_COMMIT = "ef289acaa62795009363b7b819c9186690630441"

# Fields carried over from the base JointPositionActionCfg when the released
# action term is rebuilt, independent of whether reality shaping is enabled.
_JOINT_ACTION_ARGS = (
  "entity_name", "clip", "transmission_type", "actuator_names", "scale", "offset",
  "preserve_order", "use_default_offset",
)


def configure_race_himloco_randomization(cfg):
  """Configure both policy training and race perception data collection."""
  profile = HIMLOCO_DR
  # Replace the base task's unrelated interval kick, including data collection.
  cfg.events.pop("push_robot", None)
  event = EventTermCfg(func=payload_mass_inertia, mode="startup", params={
    "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
    "payload_mass_range": profile.payload_mass_range,
  })
  cfg.events = {"base_mass_inertia": event, **cfg.events}
  cfg.events["base_com"].params["ranges"] = {i: profile.com_range for i in range(3)}
  cfg.events["foot_friction"].mode = "reset"
  cfg.events["foot_friction"].params["ranges"] = profile.friction_range
  cfg.events["reset_robot_joints"] = EventTermCfg(func=initial_joint_positions, mode="reset", params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    "position_range": profile.initial_joint_pos_range,
  })
  cfg.events["himloco_pd_motor"] = EventTermCfg(func=pd_motor_strength, mode="reset", params={
    "asset_cfg": SceneEntityCfg("robot"), "kp_range": profile.kp_range,
    "kd_range": profile.kd_range, "motor_strength_range": profile.motor_strength_range,
  })
  cfg.events["himloco_push"] = EventTermCfg(func=push_xy_velocity, mode="step", params={
    "asset_cfg": SceneEntityCfg("robot"), "interval_s": profile.push_interval_s,
    "max_velocity": profile.max_push_vel_xy,
  })
  # ★ 2026-11-19（对齐 WMP）：深度相机安装位姿随机化——位置 ±1 cm、俯仰 ±5°、yaw/roll ±2°。
  cfg.events["depth_camera_pose"] = EventTermCfg(func=depth_camera_pose, mode="reset", params={
    "asset_cfg": SceneEntityCfg("robot", site_names=("mgdp_depth",)),
    "pos_noise_m": .01, "pitch_noise_deg": 5., "yaw_roll_noise_deg": 2.,
  })
  original = cfg.actions["joint_pos"]
  args = {name: getattr(original, name) for name in _JOINT_ACTION_ARGS if hasattr(original, name)}
  release = dict(max_delay_steps=cfg.decimation - 1, force_range=profile.disturbance_range,
                 force_interval_steps=profile.disturbance_interval)
  reality = profile.actuator_reality
  if reality.enabled:
    # One unified actuator-reality event carries every per-episode actuator
    # non-ideality that lives in the physics model (friction/damping DR); the
    # delay, envelope and dead band live on the action term because they are
    # applied per physics substep.
    cfg.events["actuator_reality"] = EventTermCfg(func=actuator_reality, mode="reset", params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "frictionloss_scale_range": reality.frictionloss_scale_range,
      "damping_scale_range": reality.damping_scale_range,
    })
    cfg.actions["joint_pos"] = ActuatorRealityJointPositionActionCfg(
      **args, **release,
      action_delay_steps=reality.action_delay_steps,
      torque_speed_envelope=reality.torque_speed_envelope,
      base_speed_fraction=reality.base_speed_fraction,
      min_torque_factor=reality.min_torque_factor,
      no_load_speeds=dict(reality.no_load_speeds),
      target_deadband=reality.target_deadband,
    )
  else:
    cfg.actions["joint_pos"] = HIMLocoJointPositionActionCfg(**args, **release)


def himloco_randomization_metadata(cfg):
  if "himloco_pd_motor" not in cfg.events:
    return None
  return {
    "profile": "himloco_released_full", "upstream_commit": HIMLOCO_REFERENCE_COMMIT,
    **asdict(HIMLOCO_DR),
    "delay_physics_steps": (0, cfg.actions["joint_pos"].max_delay_steps),
    "physics_dt": cfg.sim.mujoco.timestep,
    "force_duration_s": cfg.sim.mujoco.timestep,
  }
