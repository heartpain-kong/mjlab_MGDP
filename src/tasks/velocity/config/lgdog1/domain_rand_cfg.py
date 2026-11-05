"""HIMLoco released-code full-strength profile for race MGDP."""

from dataclasses import dataclass, asdict, fields

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.himloco_randomization import (
  payload_mass_inertia, pd_motor_strength, initial_joint_positions, push_xy_velocity,
)
from src.tasks.velocity.mdp.himloco_actions import HIMLocoJointPositionActionCfg


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


HIMLOCO_DR = HIMLocoDomainRand()
HIMLOCO_REFERENCE_COMMIT = "ef289acaa62795009363b7b819c9186690630441"


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
  original = cfg.actions["joint_pos"]
  args = {f.name: getattr(original, f.name) for f in fields(original) if f.init}
  cfg.actions["joint_pos"] = HIMLocoJointPositionActionCfg(
    **args, max_delay_steps=cfg.decimation - 1, force_range=profile.disturbance_range,
    force_interval_steps=profile.disturbance_interval)


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
