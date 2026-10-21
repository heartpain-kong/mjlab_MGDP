"""Experimental LgDog blind terrain task with the existing 47 -> 12 interface."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity import mdp

from . import blind_mdp
from .env_cfgs import BASE_BODY, FOOT_GEOMS, FOOT_SITES, lgdog1_rough_env_cfg
from .reward_cfg import make_lgdog_rewards


@dataclass
class ElevatedFootRayPatternCfg(GridPatternCfg):
  """Start rays above the foot so small penetration does not cause a miss."""

  start_height: float = 0.5

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.start_height
    return offsets, directions


def lgdog1_blind_terrain_env_cfg(
  play: bool = False, *, reward_profile="terrain_v4",
  gentle_terrain: bool = True, critic_height_scan: bool = True,
):
  """Terrain training mechanics, with independent terrain/critic scan choices.

  Defaults retain the original TerrainV4/V7 distribution. Standard HIM uses the
  unmodified rough generator and removes the body grid scan from both networks.
  Four foot rays remain simulation-only labels for reward and critic clearance.
  """
  if reward_profile not in {"terrain_v4", "him", "him_standard"}:
    raise ValueError("Blind terrain tasks require terrain_v4, him or him_standard rewards")
  cfg = deepcopy(lgdog1_rough_env_cfg(
    play=play, use_height_scan=critic_height_scan, gentle_terrain=gentle_terrain
  ))
  # Actor is always blind. The optional grid scan is privileged critic data only.
  cfg.observations["actor"].terms.pop("height_scan", None)
  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      # LgDog visual meshes are group 1; collisions are group 3. Terrain is 0.
      sensor.include_geom_groups = (0,)
      sensor.debug_vis = False

  foot_rays = tuple(
    RayCastSensorCfg(
      name=f"{site}_ground_ray",
      frame=ObjRef(type="site", name=site, entity="robot"),
      pattern=ElevatedFootRayPatternCfg(size=(0.0, 0.0), resolution=0.1),
      ray_alignment="world", include_geom_groups=(0,), max_distance=3.0,
    )
    for site in FOOT_SITES
  )
  ray_names = tuple(sensor.name for sensor in foot_rays)
  trunk = ContactSensorCfg(
    name="trunk_ground_contact",
    primary=ContactMatch(mode="geom", pattern="trunk_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4,
  )
  legs = ContactSensorCfg(
    name="legs_ground_contact",
    primary=ContactMatch(mode="geom", pattern=r".*_collision\d*$", entity="robot",
                         exclude=(*FOOT_GEOMS, "trunk_collision")),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4,
  )
  cfg.scene.sensors = tuple(
    sensor for sensor in cfg.scene.sensors if sensor.name != "nonfoot_ground_touch"
  ) + (*foot_rays, trunk, legs)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": trunk.name, "force_threshold": 10.0},
  )
  cfg.terminations["sustained_leg_contact"] = TerminationTermCfg(
    func=blind_mdp.sustained_leg_contact,
    params={"sensor_name": legs.name, "force_threshold": 10.0,
            "severe_threshold": 80.0, "duration_s": 0.1},
  )
  cfg.rewards = make_lgdog_rewards(
    reward_profile, base_body=BASE_BODY, foot_sites=FOOT_SITES, terrain_ray_names=ray_names,
  )
  foot_height = cfg.observations["critic"].terms["foot_height"]
  foot_height.func = blind_mdp.relative_foot_height
  foot_height.params = {"sensor_names": ray_names}

  cfg.metrics.update({
    "traveled_distance": MetricsTermCfg(func=blind_mdp.world_distance),
    "commanded_distance": MetricsTermCfg(func=blind_mdp.commanded_distance),
    "tracking_progress": MetricsTermCfg(func=blind_mdp.tracking_progress),
    "outside_spawn_platform": MetricsTermCfg(func=blind_mdp.outside_spawn_platform),
    "velocity_error_xy_mps": MetricsTermCfg(func=blind_mdp.velocity_error_xy),
  })
  # Keep speed modest while testing terrain fixes; enlarge only after evaluation.
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (-0.4, 0.8)
  twist.ranges.lin_vel_y = (-0.3, 0.3)
  twist.ranges.ang_vel_z = (-0.6, 0.6)
  if not play:
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=blind_mdp.terrain_levels_tracking,
      # The standard Rough + HIM run gets the additional velocity-quality gate.
      # Keep legacy TerrainV4/V7 promotion behavior unchanged for checkpoint
      # comparisons.
      params={"max_velocity_error": 0.8} if reward_profile == "him_standard" else {},
    )
  return cfg
