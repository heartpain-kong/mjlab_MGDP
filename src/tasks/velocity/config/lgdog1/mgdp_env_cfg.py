"""LgDog1 + MGDP perception task configuration (mjlab port).

This is the mjlab counterpart of MGDP's "stage 2" controller: a 47-dim
proprioception history plus a 64-dim depth feature produced by a pretrained
contrastive perception model.

Pipeline
--------
    RayCastSensor (16x16 pinhole, 87 deg FOV, horizontal forward)
        -> normalised depth in [0, 1] (metres / depth_max)
        -> two-frame stack
        -> MGPDPerception (U-Net encoder + GRU)  [frozen, pretrained]
        -> 64-dim "perception" observation group
    actor  = 4 frames x 47 proprio  +  64 perception
    critic = proprio + privileged height scan / foot info (no perception)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers import TerminationTermCfg
from src.tasks.velocity import mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg

from src.assets.robots.lgdog1.lgdog1_constants import get_lgdog1_robot_cfg, get_spec
from src.tasks.velocity.config.lgdog1.env_cfgs import BASE_BODY, FOOT_GEOMS, FOOT_SITES, lgdog1_rough_env_cfg
from src.tasks.velocity.config.lgdog1.race_depth_terrain_cfg import (
    lgdog1_race_depth_terrain_cfg,
)
from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerceptionCfg
from src.tasks.velocity.mdp.curriculums import RewardBasedVelocityCurriculum
from src.tasks.velocity.mdp import passage
from . import blind_mdp
from .reward_cfg import make_lgdog_rewards, make_mgdp_passage_rewards
from .passage_terrain_cfg import lgdog1_passage_terrain_cfg, lesson_route, GOAL_X, LANE_HALF_WIDTH
from .domain_rand_cfg import add_race_mass_inertia_randomization

DEPTH_SITE = "mgdp_depth"
DEPTH_HW = (16, 16)
DEPTH_FOVY = 87.0
DEPTH_MAX = 5.0
PROPRIO_HISTORY = 4

# 前头顶安装，站立不动时光轴水平朝前（+X），与地面平行。
# MuJoCo 相机沿 -Z 看，绕 Y 轴 -90° 即可把光轴转到 +X。
DEPTH_SITE_POS = (0.02, 0.0, 0.075)
# Keep the trained depth tensor's original pixel ordering. The RGB display
# uses an upright camera at the same mount and optical axis instead.
DEPTH_SITE_QUAT = (0.7071067812, 0.0, -0.7071067812, 0.0)

DEFAULT_CHECKPOINT = str(
    Path("logs") / "mgdp" / "perception" / "mgdp_perception.pt"
)
DEFAULT_RACE_CHECKPOINT = str(Path("logs") / "mgdp" / "perception" / "mgdp_race_v3.pt")


def _mgdp_robot_spec():
    """Robot spec with the horizontal forward depth site added to the trunk."""
    spec = get_spec()
    body = spec.body(BASE_BODY)
    if not any(site.name == DEPTH_SITE for site in body.sites):
        body.add_site(name=DEPTH_SITE, pos=DEPTH_SITE_POS, quat=DEPTH_SITE_QUAT)
    return spec


def _mgdp_robot_cfg():
    return replace(get_lgdog1_robot_cfg(), spec_fn=_mgdp_robot_spec)


def lgdog1_mgdp_env_cfg(
    play: bool = False,
    checkpoint: str = DEFAULT_CHECKPOINT,
    with_perception: bool = True,
    proprio_history: int = PROPRIO_HISTORY,
    depth_hw: tuple[int, int] = DEPTH_HW,
):
    """Build the MGDP velocity task env config."""
    cfg = lgdog1_rough_env_cfg(play=play)

    # Robot + depth camera site.
    cfg.scene.entities = {"robot": _mgdp_robot_cfg()}

    # The actor no longer sees the raw 187-dim height scan; the critic keeps it
    # as privileged information (same asymmetry as MGDP).
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["actor"].history_length = proprio_history
    cfg.observations["actor"].flatten_history_dim = True
    cfg.observations["actor"].concatenate_dim = -1

    # Depth rays: a pinhole camera made of rays (cheap at thousands of envs).
    depth_sensor = RayCastSensorCfg(
        name="depth_rays",
        frame=ObjRef(type="site", name=DEPTH_SITE, entity="robot"),
        ray_alignment="base",
        pattern=PinholeCameraPatternCfg(
            width=depth_hw[1], height=depth_hw[0], fovy=DEPTH_FOVY
        ),
        max_distance=DEPTH_MAX,
        exclude_parent_body=True,
        debug_vis=False,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (depth_sensor,)

    perception_cfg = MGPDPerceptionCfg(
        depth_hw=depth_hw,
        height_hw=(17, 11),
        depth_max=DEPTH_MAX,
    )

    if with_perception:
        cfg.observations["perception"] = ObservationGroupCfg(
            terms={
                "depth_feature": ObservationTermCfg(
                    func=mgdp_depth_feature,
                    params={
                        "sensor_name": "depth_rays",
                        "cfg": perception_cfg,
                        "checkpoint": checkpoint,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        )

    # Critic keeps one frame of the privileged set.
    cfg.observations["critic"].history_length = 1
    cfg.observations["critic"].flatten_history_dim = True
    _configure_mgdp_training(cfg, play)
    return cfg


def _configure_mgdp_training(cfg, play):
    trunk = ContactSensorCfg(name="trunk_ground_contact",
      primary=ContactMatch(mode="geom", pattern="trunk_collision", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)
    legs = ContactSensorCfg(name="legs_ground_contact",
      primary=ContactMatch(mode="geom", pattern=r".*_collision\d*$", entity="robot",
                           exclude=(*FOOT_GEOMS, "trunk_collision")),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "nonfoot_ground_touch") + (trunk, legs)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(func=mdp.illegal_contact,
      params={"sensor_name": trunk.name, "force_threshold": 10.0})
    cfg.terminations["sustained_leg_contact"] = TerminationTermCfg(func=blind_mdp.sustained_leg_contact,
      params={"sensor_name": legs.name, "force_threshold": 10.0,
              "severe_threshold": 80.0, "duration_s": 0.1})
    cfg.rewards = make_lgdog_rewards("mgdp", base_body=BASE_BODY, foot_sites=FOOT_SITES)
    cfg.metrics.update({name: MetricsTermCfg(func=func) for name, func in (
      ("velocity_error_xy_mps", blind_mdp.velocity_error_xy),
      ("velocity_tracking_score", blind_mdp.velocity_tracking_score),
      ("velocity_tracking_active", blind_mdp.velocity_tracking_active),
      ("tracking_progress", blind_mdp.tracking_progress),
      ("commanded_distance", blind_mdp.commanded_distance))})
    if not play:
      cfg.commands["twist"].ranges.lin_vel_x = (-.2, .35)
      cfg.commands["twist"].ranges.lin_vel_y = (-.1, .1)
      cfg.commands["twist"].ranges.ang_vel_z = (-.4, .4)
      cfg.curriculum["command_vel"] = CurriculumTermCfg(func=RewardBasedVelocityCurriculum,
        params={"command_name": "twist", "score_metric_name": "velocity_tracking_score",
          "active_metric_name": "velocity_tracking_active", "progress_metric_name": "tracking_progress",
          "distance_metric_name": "commanded_distance", "min_progress_ratio": .6,
          "regress_min_progress_ratio": .4, "linear_threshold": .78, "linear_increment": .025,
          "linear_patience": 3, "linear_regress_threshold": .62, "linear_regress_patience": 2,
          "min_episode_fraction": .8, "max_failure_fraction": .25,
          "regress_min_episode_fraction": .65, "regress_max_failure_fraction": .35,
          "lin_vel_x_limit": (-.4, .6), "lin_vel_y_limit": (-.2, .2), "check_interval_s": 20.})


def lgdog1_race_mgdp_env_cfg(
    play: bool = False,
    checkpoint: str = DEFAULT_RACE_CHECKPOINT,
    with_perception: bool = True,
):
    """MGDP task on the 2027 race terrain, including overhead beam + tyre ring."""
    cfg = lgdog1_mgdp_env_cfg(
        play=play, checkpoint=checkpoint, with_perception=with_perception
    )
    if not play:
      add_race_mass_inertia_randomization(cfg)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = lgdog1_passage_terrain_cfg()
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = 75.
    # Safe pad at the terrain origin, always facing the obstacle strip.
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (-.03, .03), "y": (-.03, .03), "z": (0., 0.), "yaw": (-.03, .03)}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events.pop("push_robot", None)
    cfg.events.pop("randomize_terrain", None)
    cfg.commands["twist"] = passage.PassageCommandCfg(entity_name="robot",
      resampling_time_range=(76., 76.), heading_command=True,
      ranges=passage.PassageCommandCfg.Ranges(lin_vel_x=(.25,.35), lin_vel_y=(-.1,.1),
        ang_vel_z=(-.8,.8), heading=(-3.141593,3.141593)), goal_x=GOAL_X, lane_half_width=LANE_HALF_WIDTH,
      waypoints={name:lesson_route(name) for name in cfg.scene.terrain.terrain_generator.sub_terrains})
    cfg.terminations["passage_success"] = TerminationTermCfg(func=passage.passage_success)
    cfg.terminations["left_lane"] = TerminationTermCfg(func=passage.left_lane)
    cfg.rewards = make_mgdp_passage_rewards(base_body=BASE_BODY, foot_sites=FOOT_SITES)
    cfg.curriculum = {} if play else {"passage": CurriculumTermCfg(func=passage.PassageCurriculum,
      params={"min_episodes": 32, "success_threshold": .7, "max_error": .25, "speed_limit": .6})}
    return cfg
