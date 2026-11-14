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

from dataclasses import replace, fields
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
from .domain_rand_cfg import configure_race_himloco_randomization

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
    terrain_only_sensing: bool = True,
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

    # MGDP's released WarpManager builds one terrain mesh, without the robot.
    # Keep privileged heights and depth on the same terrain-only convention.
    # Legacy policies restore their original sensor filters on checkpoint load.
    if terrain_only_sensing:
        for sensor in cfg.scene.sensors or ():
            if isinstance(sensor, RayCastSensorCfg):
                sensor.include_geom_groups = (0,)

    # Depth rays: a pinhole camera made of rays (cheap at thousands of envs).
    depth_sensor = RayCastSensorCfg(
        name="depth_rays",
        frame=ObjRef(type="site", name=DEPTH_SITE, entity="robot"),
        ray_alignment="base",
        pattern=PinholeCameraPatternCfg(
            width=depth_hw[1], height=depth_hw[0], fovy=DEPTH_FOVY
        ),
        max_distance=DEPTH_MAX,
        include_geom_groups=(0,) if terrain_only_sensing else None,
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
                        "inference_batch_size": 1024,
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
    terrain_only_sensing: bool = True,
):
    """MGDP task on the 2027 race terrain, including overhead beam + tyre ring."""
    cfg = lgdog1_mgdp_env_cfg(
        play=play, checkpoint=checkpoint, with_perception=with_perception,
        terrain_only_sensing=terrain_only_sensing,
    )
    if not play:
      configure_race_himloco_randomization(cfg)
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
      params={"min_episodes": 32, "success_threshold": .7, "max_error": .25,
              "speed_limit": passage.DEFAULT_SPEED_LIMIT})}
    return cfg


def lgdog1_wall_training_env_cfg(play=False, **kwargs):
    """V4: every obstacle reaches competition dimensions at stage six."""
    cfg = lgdog1_race_mgdp_env_cfg(play=play, **kwargs)
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from .passage_terrain_cfg import StrictStonesTerrainCfg
    from .course_v4_terrain_cfg import UnifiedWallTerrainCfg, UnifiedGapTerrainCfg, UnifiedObstacleTerrainCfg
    from src.tasks.velocity.mdp.course_v3 import (
        BodyAwarePassageCommandCfg,SAFE_SLALOM_ROUTE,PositionHistoryContactSensorCfg,
        StoneGroundContact,SlalomBodyProjection,
    )
    from src.tasks.velocity.mdp.wall_training import (
        NoPassageProgress, WallCrossingBonus, WallSwingClearance,
        WallWaitingPenalty,guard_wall_shaping_rewards,
    )
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 8
    generator.num_cols = 20
    generator.size = (12.,8.)
    lessons = {}
    for name, terrain in generator.sub_terrains.items():
        if name == "gap_steps":
            lessons[name] = UnifiedGapTerrainCfg(size=generator.size)
        elif name == "stepping_stones":
            lessons[name] = StrictStonesTerrainCfg(size=generator.size)
        elif name == "high_wall":
            lessons[name] = UnifiedWallTerrainCfg(size=generator.size,
                counts=(3,)*8 if play else (1,1,1,2,2,3,3,3))
        else:
            lessons[name] = UnifiedObstacleTerrainCfg(kind=name,size=generator.size)
    for terrain in lessons.values():
        terrain.proportion = 1./len(lessons)
    generator.sub_terrains = lessons
    old=cfg.commands['twist']
    cfg.commands['twist']=BodyAwarePassageCommandCfg(**{f.name:getattr(old,f.name) for f in fields(old) if f.init})
    cfg.commands["twist"].waypoints = {name:lesson_route(name) for name in lessons}
    cfg.commands['twist'].waypoints['gap_steps']=((0.,0.),(10.4,0.))
    cfg.commands['twist'].waypoints['slalom_brick']=SAFE_SLALOM_ROUTE
    cfg.episode_length_s=120.
    cfg.commands['twist'].resampling_time_range=(121.,121.)
    cfg.scene.sensors=tuple(cfg.scene.sensors)+(PositionHistoryContactSensorCfg(
        name='stone_support_contact',
        primary=ContactMatch(mode='geom',pattern=FOOT_GEOMS,entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),
        fields=('found','force','pos','normal'),reduce='maxforce',num_slots=4,history_length=cfg.decimation),)
    cfg.terminations['stone_ground_contact']=TerminationTermCfg(func=StoneGroundContact,
        params={'sensor_name':'stone_support_contact','entry_end_x':2.2,'exit_start_x':5.9,
                'floor_tolerance_m':.008,'min_force_n':1.})
    cfg.terminations['slalom_body_projection']=TerminationTermCfg(func=SlalomBodyProjection,
        params={'asset_cfg':SceneEntityCfg('robot',geom_names=('trunk_collision',)),
                'post_radius_m':.05,'margin_m':.03})
    guard_wall_shaping_rewards(cfg.rewards)
    cfg.rewards['wall_waiting']=RewardTermCfg(func=WallWaitingPenalty,weight=-1.,
        params={'asset_cfg':SceneEntityCfg('robot',site_names=FOOT_SITES),
                'approach_m':.75,'grace_s':.75,'ramp_s':1.5,'max_rate':2.,
                'min_progress_m':.02,'min_command_mps':.1})
    if play:
        return cfg
    cfg.terminations["no_passage_progress"] = TerminationTermCfg(
        func=NoPassageProgress, time_out=False,
        params={"timeout_s": 4., "min_progress_m": .05, "min_command_mps": .1})
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    cfg.rewards["wall_crossing"] = RewardTermCfg(func=WallCrossingBonus, weight=2.,
        params={"asset_cfg": feet, "root_clearance_m": .35, "foot_clearance_m": .03})
    cfg.rewards["wall_swing_clearance"] = RewardTermCfg(func=WallSwingClearance, weight=-.2,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
                "sensor_name": "feet_ground_contact", "margin_m": .03,
                "approach_m": .25, "departure_m": .08})
    return cfg
