"""LgDog1 + MGDP perception task configuration (mjlab port).

This is the mjlab counterpart of MGDP's "stage 2" controller: a 47-dim
proprioception history plus a 64-dim depth feature produced by a pretrained
contrastive perception model.

Pipeline
--------
    RayCastSensor (16x16 pinhole, 87 deg FOV, forward/down 45 deg)
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
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg

from src.assets.robots.lgdog1.lgdog1_constants import get_lgdog1_robot_cfg, get_spec
from src.tasks.velocity.config.lgdog1.env_cfgs import BASE_BODY, lgdog1_rough_env_cfg
from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerceptionCfg

DEPTH_SITE = "mgdp_depth"
DEPTH_HW = (16, 16)
DEPTH_FOVY = 87.0
DEPTH_MAX = 5.0
PROPRIO_HISTORY = 4

# Camera pose from MGDP's Go2 table, applied to LgDog1's trunk:
# 0.34 m forward, 0.06 m up, pitched 45 deg down (MuJoCo cameras look along -Z).
DEPTH_SITE_POS = (0.34, 0.0, 0.06)
DEPTH_SITE_QUAT = (0.9238795325, 0.0, -0.3826834324, 0.0)

DEFAULT_CHECKPOINT = str(
    Path("logs") / "mgdp" / "perception" / "mgdp_perception.pt"
)


def _mgdp_robot_spec():
    """Robot spec with the forward/down depth site added to the trunk."""
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
    return cfg
