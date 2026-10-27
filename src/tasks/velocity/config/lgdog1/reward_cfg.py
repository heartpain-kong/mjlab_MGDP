"""LgDog 奖励调参入口。修改后重启训练进程；已有进程不会热更新。

1. COMMON_WEIGHTS：所有 LgDog 任务的公共权重。
2. REWARD_PROFILES：只改 Flat / Rough / TerrainV4 / HIM / HIM-Standard。
3. make_lgdog_rewards：奖励函数绑定、公共参数和 LgDog 身体/足端绑定。

查看实际结果：python scripts/list_rewards.py LgDog1-Rough-Blind-HIM
奖励公式：src/tasks/velocity/mdp/rewards.py；相对离地高度：blind_mdp.py。
"""

from copy import deepcopy
import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity import mdp
from src.reward_semantics import validate_reward_weights
from . import blind_mdp


# 公共权重：保持本次重构前的实际数值。正数通常是奖励，负数通常是代价。
# 最终每步贡献 = 函数返回值 × weight × env.step_dt。0 表示关闭该项。
COMMON_WEIGHTS = {
  # 跟踪、姿态和步态
  "track_linear_velocity": 2.0,
  "track_angular_velocity": 1.0,
  "body_orientation_l2": -1.0,
  "pose": 1.5,
  "foot_gait": 0.5,
  # 平滑和身体稳定
  "body_ang_vel": -0.05,
  "angular_momentum": -0.025,
  "joint_acc_l2": -2.5e-7,
  "joint_pos_limits": -10.0,
  "action_rate_l2": -0.05,
  # 足端和地形
  "foot_clearance": -1.0,
  "foot_slip": -0.25,
  "soft_landing": -1e-3,
  "feet_stumble": -0.5,
  "feet_air_time": 0.0,
  "leg_contact_force": -1.0,
  # 站立和终止
  "stand_still": -1.0,
  "is_terminated": -200.0,
}


# 各配置分别覆盖 COMMON_WEIGHTS 和下方公共 params，不相互继承。
# Rough / Blind / Gentle 共用 rough；TerrainV4 与两类 HIM 使用相对地面高度。
# 只改 HIM 示例：在 him.weights 中填 "foot_clearance": -0.8。
REWARD_PROFILES = {
  "flat": {
    "weights": {"feet_stumble": 0.0, "feet_air_time": 0.0, "leg_contact_force": 0.0},
    "params": {"foot_clearance": {"target_height": 0.10}},
  },
  "rough": {
    "weights": {},
    "params": {},
  },
  "terrain_v4": {
    "weights": {},
    "params": {
      "foot_clearance": {"target_height": 0.10},
      "leg_contact_force": {"sensor_name": "legs_ground_contact"},
    },
  },
  "him": {
    "weights": {},
    "params": {
      "foot_clearance": {"target_height": 0.10},
      "leg_contact_force": {"sensor_name": "legs_ground_contact"},
    },
  },
  # 新标准崎岖地形任务独立调参；不会覆盖 V7 的 him 配置。
  # V21 regression recovery (2026-10-27): restore the V19 reward baseline.
  # Clearance is a nonnegative error cost; zero disables it.  Foot rays still
  # supply critic clearance labels, but no clearance reward runs at weight 0.
  "him_standard": {
    "weights": {
      "foot_clearance": 0.0,
      "feet_stumble": -0.5,
      "feet_air_time": 1.0,
      "soft_landing": -0.002,
      "track_linear_velocity": 2.0,
      "pose": 1.5,
      "action_rate_l2": -0.1,
    },
    "params": {
      # Keep relative shaping available for a separate controlled experiment.
      "track_linear_velocity": {
        "relative_weight": 0.0,
        "relative_std": 0.5,
        "min_command_speed": 0.2,
      },
      "foot_clearance": {"target_height": 0.10},
      "leg_contact_force": {"sensor_name": "legs_ground_contact"},
    },
  },
}


# 面板保存的方案级覆盖。公共 COMMON_WEIGHTS 保持为基线，调参时只影响当前 profile。
# 由本地浏览器工作台维护，不需要导入 mjlab。
# BEGIN LGDOG DASHBOARD OVERRIDES
DASHBOARD_REWARD_OVERRIDES = {'flat': {'weights': {}, 'params': {}},
 'rough': {'weights': {}, 'params': {}},
 'terrain_v4': {'weights': {}, 'params': {}},
 'him': {'weights': {}, 'params': {}},
 'him_standard': {'weights': {}, 'params': {}}}
# END LGDOG DASHBOARD OVERRIDES


def make_lgdog_rewards(profile, *, base_body, foot_sites, terrain_ray_names=()):
  """Build independent, complete RewardTermCfg objects for one LgDog task.

  Environment factories supply existing body/site names and terrain sensors.
  Keep numerical tuning here, rather than adding later overrides in env_cfgs.py.
  """
  if profile not in REWARD_PROFILES:
    raise ValueError(f"Unknown LgDog reward profile {profile!r}; choose {tuple(REWARD_PROFILES)}")
  settings = deepcopy(REWARD_PROFILES[profile])
  dashboard = deepcopy(DASHBOARD_REWARD_OVERRIDES.get(profile, {}))
  settings["weights"].update(dashboard.get("weights", {}))
  settings["params"].update(dashboard.get("params", {}))
  unknown = (settings["weights"].keys() | settings["params"].keys()) - COMMON_WEIGHTS.keys()
  if unknown:
    raise ValueError(f"Unknown reward names in {profile!r}: {sorted(unknown)}")
  weights = {**COMMON_WEIGHTS, **settings["weights"]}
  validate_reward_weights(weights)

  def term(name, func, **params):
    params.update(settings["params"].get(name, {}))
    return RewardTermCfg(func=func, weight=weights[name], params=params)

  # 公共参数也在这里改。asset_cfg 每次新建，避免任务之间共享可变对象。
  rewards = {
    "track_linear_velocity": term(
      "track_linear_velocity", mdp.track_linear_velocity,
      command_name="twist", std=math.sqrt(0.25),
    ),
    "track_angular_velocity": term(
      "track_angular_velocity", mdp.track_angular_velocity,
      command_name="twist", std=math.sqrt(0.5),
    ),
    "body_orientation_l2": term(
      "body_orientation_l2", mdp.body_orientation_l2,
      asset_cfg=SceneEntityCfg("robot", body_names=(base_body,)),
    ),
    "pose": term(
      "pose", mdp.variable_posture,
      asset_cfg=SceneEntityCfg("robot", joint_names=".*"), command_name="twist",
      std_standing={
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
      },
      std_walking={
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
      },
      std_running={
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
      },
      walking_threshold=0.1, running_threshold=1.5,
    ),
    "foot_gait": term(
      "foot_gait", mdp.feet_gait, period=0.6, offset=[0.0, 0.5, 0.5, 0.0],
      threshold=0.56, command_threshold=0.1, command_name="twist",
      sensor_name="feet_ground_contact",
    ),
    "body_ang_vel": term(
      "body_ang_vel", mdp.body_angular_velocity_penalty,
      asset_cfg=SceneEntityCfg("robot", body_names=(base_body,)),
    ),
    "angular_momentum": term(
      "angular_momentum", mdp.angular_momentum_penalty, sensor_name="robot/root_angmom",
    ),
    "joint_acc_l2": term("joint_acc_l2", mdp.joint_acc_l2),
    "joint_pos_limits": term("joint_pos_limits", mdp.joint_pos_limits),
    "action_rate_l2": term("action_rate_l2", mdp.action_rate_l2),
    "foot_clearance": term(
      "foot_clearance", mdp.feet_clearance, target_height=0.15,
      command_name="twist", command_threshold=0.1,
      asset_cfg=SceneEntityCfg("robot", site_names=foot_sites),
    ),
    "foot_slip": term(
      "foot_slip", mdp.feet_slip, sensor_name="feet_ground_contact",
      command_name="twist", command_threshold=0.1,
      asset_cfg=SceneEntityCfg("robot", site_names=foot_sites),
    ),
    "soft_landing": term(
      "soft_landing", mdp.soft_landing, sensor_name="feet_ground_contact",
      command_name="twist", command_threshold=0.1,
    ),
    "feet_stumble": term(
      "feet_stumble", mdp.feet_stumble, sensor_name="feet_ground_contact",
      command_name="twist", command_threshold=0.1, horizontal_ratio=4.0,
    ),
    "feet_air_time": term(
      "feet_air_time", mdp.feet_air_time, sensor_name="feet_ground_contact",
      threshold=0.4, command_name="twist", command_threshold=0.1,
    ),
    "leg_contact_force": term(
      "leg_contact_force", mdp.leg_contact_force, sensor_name="nonfoot_ground_touch",
      command_name="twist", command_threshold=0.1,
    ),
    "stand_still": term(
      "stand_still", mdp.stand_still, command_name="twist", command_threshold=0.1,
      asset_cfg=SceneEntityCfg("robot", joint_names=".*"),
    ),
    "is_terminated": term("is_terminated", mdp.is_terminated),
  }
  if rewards.keys() != COMMON_WEIGHTS.keys():
    raise ValueError("Every COMMON_WEIGHTS entry must have a reward binding")

  if profile in {"terrain_v4", "him", "him_standard"}:
    if len(terrain_ray_names) != len(foot_sites):
      raise ValueError(f"{profile} clearance requires one terrain ray for each foot")
    rewards["foot_clearance"].func = blind_mdp.feet_clearance_terrain
    rewards["foot_clearance"].params.setdefault("sensor_names", tuple(terrain_ray_names))
  return rewards
