from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from . import ops

if TYPE_CHECKING:
  from collections.abc import Sequence

  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  relative_weight: float = 0.0,
  relative_std: float = 0.5,
  min_command_speed: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  absolute = ops.linear_velocity_tracking(command, actual, std)
  if relative_weight <= 0.0:
    return absolute
  if relative_weight > 1.0:
    raise ValueError("relative_weight must be between 0 and 1")
  relative = ops.linear_velocity_tracking_relative(
    command, actual, relative_std, min_command_speed,
  )
  return (1.0 - relative_weight) * absolute + relative_weight * relative


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  return ops.angular_velocity_tracking(command, actual, std)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
  """罚动作的二阶差分（WMP 的 ``_reward_smoothness``）。

  ``sum((a_{t-2} - 2 a_{t-1} + a_t)^2)``。比 ``action_rate_l2`` 更针对高频
  抖动：动作匀速变化（``a_t - a_{t-1}`` 恒定）时该项为 0，来回抖才被罚。

  两帧动作历史直接复用 mjlab ``ActionManager`` 的 ``action`` /
  ``prev_action`` / ``prev_prev_action`` 缓存；``ActionManager.reset`` 在
  episode reset 时会把这三者一起清零，所以这里不需要另建状态。口径与 mjlab
  自带的 ``action_acc_l2`` 相同（同样的三帧差分）。
  """
  manager = env.action_manager
  return torch.sum(
    torch.square(
      manager.action - 2.0 * manager.prev_action + manager.prev_prev_action
    ),
    dim=1,
  )


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time, counting feet whose swing is in the target window.

  口径与 mjlab 自带 ``feet_air_time`` 一致：**数有多少只脚正处在
  ``(threshold_min, threshold_max)`` 的腾空时间里**，而不是去追某个"单支撑相时长"。

  为什么换掉旧写法：旧版奖励  ``min(mode_time)``（最近一次抬脚/落脚以来的时间）
  只有在恰好两只脚触地时才生效，而实测 V28 的策略 47% 的时间四脚着地、单支撑只占
  35%，并且腾空时间只有 0.10 s（p90 0.225 s）—— 旧版阈值 0.4 s 永远够不到，
  整段平均只有 0.021（中位数 0），等于没开。换成计数口径后同一轨迹上约 0.6，
  而且**每只脚抬起来就有信号**，不再依赖"刚好两只脚接地的瞬间"。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  if air_time is None:  # pragma: no cover - 传感器没开 track_air_time 时才会发生
    raise ValueError(f"{sensor_name} 没有记录 air time，检查 ContactSensorCfg.track_air_time")
  in_range = (air_time > threshold_min) & (air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = air_time > 0.0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(air_time * in_air.float()) / torch.clamp(num_in_air, min=1)
  env.extras.setdefault("log", {})["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      reward = reward * ops.command_active(command, command_threshold)
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      cost = cost * ops.command_active(command, command_threshold)
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            reward *= ops.command_active(command, command_threshold)
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    active = ops.command_active(command, command_threshold)
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = ops.command_active(command, command_threshold)
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  horizontal_ratio: float = 4.0,
  log_metric: bool = True,
) -> torch.Tensor:
  """罚【摆动足撞到竖直面】—— 上楼梯时脚磕在台阶立面上。

  ★ 为什么地形训练需要它
    地形生成器里 **40% 是台阶**（正金字塔 20% 上楼梯 + 倒金字塔 20% 下楼梯，
    单级高度由地形配置决定），另有斜坡 / 随机起伏 / 波浪。平地训练不需要这条，但上台阶
    最常见的失败就是**脚踢到台阶立面被磕住** —— 这时策略只拿到"往前推不动"的
    模糊信号，很难自己学明白。

  判据（与 legged_gym 的 `_reward_feet_stumble` 同源）：
      水平接触力 > horizontal_ratio × 竖直接触力   ⇒  判定为"撞立面"而非"踩地面"
    正常踩地时接触力几乎全是竖直的（水平分量 ≈ 摩擦力，很小）；撞立面时相反。

  与 legged_gym 版的差别：那边返回 `any(...)` 的 0/1 布尔，这里返回**连续的超出量**
  `max(水平力 − ratio×竖直力, 0)`，撞得越狠罚越多。判据不检查足端是否处于摆动期。

  本工程每控制步的贡献是 `cost × weight × step_dt`。这里的 cost 是力的超出量，
  不是 0/1；不能只按 dt 与其他实现换算权重，也不能把 -0.5 理解为每步固定扣 0.01。
  标准 HIM 恢复 V19 的 -0.5 基线，后续结合接触力和该项实际贡献做单独实验。
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  # reduce="netforce" ⇒ 力是【全局坐标系】，shape [B, N, 3]
  forces = sensor_data.force
  f_horizontal = torch.norm(forces[:, :, :2], dim=-1)  # 水平分量
  f_vertical = torch.abs(forces[:, :, 2])  # 竖直分量
  stumble = torch.clamp(f_horizontal - horizontal_ratio * f_vertical, min=0.0)
  cost = torch.max(stumble, dim=1)[0]  # 该环境里最严重的那只脚

  if log_metric:
    env.extras["log"]["Metrics/feet_stumble_mean"] = torch.mean(cost)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      cost = cost * ops.command_active(command, command_threshold)
  return cost


def leg_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  log_metric: bool = True,
) -> torch.Tensor:
  """罚【腿部（大腿/小腿/髋）撞到地形】—— 过台阶时的另一个典型失败。

  ★ 为什么地形训练需要它
    过台阶有两大失败模式，各管一头：
      · 脚磕台阶立面          → `feet_stumble`（已加）
      · **大腿/小腿撞台阶**    → 本条
    平地上腿不会碰地，所以平地任务用不上；上台阶时小腿很容易磕在台阶棱上，
    只靠 `illegal_contact`（10 N 直接终止）给不了梯度 —— 终止是 0/1 信号，
    碰一下和撞飞了 penalty 一样。这里给连续的力惩罚，让策略在【终止之前】
    就学到"腿别去蹭台阶"。

  数据源：`nonfoot_ground_touch` 传感器（`reduce="none"`，排除足端的那组碰撞体，
  force shape [B, N_geom, 3]，全局坐标系）。取所有非足端 geom 里最大的接触力。

  ⚠️ 权重换算：mjlab 会乘 dt(0.02)。legged_gym 的 `collision = -0.05`
  （罚 thigh/calf/base 接触力²）不乘 dt，等效量级约 weight ≈ -2.5。
  建议 -1.0 起步（原始值约 10~60 N → -0.2~-1.2 × 0.02 ≈ -0.004~-0.024/控制步）。
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  data = contact_sensor.data
  assert data.force is not None
  forces = data.force  # [B, N_geom, 3]，全局坐标系
  force_mag = torch.norm(forces, dim=-1)  # [B, N_geom]
  cost = torch.max(force_mag, dim=1)[0]  # 该环境里撞得最狠的那个 geom

  if log_metric:
    env.extras["log"]["Metrics/leg_contact_force_max"] = torch.mean(cost)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      cost = cost * ops.command_active(command, command_threshold)
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      cost = cost * ops.command_active(command, command_threshold)
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            reward *= 1.0 - ops.command_active(command, command_threshold)
    return reward


# 越障/绕桩的地形列种类。cheat 只在这些列里生效；地形生成器里没有的种类自动跳过。
OBSTACLE_KINDS = (
  "gap_steps", "high_wall", "clearance_frame", "tire_array", "slalom_brick",
)


def obstacle_columns(env: ManagerBasedRlEnv, kinds=OBSTACLE_KINDS) -> torch.Tensor:
  """返回长度 ``num_cols`` 的 bool 掩码：哪些地形列属于越障/绕桩。

  和 ``course_v3.kind_columns`` 一样按 ``terrain_generator.sub_terrains`` 的
  proportion 把列切开；``kinds`` 里当前地形没有的种类直接跳过，所以同一份
  奖励配置在只含部分障碍的任务（如 rough 地形）上也能安全求值。
  """
  terrain = env.scene.terrain
  generator = getattr(terrain.cfg, "terrain_generator", None)
  if generator is None:
    return torch.zeros(0, dtype=torch.bool, device=env.device)
  cache = getattr(terrain, "_obstacle_columns_cache", None)
  key = tuple(kinds)
  if cache is not None and cache[0] == key:
    return cache[1]
  from .course_v3 import kind_columns
  present = set(generator.sub_terrains)
  columns = torch.zeros(generator.num_cols, dtype=torch.bool, device=env.device)
  for kind in key:
    if kind in present:
      columns = columns | kind_columns(env, kind)
  # 地形在 env 创建时就固定了，缓存起来避免每个控制步重复切列。
  setattr(terrain, "_obstacle_columns_cache", (key, columns))
  return columns


def cheat(
  env: ManagerBasedRlEnv,
  heading_threshold: float = 1.0,
  kinds=OBSTACLE_KINDS,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """罚「绕开障碍物」的作弊行为（WMP 的 ``_reward_cheat``）。

  机器人在越障/绕桩列里、机头相对地形 +X 的偏航 ``|heading| > 1.0 rad`` 时
  每步返回 1（配负权重即惩罚）；其它列返回 0。WMP 用 ``gap_start_idx`` 到
  ``crawl_end_idx`` 的连续 env 区间，这里改成按地形列掩码，至少覆盖
  gap_steps / high_wall / clearance_frame / tire_array / slalom_brick。

  ``heading`` 取 ``robot.data.heading_w``，其定义就是 WMP 里的
  ``atan2(forward_w.y, forward_w.x)``。
  """
  columns = obstacle_columns(env, tuple(kinds))
  if columns.numel() == 0:
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  obstacle = columns[env.scene.terrain.terrain_types]
  heading = asset.data.heading_w
  return (obstacle & (heading.abs() > heading_threshold)).float()


# ★ 贴边惩罚（CMU Extreme Parkour 的 edge placement penalty）。
#   目的：脚落在距地形边缘/台阶棱 ``margin_m``（默认 5 cm）以内就扣分，
#   既防"贴着墙/沟沿走省力"，也对应真机"脚踩一半悬空→踩空"的风险。
_DEFAULT_FOOT_SITES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
# 与 terrain_scan 的 GridPatternCfg 默认值一致：1.6 m x 1.0 m、0.1 m 分辨率
# ⇒ (1.6/0.1+1) x (1.0/0.1+1) = 17 x 11 = 187 个采样点。
_DEFAULT_SCAN_GRID_SIZE = (1.6, 1.0)
_DEFAULT_SCAN_RESOLUTION = 0.1


def _scene_element(env: ManagerBasedRlEnv, name: str | None):
  """``env.scene[name]``，不存在时返回 None（不抛错）。

  mjlab 的 ``Scene`` 没有 ``__contains__``（用 ``in`` 会 TypeError），缺失项只在
  ``__getitem__`` 里抛 ``KeyError``；奖励项必须在"没这个传感器"的环境里安全求值，
  例如 blind 任务（无 terrain_scan）或只用 height_scan 张量做离线的调用方。
  """
  if not name:
    return None
  try:
    return env.scene[name]
  except (KeyError, TypeError):
    return None


def _log_reward_metric(env: ManagerBasedRlEnv, name: str, value: torch.Tensor) -> None:
  extras = getattr(env, "extras", None)
  if extras is None:
    return
  extras.setdefault("log", {})[name] = value


def _fallback_scan_xy(
  env: ManagerBasedRlEnv,
  count: int,
  grid_size: tuple[float, float],
  resolution: float,
  device,
  dtype: torch.dtype,
) -> torch.Tensor | None:
  """无射线传感器时，按 grid 形状参数重建网格点的世界 XY。

  仅在传感器缺失、但调用方直接给了 ``height_scan`` 时使用；假设网格与世界
  轴对齐、中心在环境原点（训练中的真实调用都走 ``hit_pos_w``，不受此限制）。
  布局与 ``mjlab.sensor.GridPatternCfg`` 的 ``meshgrid(x, y, indexing="xy")``
  展平顺序一致。
  """
  size_x, size_y = grid_size
  nx = int(round(size_x / resolution)) + 1
  ny = int(round(size_y / resolution)) + 1
  if nx < 2 or ny < 2 or nx * ny != count:
    return None
  xs = torch.linspace(
    -0.5 * size_x, 0.5 * size_x, nx, device=device, dtype=dtype
  )
  ys = torch.linspace(
    -0.5 * size_y, 0.5 * size_y, ny, device=device, dtype=dtype
  )
  grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
  xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [N, 2]
  origins = getattr(env.scene, "env_origins", None)
  if origins is not None:
    xy = xy + origins[:, :2].to(device=device, dtype=dtype).unsqueeze(1)
  return xy.unsqueeze(0).expand(int(env.num_envs), -1, -1)


def edge_place_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  foot_site_names: Sequence[str] = _DEFAULT_FOOT_SITES,
  margin_m: float = 0.05,
  height_jump_threshold_m: float = 0.03,
  contact_sensor_name: str | None = "feet_ground_contact",
  min_force_n: float = 1.0,
  asset_cfg: SceneEntityCfg | None = None,
  height_scan: torch.Tensor | None = None,
  grid_size: tuple[float, float] | None = None,
  resolution: float | None = None,
  log_metric: bool = True,
) -> torch.Tensor:
  """罚「脚踩在地形边缘/台阶棱上」—— 返回每步落在边缘上的脚数（0~4）。

  ★ 为什么需要它（CMU Extreme Parkour 的 edge placement penalty）
    贴边省力：墙根、沟沿、台阶棱常常是策略"蹭着过去"的捷径，但真机上脚掌
    一半悬空就是踩空。CMU 的做法是**脚落在距边缘 5 cm 以内就扣分**，我们照搬。

  判据（每只脚独立，满足则计 1，最后对 4 只脚求和 → ``[B]``，取值 0~4）：
    1. **触地**：``feet_ground_contact`` 的 ``found > 0`` 或接触力范数
       ``> min_force_n``（默认 1 N）—— 悬空脚本来就不该被罚；
    2. **脚下地形有高度跃变**：在高度网格上取脚周围半径
       ``margin_m + 0.5*分辨率`` 的局部窗口（0.1 m 分辨率下约 ±1 格），
       窗口内采样高度的极差 ``max(h) - min(h) > height_jump_threshold_m``
       （默认 3 cm）。平地极差为 0；脚踩在 10 cm 台阶棱上极差约 10 cm。

  数据来源两种，都支持：
    · 默认：从 ``env.scene[sensor_name]``（``RayCastSensor``，如 terrain_scan）
      现算高度 —— ``pos_w.z - hit_pos_w.z``，并用 ``distances >= 0`` 丢掉没打中
      的射线，采样点 XY 取 ``hit_pos_w``（已含传感器 yaw，天然是真实世界坐标）；
    · 直接传入扁平的 ``height_scan`` ``[B, N]``（单位米，**未缩放**），此时
      若没有传感器就用 ``grid_size`` / ``resolution`` 重建采样点（默认
      ``(1.6, 1.0)`` / ``0.1``，即 187 点）。这条路径方便离线/单测。

  ⚠️ 使用前提
    · 扫描要**只打地形**：mgdp/wall 任务里 terrain_scan 配了
      ``include_geom_groups=(0,)``，不会打到机器人自己的脚；如果射线会命中
      机器人小腿，脚边会出现假的"高度跃变"，此时请改扫描过滤或调大阈值。
    · 0.1 m 网格的边缘定位精度就是 0.1 m，5 cm 的 margin 实际被量化到约一格；
      想要更细请提高扫描分辨率。
    · 斜坡是连续梯度，不算"棱"；但 *rough / wave* 地形的局部起伏也可能超过
      3 cm，需要按任务把阈值抬到台阶高度的一半左右（如 0.04~0.05 m）。

  Args:
    sensor_name: 高度扫描的 ``RayCastSensor`` 名，默认 ``terrain_scan``。
    foot_site_names: 四个足端 site 名（顺序要与接触传感器的 slot 顺序一致，
      本工程都是 FR/FL/RR/RL）。传了 ``asset_cfg`` 时以它的 ``site_ids`` 为准。
    margin_m: 允许的离边距离（米），默认 0.05。
    height_jump_threshold_m: 局部窗口内触发惩罚的高度跃变（米），默认 0.03。
    contact_sensor_name: 足端接触传感器名；为 None / 不存在时返回全 0。
    min_force_n: 触地判定的接触力阈值（牛顿），默认 1.0；``found > 0`` 也算触地。
    asset_cfg: 可选，机器人实体 + site_ids（默认 ``robot`` + ``foot_site_names``）。
    height_scan: 可选，直接给 ``[B, N]`` 高度（米，未缩放），优先于传感器。
    grid_size / resolution: 无传感器时的网格形状参数（默认 1.6x1.0 / 0.1）。
    log_metric: 是否写 ``env.extras["log"]`` 诊断指标。

  Returns:
    ``[B]`` float 张量：每步落在边缘上的脚数（0~4）。没有 height_scan 传感器 /
    没有接触传感器 / 没有机器人时返回全 0，**不抛错**。

  权重换算：mjlab 会乘 ``dt``（0.02 s），本项原始值 0~4。weight=-0.5 时每只
  踩边脚每控制步约 -0.01，四脚同时踩边约 -0.04/步。mgdp_v6 这类已有
  ``feet_stumble=-0.1``、``leg_contact_force=-0.25`` 的档位，建议从 -0.5 起步。
  """
  num_envs = int(env.num_envs)
  device = env.device
  zeros = torch.zeros(num_envs, device=device)

  sensor = _scene_element(env, sensor_name)
  data = getattr(sensor, "data", None)
  pattern = getattr(getattr(sensor, "cfg", None), "pattern", None)

  # ── 1) 高度 [B, N]（米）。没有传感器也没显式给 height_scan ⇒ 全 0。 ──
  if height_scan is None:
    hit_pos_w = getattr(data, "hit_pos_w", None)
    pos_w = getattr(data, "pos_w", None)
    if hit_pos_w is None or pos_w is None or hit_pos_w.dim() != 3:
      return zeros
    heights = pos_w[:, 2].unsqueeze(1) - hit_pos_w[..., 2]  # [B, N]
    distances = getattr(data, "distances", None)
    if distances is not None and distances.shape == heights.shape:
      valid = distances >= 0
    else:
      valid = torch.isfinite(heights)
  else:
    heights = height_scan.to(device=device, dtype=torch.float32)
    if heights.dim() == 1:
      heights = heights.unsqueeze(0)
    if heights.dim() != 2 or heights.shape[0] != num_envs:
      return zeros
    valid = torch.isfinite(heights)

  heights = heights.to(device=device, dtype=torch.float32)
  valid = valid.to(device=device)
  count = int(heights.shape[1])
  if count == 0:
    return zeros

  # ── 2) 采样点的世界 XY [B, N, 2]。 ──
  hit_pos_w = getattr(data, "hit_pos_w", None)
  if hit_pos_w is not None and hit_pos_w.dim() == 3 and hit_pos_w.shape[1] == count:
    scan_xy = hit_pos_w[..., :2].to(device=device, dtype=torch.float32)
  else:
    res = resolution
    if res is None:
      res = getattr(pattern, "resolution", None)
    if not isinstance(res, (int, float)) or res <= 0:
      res = _DEFAULT_SCAN_RESOLUTION
    size = grid_size
    if size is None:
      size = getattr(pattern, "size", None)
    if not (isinstance(size, (tuple, list)) and len(size) == 2):
      size = _DEFAULT_SCAN_GRID_SIZE
    scan_xy = _fallback_scan_xy(
      env, count, (float(size[0]), float(size[1])), float(res), device,
      torch.float32,
    )
    if scan_xy is None:
      return zeros

  # ── 3) 足端世界 XY [B, F, 2]。 ──
  asset_name = asset_cfg.name if asset_cfg is not None else "robot"
  asset = _scene_element(env, asset_name)
  if asset is None:
    return zeros
  site_ids = list(getattr(asset_cfg, "site_ids", None) or ()) if asset_cfg else []
  if not site_ids:
    find_sites = getattr(asset, "find_sites", None)
    if find_sites is None:
      return zeros
    site_ids, _ = find_sites(foot_site_names)
  if not site_ids:
    return zeros
  site_pos_w = getattr(getattr(asset, "data", None), "site_pos_w", None)
  if site_pos_w is None:
    return zeros
  foot_xy = site_pos_w[:, site_ids, :2].to(device=device, dtype=torch.float32)

  # ── 4) 触地掩码 [B, F]。 ──
  contact_sensor = _scene_element(env, contact_sensor_name)
  contact = None
  if contact_sensor is not None:
    cdata = getattr(contact_sensor, "data", None)
    found = getattr(cdata, "found", None)
    if found is not None:
      found = found.to(device=device)
      while found.dim() > 2 and found.shape[-1] == 1:
        found = found.squeeze(-1)
      contact = found > 0
    force = getattr(cdata, "force", None)
    if force is not None:
      force_mag = torch.norm(force.to(device=device), dim=-1)  # [B, F]
      is_contact = force_mag > float(min_force_n)
      contact = is_contact if contact is None else (contact | is_contact)
  if contact is None:
    return zeros
  if contact.shape[0] != num_envs or contact.shape[1] != len(site_ids):
    # slot 数对不上时无法把力正确的映射到脚上，宁可不罚也不乱扣。
    return zeros
  contact = contact.bool()

  # ── 5) 每只脚的局部高度极差 [B, F]。 ──
  if resolution is not None and resolution > 0:
    res = float(resolution)
  else:
    res = getattr(pattern, "resolution", None)
    if not isinstance(res, (int, float)) or res <= 0:
      res = _DEFAULT_SCAN_RESOLUTION
  # 窗口 = margin + 半格：脚正好在两格之间时也能同时看到棱两侧的采样点。
  window = float(margin_m) + 0.5 * float(res)
  window2 = (window + 1e-6) ** 2
  dist2 = torch.sum(
    (foot_xy.unsqueeze(2) - scan_xy.unsqueeze(1)) ** 2, dim=-1
  )  # [B, F, N]
  in_window = (dist2 <= window2) & valid.unsqueeze(1)

  neg_fill = torch.finfo(torch.float32).min
  pos_fill = torch.finfo(torch.float32).max
  local_max = torch.where(in_window, heights.unsqueeze(1), neg_fill).amax(dim=-1)
  local_min = torch.where(in_window, heights.unsqueeze(1), pos_fill).amin(dim=-1)
  has_points = in_window.any(dim=-1)
  jump = torch.where(
    has_points, local_max - local_min, torch.zeros_like(local_max)
  )  # [B, F]
  on_edge = has_points & (jump > float(height_jump_threshold_m)) & contact

  # ── 6) 计数 + 诊断日志。 ──
  count_edges = on_edge.to(torch.float32).sum(dim=-1)  # [B]，0~4
  if log_metric:
    _log_reward_metric(env, "Metrics/edge_place_penalty_mean", count_edges.mean())
    _log_reward_metric(env, "Metrics/edge_place_penalty_fraction", on_edge.float().mean())
  return count_edges

