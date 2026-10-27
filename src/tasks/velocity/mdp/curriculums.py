from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def velocity_ranges_from_bounds(
  min_abs: float, max_abs: float,
) -> dict[str, tuple[float, float]]:
  """Build signed command ranges from positive lower/upper magnitudes."""
  minimum = float(min_abs)
  maximum = float(max_abs)
  if not math.isfinite(minimum) or not math.isfinite(maximum):
    raise ValueError("速度范围必须使用有限数")
  if minimum <= 0.0 or maximum <= 0.0:
    raise ValueError("最低和最高速度绝对值必须大于 0")
  if minimum > maximum:
    raise ValueError("最低速度绝对值不能大于最高速度绝对值")
  return {
    "lin_vel_x": (-minimum, maximum),
    "lin_vel_y": (-0.375 * maximum, 0.375 * maximum),
    "ang_vel_z": (-0.75 * maximum, 0.75 * maximum),
  }


def make_velocity_stages(
  min_abs: float = 0.4,
  max_abs: float = 0.8,
  *,
  final_step: int = 240_000,
) -> list[VelocityStage]:
  """Create a control-step speed curriculum for the blind LgDog task."""
  minimum = float(min_abs)
  maximum = float(max_abs)
  if final_step < 1:
    raise ValueError("速度课程最终阶段必须位于第 1 个控制步之后")
  final_ranges = velocity_ranges_from_bounds(minimum, maximum)
  start_ranges = velocity_ranges_from_bounds(minimum * 0.5, maximum * 0.5)
  middle_ranges = {
    key: tuple((a + b) / 2.0 for a, b in zip(start_ranges[key], final_ranges[key]))
    for key in start_ranges
  }
  return [
    {"step": 0, **start_ranges},
    {"step": final_step // 2, **middle_ranges},
    {"step": final_step, **final_ranges},
  ]


class RewardBasedVelocityCurriculum(ManagerTermBase):
  """Expand command speed only after completed episodes track it well.

  This follows the HIMLoco/legged-gym rule rather than a wall-clock schedule:
  aggregate full-episode tracking scores, and widen the planar command range
  only when the linear score clears the configured threshold.  The final
  bounds remain the positive/negative limits selected in the workbench.
  Active-command scores can additionally require survival and failure gates.
  """

  _range_names = ("lin_vel_x", "lin_vel_y")
  _reward_names = ("track_linear_velocity", "track_angular_velocity")

  def __init__(self, cfg, env):
    super().__init__(env)
    params = cfg.params
    self.command_name = params.get("command_name", "twist")
    self.command_term = env.command_manager.get_term(self.command_name)
    if self.command_term is None:
      raise ValueError(f"Velocity curriculum command not found: {self.command_name}")
    self.linear_threshold = float(params.get("linear_threshold", 0.8))
    self.linear_increment = float(params.get("linear_increment", 0.1))
    self.linear_patience = max(1, int(params.get("linear_patience", 1)))
    self.linear_regress_threshold = float(
      params.get("linear_regress_threshold", max(0.0, self.linear_threshold - 0.2))
    )
    self.linear_regress_patience = max(1, int(params.get("linear_regress_patience", 3)))
    self.linear_regress_increment = float(
      params.get("linear_regress_increment", self.linear_increment * 0.5)
    )
    self.score_ema_alpha = float(params.get("score_ema_alpha", 0.5))
    # Optional direct metrics make the promotion score independent of the
    # reward mix and, most importantly, allow us to exclude standing commands.
    self.score_metric_name = params.get("score_metric_name")
    self.active_metric_name = params.get("active_metric_name")
    if bool(self.score_metric_name) != bool(self.active_metric_name):
      raise ValueError("Velocity curriculum needs both score_metric_name and active_metric_name")
    # Defaults preserve callers which do not opt into safety gates.  Standard
    # HIM config enables them explicitly; a high active-step score alone must
    # not promote a policy whose episodes mostly end in failure.
    self.min_episode_fraction = float(params.get("min_episode_fraction", 0.0))
    self.max_failure_fraction = float(params.get("max_failure_fraction", 1.0))
    self.regress_min_episode_fraction = float(params.get("regress_min_episode_fraction", 0.0))
    self.regress_max_failure_fraction = float(params.get("regress_max_failure_fraction", 1.0))
    for name in ("min_episode_fraction", "max_failure_fraction",
                 "regress_min_episode_fraction", "regress_max_failure_fraction"):
      value = getattr(self, name)
      if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Velocity curriculum {name} must be between 0 and 1")
    if self.regress_min_episode_fraction > self.min_episode_fraction:
      raise ValueError("Velocity curriculum regression survival gate exceeds promotion gate")
    if self.regress_max_failure_fraction < self.max_failure_fraction:
      raise ValueError("Velocity curriculum regression failure gate is below promotion gate")
    self.limits = {
      name: tuple(float(value) for value in params.get(f"{name}_limit", (-1.0, 1.0)))
      for name in self._range_names
    }
    interval_s = float(params.get("check_interval_s", env.max_episode_length_s))
    if not math.isfinite(interval_s) or interval_s <= 0.0:
      raise ValueError("Velocity curriculum check_interval_s must be positive")
    if not 0.0 <= self.linear_threshold <= 1.0:
      raise ValueError("Velocity curriculum linear_threshold must be between 0 and 1")
    if not 0.0 <= self.linear_regress_threshold <= self.linear_threshold:
      raise ValueError("Velocity curriculum regress threshold must be below linear threshold")
    if not math.isfinite(self.linear_increment) or self.linear_increment <= 0.0:
      raise ValueError("Velocity curriculum linear_increment must be positive")
    if not math.isfinite(self.linear_regress_increment) or self.linear_regress_increment <= 0.0:
      raise ValueError("Velocity curriculum regress increment must be positive")
    if not 0.0 < self.score_ema_alpha <= 1.0:
      raise ValueError("Velocity curriculum score_ema_alpha must be in (0, 1]")
    for name, (low, high) in self.limits.items():
      if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Invalid velocity curriculum limit for {name}")
      initial_low, initial_high = getattr(self.command_term.cfg.ranges, name)
      if not low <= initial_low <= initial_high <= high:
        raise ValueError(f"Initial {name} range is outside its curriculum limits")
    self.initial_ranges = {
      name: tuple(float(value) for value in getattr(self.command_term.cfg.ranges, name))
      for name in self._range_names
    }
    self.check_interval_steps = max(1, math.ceil(interval_s / env.step_dt))
    self.last_check_step = int(env.common_step_counter)
    self.episode_count = 0
    self.reward_sums = torch.zeros(2, device=self.device)
    self.scores = torch.zeros(2, device=self.device)
    self.score_ema = torch.zeros(2, device=self.device)
    self.has_score_ema = False
    self.metric_score_sum = torch.zeros((), device=self.device)
    self.metric_active_steps = torch.zeros((), device=self.device)
    self.episode_steps_sum = torch.zeros((), device=self.device)
    self.failed_episode_count = torch.zeros((), device=self.device)
    self.episode_fraction = torch.zeros((), device=self.device)
    self.failure_fraction = torch.zeros((), device=self.device)
    self.promotion_allowed = False
    self.promotion_streak = 0
    self.regression_streak = 0

  def __call__(self, env, env_ids, **params):
    del params
    if env.common_step_counter == 0:
      return self._metrics()
    if isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(env_ids):
      sums = env.reward_manager._episode_sums
      self.reward_sums += torch.stack([
        sums[name][env_ids].sum() for name in self._reward_names
      ])
      self.episode_count += len(env_ids)
      self.episode_steps_sum += env.episode_length_buf[env_ids].clamp(
        max=env.max_episode_length
      ).sum()
      self.failed_episode_count += env.termination_manager.terminated[env_ids].sum()
      metric_sums = getattr(getattr(env, "metrics_manager", None), "_episode_sums", {})
      if self.score_metric_name and self.active_metric_name:
        score_values = metric_sums.get(self.score_metric_name)
        active_values = metric_sums.get(self.active_metric_name)
        if score_values is None or active_values is None:
          raise ValueError("Velocity curriculum tracking metrics are missing; check env.metrics")
        self.metric_score_sum += score_values[env_ids].sum()
        self.metric_active_steps += active_values[env_ids].sum()
    if (
      env.common_step_counter - self.last_check_step < self.check_interval_steps
      or self.episode_count == 0
    ):
      return self._metrics()
    weights = torch.tensor([
      env.reward_manager.get_term_cfg(name).weight for name in self._reward_names
    ], device=self.device)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
      raise ValueError("Velocity curriculum tracking rewards need positive weights")
    duration = (
      env.max_episode_length_s
      if getattr(env.reward_manager, "_scale_by_dt", True)
      else env.max_episode_length
    )
    reward_scores = self.reward_sums / (weights * duration * self.episode_count)
    self.scores.copy_(reward_scores)
    if float(self.metric_active_steps.item()) > 0.0:
      # Replace only the linear score.  Angular score remains the legacy
      # reward-normalized value because the curriculum expands planar speed.
      self.scores[0] = self.metric_score_sum / self.metric_active_steps
    self.episode_fraction.copy_(
      (self.episode_steps_sum / (env.max_episode_length * self.episode_count)).clamp(0.0, 1.0)
    )
    self.failure_fraction.copy_(self.failed_episode_count / self.episode_count)
    has_moving_samples = not self.score_metric_name or bool(self.metric_active_steps > 0)
    self.promotion_allowed = (
      has_moving_samples
      and bool(self.episode_fraction >= self.min_episode_fraction)
      and bool(self.failure_fraction <= self.max_failure_fraction)
    )
    unsafe = (
      bool(self.episode_fraction < self.regress_min_episode_fraction)
      or bool(self.failure_fraction > self.regress_max_failure_fraction)
    )
    if not self.has_score_ema:
      self.score_ema.copy_(self.scores)
      self.has_score_ema = True
    else:
      self.score_ema.lerp_(self.scores, self.score_ema_alpha)
    if self.score_ema[0] >= self.linear_threshold and self.promotion_allowed:
      self.promotion_streak += 1
      self.regression_streak = 0
    elif self.score_ema[0] < self.linear_regress_threshold or unsafe:
      self.regression_streak += 1
      self.promotion_streak = 0
    else:
      self.promotion_streak = 0
      self.regression_streak = 0
    if self.promotion_streak >= self.linear_patience:
      self._expand("lin_vel_x", self.linear_increment)
      self._expand("lin_vel_y", self.linear_increment)
      self.promotion_streak = 0
    elif self.regression_streak >= self.linear_regress_patience:
      self._shrink("lin_vel_x", self.linear_regress_increment)
      self._shrink("lin_vel_y", self.linear_regress_increment)
      self.regression_streak = 0
    self._reset_window()
    self.last_check_step = int(env.common_step_counter)
    return self._metrics()

  def _expand(self, name, increment):
    low, high = getattr(self.command_term.cfg.ranges, name)
    limit_low, limit_high = self.limits[name]
    setattr(self.command_term.cfg.ranges, name, (
      round(max(low - increment, limit_low), 10),
      round(min(high + increment, limit_high), 10),
    ))

  def _shrink(self, name, decrement):
    low, high = getattr(self.command_term.cfg.ranges, name)
    initial_low, initial_high = self.initial_ranges[name]
    setattr(self.command_term.cfg.ranges, name, (
      round(min(low + decrement, initial_low), 10),
      round(max(high - decrement, initial_high), 10),
    ))

  def _metrics(self):
    return {
      "linear_score": self.scores[0],
      "angular_score": self.scores[1],
      "linear_score_ema": self.score_ema[0],
      "promotion_streak": float(self.promotion_streak),
      "regression_streak": float(self.regression_streak),
      "episode_fraction": self.episode_fraction,
      "failure_fraction": self.failure_fraction,
      "promotion_allowed": float(self.promotion_allowed),
      "curriculum_vx_max": float(self.command_term.cfg.ranges.lin_vel_x[1]),
      "curriculum_vx_min": float(self.command_term.cfg.ranges.lin_vel_x[0]),
    }

  def state_dict(self):
    return {
      "version": 2,
      "command_name": self.command_name,
      "ranges": {
        name: tuple(getattr(self.command_term.cfg.ranges, name))
        for name in self._range_names
      },
      "last_check_step": self.last_check_step,
      "episode_count": self.episode_count,
      "reward_sums": self.reward_sums.detach().cpu().clone(),
      "scores": self.scores.detach().cpu().clone(),
      "score_ema": self.score_ema.detach().cpu().clone(),
      "has_score_ema": self.has_score_ema,
      "metric_score_sum": self.metric_score_sum.detach().cpu().clone(),
      "metric_active_steps": self.metric_active_steps.detach().cpu().clone(),
      "episode_steps_sum": self.episode_steps_sum.detach().cpu().clone(),
      "failed_episode_count": self.failed_episode_count.detach().cpu().clone(),
      "episode_fraction": self.episode_fraction.detach().cpu().clone(),
      "failure_fraction": self.failure_fraction.detach().cpu().clone(),
      "promotion_streak": self.promotion_streak,
      "regression_streak": self.regression_streak,
    }

  def load_state_dict(self, state):
    if state.get("version") not in (1, 2) or state.get("command_name") != self.command_name:
      raise ValueError("Incompatible velocity curriculum checkpoint state")
    ranges = self._checked_ranges(state.get("ranges", {}))
    reward_sums = torch.as_tensor(state.get("reward_sums"), device=self.device)
    scores = torch.as_tensor(state.get("scores"), device=self.device)
    if (
      reward_sums.shape != (2,) or scores.shape != (2,)
      or not torch.isfinite(reward_sums).all() or not torch.isfinite(scores).all()
      or (reward_sums < 0).any() or (scores < 0).any()
      or int(state.get("episode_count", -1)) < 0
      or int(state.get("last_check_step", -1)) < 0
    ):
      raise ValueError("Invalid velocity curriculum checkpoint statistics")
    for name, bounds in ranges.items():
      setattr(self.command_term.cfg.ranges, name, bounds)
    self.reward_sums.copy_(reward_sums)
    self.scores.copy_(scores)
    score_ema = torch.as_tensor(state.get("score_ema", scores), device=self.device)
    if score_ema.shape != (2,) or not torch.isfinite(score_ema).all() or (score_ema < 0).any():
      raise ValueError("Invalid velocity curriculum EMA statistics")
    self.score_ema.copy_(score_ema)
    self.has_score_ema = bool(state.get("has_score_ema", True))
    metric_score_sum = torch.as_tensor(
      state.get("metric_score_sum", 0.0), device=self.device
    )
    metric_active_steps = torch.as_tensor(
      state.get("metric_active_steps", 0.0), device=self.device
    )
    if (
      metric_score_sum.numel() != 1 or metric_active_steps.numel() != 1
      or not torch.isfinite(metric_score_sum).all()
      or not torch.isfinite(metric_active_steps).all()
      or (metric_score_sum < 0).any() or (metric_active_steps < 0).any()
    ):
      raise ValueError("Invalid velocity curriculum direct metric statistics")
    self.metric_score_sum.copy_(metric_score_sum.reshape(()))
    self.metric_active_steps.copy_(metric_active_steps.reshape(()))
    self.promotion_streak = int(state.get("promotion_streak", 0))
    self.regression_streak = int(state.get("regression_streak", 0))
    if self.promotion_streak < 0 or self.regression_streak < 0:
      raise ValueError("Invalid velocity curriculum streak state")
    self.episode_count = int(state.get("episode_count", 0))
    self.last_check_step = min(int(state.get("last_check_step", 0)), int(self._env.common_step_counter))
    if state["version"] == 1:
      # Old partial windows have no survival statistics.  Keep their restored
      # ranges, but require a fresh window and fresh streaks for the new gates.
      self._reset_window()
      self.promotion_streak = self.regression_streak = 0
      self.last_check_step = int(self._env.common_step_counter)
      self.episode_fraction.zero_()
      self.failure_fraction.zero_()
    else:
      for name in ("episode_steps_sum", "failed_episode_count", "episode_fraction", "failure_fraction"):
        if state.get(name) is None:
          raise ValueError("Missing velocity curriculum survival statistics")
        value = torch.as_tensor(state.get(name), device=self.device)
        if value.numel() != 1 or not torch.isfinite(value).all() or (value < 0).any():
          raise ValueError("Invalid velocity curriculum survival statistics")
        if name in {"episode_fraction", "failure_fraction"} and (value > 1).any():
          raise ValueError("Invalid velocity curriculum survival fraction")
        getattr(self, name).copy_(value.reshape(()))
      if self.failed_episode_count > self.episode_count:
        raise ValueError("Invalid velocity curriculum failure count")
      if self.episode_steps_sum > self.episode_count * self._env.max_episode_length:
        raise ValueError("Invalid velocity curriculum episode step count")
    # Re-evaluate the permission against the next actual completed window.
    self.promotion_allowed = False

  def _reset_window(self):
    self.reward_sums.zero_()
    self.metric_score_sum.zero_()
    self.metric_active_steps.zero_()
    self.episode_steps_sum.zero_()
    self.failed_episode_count.zero_()
    self.episode_count = 0

  def restore_legacy_ranges(self, saved_env_cfg: dict) -> bool:
    """Restore the last range from a checkpoint made with step-based stages."""
    ranges = dict(
      saved_env_cfg.get("commands", {}).get(self.command_name, {}).get("ranges", {})
    )
    if any(name not in ranges for name in self._range_names):
      return False
    ranges = self._checked_ranges(ranges)
    for name, bounds in ranges.items():
      setattr(self.command_term.cfg.ranges, name, bounds)
    self.last_check_step = int(self._env.common_step_counter)
    self._reset_window()
    self.scores.zero_()
    self.score_ema.zero_()
    self.has_score_ema = False
    self.episode_fraction.zero_()
    self.failure_fraction.zero_()
    self.promotion_streak = self.regression_streak = 0
    self.promotion_allowed = False
    return True

  def _checked_ranges(self, ranges: dict) -> dict[str, tuple[float, float]]:
    result = {}
    for name, (limit_low, limit_high) in self.limits.items():
      try:
        low, high = (float(value) for value in ranges[name])
      except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing checkpoint command range for {name}") from error
      if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Invalid checkpoint command range for {name}")
      result[name] = (
        min(max(low, limit_low), limit_high),
        min(max(high, limit_low), limit_high),
      )
    return result


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  use_path_length: bool = True,
) -> torch.Tensor:
  """按「本回合走了多远」升降地形等级。

  ★★ 2026-10-15 修：原来用【净位移】判定，导致课程整体失效（地形等级一路塌到 0）。

  原逻辑：
      distance = ||当前位置 − 出生点||          # ← 直线位移
      move_up  = distance > 地块边长/2 = 4.0 m
  问题：训练指令含 yaw 转向（`ang_vel_z ∈ ±1`），机器人一边走一边转，走的是圆弧。
  实测（LgDog1-Flat，512 回合）：
      净位移 2.77 m   vs   路径长度 6.14 m      ← 差 2.21 倍
      61% 的环境 |wz|>0.3（在转圈）
  ⇒ 明明走了 6.14 m（远超 4 m 门槛），净位移只有 2.77 m ⇒ 被判"没走够"，78% 的回合
    被降级 ⇒ 地形等级塌到 0 ⇒ 整个地形课程失效（实测 LgDog1-Rough 与
    LgDog1-Rough-Blind 两轮都塌到 level=0）。
  改用路径长度后，同一批数据里能升级的占比 22% → 83%。

  路径长度由 metrics 项 `traveled_distance` 每步累加、回合结束时读取
  （`use_path_length=False` 可退回原来的净位移口径做对照）。
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  from src.tasks.velocity.mdp.metrics import get_episode_traveled_distance

  distance = None
  if use_path_length:
    distance = get_episode_traveled_distance(env, env_ids)

  if distance is None:
    # 回退：净位移（原来的口径；也用于 metrics 不可用时的兜底）
    distance = torch.norm(
      asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
      dim=1,
    )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # ★ 2026-10-16 修：原来这里用 `distance < ||cmd_xy|| × 20s × 0.5`，
  #   指令大时阈值能到 10~20 m，**比升级门槛(4 m)还大** ⇒ 变成"要么升级、要么降级"，
  #   没有稳定带 ⇒ 反复横跳、整体被压到底。现在把降级阈值**封顶在升级门槛的一半**，
  #   形成 [size/4, size/2] = [2.0, 4.0] 的滞回带：走 2~4 m 的环境【保持原地不动】。
  #   小指令的环境仍按 `cmd × 20 × 0.5` 判（保留"要求低、门槛也低"的原意）。
  down_thr_cap = terrain_generator.size[0] / 4
  cmd_thr = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  down_thr = torch.minimum(cmd_thr, torch.full_like(cmd_thr, down_thr_cap))
  move_down = distance < down_thr
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
