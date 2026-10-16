from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
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
