"""Velocity 任务的 Metrics 项。

Metrics 与 Reward 的区别：Metrics 只记录、不进奖励，但（关键）
**每步都会计算并被 MetricsManager 累加到 _episode_sums**，回合结束时清零。
课程 `terrain_levels_vel` 就靠这个拿到"本回合累计量"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


_DEFAULT_ASSET_CFG = None


def traveled_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """本步走过的路程 `|v_xy| × dt`（米）。

  ★ 为什么需要它（2026-10-15 查出来的坑）
    `terrain_levels_vel` 原本用【净位移】`||pos - env_origin||` 判定升降级：
        走出去 > 4 m（地块的一半）→ 升级
    但训练指令里含 **yaw 转向**（`ang_vel_z ∈ ±1`），机器人一边走一边转，
    走的是**圆弧**——实测（LgDog1-Flat，512 回合）：

        净位移（课程用的） 2.77 m
        路径长度（真走了） 6.14 m      ← 只有净位移的 2.21 倍差距
        61% 的环境 |wz|>0.3（在转圈）

    ⇒ 机器人明明走了 6.14 m（远超 4 m 门槛），净位移却只有 2.77 m，
      **每回合被判"没走够"而降级** → 78% 的回合被降级 → 地形等级一路塌到 0
      （实测：LgDog1-Rough 和 LgDog1-Rough-Blind 两轮都塌到 level=0，
       等于整个课程失效、只在最简单地形上训练）。

    改用【累计路径长度】判定后，能升级的占比从 22% → 83%（同一批数据）。

  实现上与 `mean_action_acc` 同一套路子：返回**本步增量**，
  由 MetricsManager 逐回合累加（`_episode_sums`），课程在 reset 时读取。
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")
  asset = env.scene[cfg.name]
  return torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) * env.step_dt


def get_episode_traveled_distance(env: ManagerBasedRlEnv, env_ids) -> torch.Tensor | None:
  """读取本回合累计路径长度。拿不到就返回 None（调用方回退到净位移）。"""
  mm = getattr(env, "metrics_manager", None)
  sums = getattr(mm, "_episode_sums", None)
  if sums is None or "traveled_distance" not in sums:
    return None
  return sums["traveled_distance"][env_ids]
