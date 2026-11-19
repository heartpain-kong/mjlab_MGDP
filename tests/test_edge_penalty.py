"""``edge_place_penalty`` 的 CPU 单测：假 env + 假 height_scan 张量。

覆盖需求里的四个场景，外加两条实现契约：
  · 平地网格（高度全等）                → 0
  · 脚落在台阶棱上（相邻格差 0.1 m）    → >0，且只有踩棱的那只脚计数
  · 环境没有 height_scan 传感器         → 全 0，不抛错
  · 脚未触地（found=0 / 接触力 0）      → 0
  · 直接传 height_scan + grid 形状参数  → 与传感器路径一致
  · margin / height_jump_threshold 可调 → 判定窗口与阈值真正生效

扫描布局与 mjlab ``terrain_scan`` 一致（``GridPatternCfg``）：
1.6 m x 1.0 m、0.1 m 分辨率 ⇒ ``meshgrid(x, y, indexing="xy")`` 展平成 17x11=187 点。
水平坐标世界对齐，所以假 ``hit_pos_w`` 的 XY 直接就是网格点。
"""

from types import SimpleNamespace as NS
import unittest

import torch

from src.tasks.velocity.mdp.rewards import edge_place_penalty


GRID_X = torch.linspace(-0.8, 0.8, 17)  # 1.6 m / 0.1 m + 1
GRID_Y = torch.linspace(-0.5, 0.5, 11)  # 1.0 m / 0.1 m + 1
SCAN_HEIGHT_Z = 0.45  # 传感器离地高度（heights = pos_w.z - hit_z）
FOOT_STEP_M = 0.1


def _scan_xy() -> torch.Tensor:
  """187 个采样点的世界 XY，顺序同 mjlab GridPatternCfg。"""
  grid_x, grid_y = torch.meshgrid(GRID_X, GRID_Y, indexing="xy")
  return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [187, 2]


SCAN_XY = _scan_xy()


def _flat_ground(num_envs: int) -> torch.Tensor:
  return torch.zeros(num_envs, SCAN_XY.shape[0])


def _step_ground(num_envs: int) -> torch.Tensor:
  """x >= 0 的半个平面高 0.1 m（假台阶），其余为 0。"""
  ground = torch.zeros(num_envs, SCAN_XY.shape[0])
  high = SCAN_XY[:, 0] >= -1e-6
  ground[:, high] = FOOT_STEP_M
  return ground


def _contact_force(num_envs: int, on_ground: bool = True) -> torch.Tensor:
  force = torch.zeros(num_envs, 4, 3)
  if on_ground:
    force[..., 2] = 30.0  # 约 3 kg 承重，远大于默认 min_force_n=1.0
  return force


def _fake_env(
  foot_xy: torch.Tensor,
  ground_z: torch.Tensor | None = None,
  foot_force: torch.Tensor | None = None,
  with_scan: bool = True,
  with_contact: bool = True,
) -> NS:
  """构造只要 ``edge_place_penalty`` 用到的字段的假环境。

  foot_xy: [B, 4, 3] 足端 site 世界位置；ground_z: [B, 187] 地形高度；
  foot_force: [B, 4, 3] 足端接触力（found 由力是否非零推出）。
  """
  num_envs = int(foot_xy.shape[0])
  scene = {
    "robot": NS(
      data=NS(site_pos_w=foot_xy),
      find_sites=lambda names: (list(range(len(tuple(names)))), list(names)),
    ),
  }
  if with_scan:
    assert ground_z is not None
    hit_pos_w = torch.zeros(num_envs, SCAN_XY.shape[0], 3)
    hit_pos_w[:, :, :2] = SCAN_XY
    hit_pos_w[:, :, 2] = ground_z
    scene["terrain_scan"] = NS(
      data=NS(
        pos_w=torch.full((num_envs, 3), SCAN_HEIGHT_Z),
        hit_pos_w=hit_pos_w,
        distances=torch.ones(num_envs, SCAN_XY.shape[0]),
      ),
      cfg=NS(pattern=NS(size=(1.6, 1.0), resolution=0.1)),
    )
  if with_contact:
    force = foot_force if foot_force is not None else _contact_force(num_envs)
    scene["feet_ground_contact"] = NS(
      data=NS(found=(force.norm(dim=-1) > 0).to(torch.long), force=force)
    )
  return NS(num_envs=num_envs, device="cpu", scene=scene, extras={"log": {}})


class EdgePlacePenaltyTests(unittest.TestCase):
  def test_flat_terrain_returns_zero(self):
    # 四只脚都站在平地网格上；高度全等 ⇒ 局部极差 0。
    foot = torch.full((1, 4, 3), -0.4)
    foot[..., 2] = 0.05
    env = _fake_env(foot, _flat_ground(1), _contact_force(1))
    out = edge_place_penalty(env)
    self.assertEqual(tuple(out.shape), (1,))
    torch.testing.assert_close(out, torch.zeros(1))

  def test_step_edge_counts_only_the_foot_on_the_nose(self):
    # env0：四脚都在平地；env1：第一只脚踩在 x=0 的台阶棱上，其余仍在平地。
    foot = torch.full((2, 4, 3), -0.4)
    foot[..., 2] = 0.05
    foot[1, 0, :2] = torch.tensor([0.0, 0.0])
    env = _fake_env(foot, _step_ground(2), _contact_force(2))
    out = edge_place_penalty(env)
    self.assertEqual(tuple(out.shape), (2,))
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0]))

  def test_no_height_scan_returns_zeros_without_raising(self):
    foot = torch.full((2, 4, 3), -0.4)
    foot[..., 2] = 0.05
    foot[:, 0, :2] = torch.tensor([0.0, 0.0])  # 就算站在棱上也该返回 0
    env = _fake_env(foot, _step_ground(2), _contact_force(2), with_scan=False)
    out = edge_place_penalty(env)
    torch.testing.assert_close(out, torch.zeros(2))

  def test_no_foot_contact_returns_zero(self):
    # 脚踩在棱上，但 found=0 / 接触力 0 ⇒ 没有触地，不罚。
    foot = torch.zeros(1, 4, 3)
    foot[..., 2] = 0.05
    env = _fake_env(foot, _step_ground(1), _contact_force(1, on_ground=False))
    out = edge_place_penalty(env)
    torch.testing.assert_close(out, torch.zeros(1))

  def test_explicit_height_scan_and_grid_parameters_match_sensor_path(self):
    # 没有 terrain_scan 传感器，直接用 height_scan [B, 187] + grid 形状参数。
    foot = torch.full((1, 4, 3), -0.4)
    foot[..., 2] = 0.05
    foot[0, 0, :2] = torch.tensor([0.0, 0.0])  # 只有这只脚踩在棱上
    heights = SCAN_HEIGHT_Z - _step_ground(1)
    env = _fake_env(foot, _step_ground(1), _contact_force(1), with_scan=False)
    out = edge_place_penalty(
      env, height_scan=heights, grid_size=(1.6, 1.0), resolution=0.1
    )
    torch.testing.assert_close(out, torch.tensor([1.0]))

  def test_explicit_height_scan_overrides_sensor_data(self):
    # 传感器是台阶，但显式传入平地高度 ⇒ 按传入的 height_scan 判定。
    foot = torch.zeros(1, 4, 3)
    foot[..., 2] = 0.05
    env = _fake_env(foot, _step_ground(1), _contact_force(1))
    flat_heights = torch.full((1, SCAN_XY.shape[0]), SCAN_HEIGHT_Z)
    out = edge_place_penalty(env, height_scan=flat_heights)
    torch.testing.assert_close(out, torch.zeros(1))

  def test_feet_outside_the_scan_footprint_are_not_penalized(self):
    # 扫描只覆盖 ±0.8/±0.5 m；脚跑到网格外时窗口里没有采样点，应计 0 而不是报错。
    foot = torch.full((1, 4, 3), 3.0)
    foot[..., 2] = 0.05
    env = _fake_env(foot, _step_ground(1), _contact_force(1))
    torch.testing.assert_close(edge_place_penalty(env), torch.zeros(1))

  def test_margin_and_jump_threshold_are_configurable(self):
    # 脚离台阶棱 0.15 m：默认窗口(margin .05 + 半格 .05 = .1)够不到；
    # 放大 margin 到 .15 后能同时看到棱两侧。
    foot = torch.full((1, 4, 3), -0.4)
    foot[..., 2] = 0.05
    foot[0, 0, :2] = torch.tensor([-0.15, -0.4])
    env = _fake_env(foot, _step_ground(1), _contact_force(1))
    torch.testing.assert_close(edge_place_penalty(env), torch.tensor([0.0]))
    torch.testing.assert_close(
      edge_place_penalty(env, margin_m=0.15), torch.tensor([1.0])
    )
    # 棱上的脚（0.1 m 跃变）：阈值 0.03 触发，0.2 不触发。
    edge_foot = torch.full((1, 4, 3), -0.4)
    edge_foot[..., 2] = 0.05
    edge_foot[0, 0, :2] = torch.tensor([0.0, 0.0])
    edge_env = _fake_env(edge_foot, _step_ground(1), _contact_force(1))
    torch.testing.assert_close(edge_place_penalty(edge_env), torch.tensor([1.0]))
    torch.testing.assert_close(
      edge_place_penalty(edge_env, height_jump_threshold_m=0.2),
      torch.tensor([0.0]),
    )


if __name__ == "__main__":
  unittest.main()
