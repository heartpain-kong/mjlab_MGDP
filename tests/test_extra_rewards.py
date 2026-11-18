"""WMP 借鉴项：二阶动作平滑、cheat 绕障惩罚、奖励权重课程。

smoothness / cheat 直接调用真实实现；动作历史与 RewardManager 权重生效都走
mjlab 自己的类，不依赖仿真器，确认的是真实契约而不是替身行为。
"""

from types import SimpleNamespace
import math
import unittest

import torch

from mjlab.managers.action_manager import ActionManager, ActionTerm, ActionTermCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

from src.tasks.velocity.mdp.rewards import cheat, smoothness
from src.tasks.velocity.rl.runner import (
  RewardWeightCurriculum, RewardWeightSchedule, linear_reward_weight,
)


class _DummyActionTerm(ActionTerm):
  """只需要 ActionManager 接受的最小动作项，不写仿真。"""

  @property
  def action_dim(self) -> int:
    return 2

  def process_actions(self, actions):
    self._last = actions

  def apply_actions(self):
    pass

  @property
  def raw_action(self):
    return getattr(self, "_last", None)


class _DummyActionTermCfg(ActionTermCfg):
  def build(self, env):
    return _DummyActionTerm(self, env)


class _ActionEnv:
  """仅提供 ActionManager 需要的字段（num_envs/device/scene）。"""

  def __init__(self):
    self.num_envs = 2
    self.device = "cpu"
    self.scene = {"robot": object()}
    self.action_manager = ActionManager(
      {"joints": _DummyActionTermCfg(entity_name="robot")}, self
    )


class _TerrainGenerator:
  def __init__(self, proportions, num_cols):
    self.num_cols = num_cols
    self.sub_terrains = {
      name: SimpleNamespace(proportion=value) for name, value in proportions.items()
    }


class _Terrain:
  def __init__(self, proportions, num_cols, types):
    self.cfg = SimpleNamespace(terrain_generator=_TerrainGenerator(proportions, num_cols))
    self.terrain_types = torch.tensor(types)


class _Scene(dict):
  pass


def _cheat_env(headings, proportions, types, num_cols=4):
  scene = _Scene(
    robot=SimpleNamespace(
      data=SimpleNamespace(heading_w=torch.tensor(headings, dtype=torch.float32))
    )
  )
  scene.terrain = _Terrain(proportions, num_cols, types)
  return SimpleNamespace(scene=scene, device="cpu", num_envs=len(headings))


class _RewardEnv:
  num_envs = 2
  device = "cpu"
  max_episode_length_s = 1.0


def _constant_two(env):
  return torch.full((env.num_envs,), 2.0, device=env.device)


class ExtraRewardTests(unittest.TestCase):
  def test_smoothness_is_zero_for_constant_actions(self):
    env = _ActionEnv()
    for _ in range(5):
      env.action_manager.process_action(torch.full((2, 2), 0.3))
    torch.testing.assert_close(smoothness(env), torch.zeros(2))

  def test_smoothness_is_positive_for_sine_actions(self):
    env = _ActionEnv()
    for step in range(6):
      action = math.sin(0.9 * step)
      env.action_manager.process_action(torch.full((2, 2), action))
    values = smoothness(env)
    self.assertTrue(bool((values > 0.0).all()), values)

  def test_action_history_is_cleared_on_reset(self):
    env = _ActionEnv()
    env.action_manager.process_action(torch.full((2, 2), 0.3))
    env.action_manager.process_action(torch.full((2, 2), 0.4))
    env.action_manager.reset()
    torch.testing.assert_close(env.action_manager.prev_action, torch.zeros(2, 2))
    torch.testing.assert_close(env.action_manager.prev_prev_action, torch.zeros(2, 2))

  def test_cheat_only_fires_in_obstacle_columns_with_large_heading(self):
    # 列分布：4 列按 flat(0.5)/gap_steps(0.25)/slalom_brick(0.25) 切成
    # [flat, flat, gap_steps, slalom_brick]；高墙/限高架/轮胎列这里不存在，
    # 实现应自动跳过而不是报错。
    env = _cheat_env(
      headings=[1.5, -2.0, 1.2, 0.9, 1.0],
      proportions={"flat": 0.5, "gap_steps": 0.25, "slalom_brick": 0.25},
      types=[0, 1, 2, 3, 2],
    )
    torch.testing.assert_close(cheat(env), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))

  def test_cheat_heading_threshold_is_configurable(self):
    env = _cheat_env(
      headings=[1.5, -2.0, 1.2, 0.9, 1.0],
      proportions={"flat": 0.5, "gap_steps": 0.25, "slalom_brick": 0.25},
      types=[0, 1, 2, 3, 2],
    )
    torch.testing.assert_close(
      cheat(env, heading_threshold=0.5), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    )

  def test_mgdp_profile_switches_on_wmp_terms(self):
    from src.tasks.velocity.config.lgdog1.reward_cfg import make_lgdog_rewards

    kwargs = {
      "base_body": "trunk",
      "foot_sites": ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
    }
    mgdp = make_lgdog_rewards("mgdp", **kwargs)
    self.assertEqual(mgdp["smoothness"].weight, -0.01)
    self.assertEqual(mgdp["cheat"].weight, -1.0)
    # 其它 profile 默认关闭，不会凭空多出奖励项。
    flat = make_lgdog_rewards("flat", **kwargs)
    self.assertNotIn("smoothness", flat)
    self.assertNotIn("cheat", flat)

  def test_linear_reward_weight_hits_start_middle_and_end(self):
    self.assertAlmostEqual(linear_reward_weight(0, 0.1, 0.25, 3000, 8000), 0.1)
    self.assertAlmostEqual(linear_reward_weight(3000, 0.1, 0.25, 3000, 8000), 0.1)
    self.assertAlmostEqual(linear_reward_weight(5500, 0.1, 0.25, 3000, 8000), 0.175)
    self.assertAlmostEqual(linear_reward_weight(8000, 0.1, 0.25, 3000, 8000), 0.25)
    self.assertAlmostEqual(linear_reward_weight(9000, 0.1, 0.25, 3000, 8000), 0.25)

  def test_schedule_rejects_an_empty_iteration_interval(self):
    with self.assertRaises(ValueError):
      RewardWeightSchedule("feet_air_time", 0.1, 0.25, 8000, 8000)

  def test_reward_manager_reads_the_mutated_weight_at_runtime(self):
    manager = RewardManager(
      {"dummy": RewardTermCfg(func=_constant_two, weight=0.1)}, env=_RewardEnv()
    )
    curriculum = RewardWeightCurriculum(
      manager, (RewardWeightSchedule("dummy", 0.1, 0.25, 3000, 8000),)
    )
    curriculum.update(0)
    torch.testing.assert_close(manager.compute(0.5), torch.full((2,), 0.1))
    curriculum.update(8000)
    torch.testing.assert_close(manager.compute(0.5), torch.full((2,), 0.25))


if __name__ == "__main__":
  unittest.main()
