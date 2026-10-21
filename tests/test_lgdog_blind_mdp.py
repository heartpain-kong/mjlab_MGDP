"""CPU regression tests. Run: python -m unittest discover -s tests -v"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


# Load tensor terms directly, without importing the GPU task registry.
PATH = Path(__file__).resolve().parents[1] / "src/tasks/velocity/config/lgdog1/blind_mdp.py"
SPEC = importlib.util.spec_from_file_location("lgdog_blind_mdp_test", PATH)
mdp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mdp)


def ray_env(foot_z, ground_z, distance):
  return NS(scene={"foot": NS(data=NS(
    pos_w=torch.tensor([[0., 0., foot_z]]),
    hit_pos_w=torch.tensor([[[0., 0., ground_z]]]),
    distances=torch.tensor([[distance]]),
  ))})


class ClearanceTests(unittest.TestCase):
  def test_world_translation_invariance(self):
    flat, _ = mdp.foot_clearance_relative(ray_env(.15, 0., .65), ["foot"])
    raised, _ = mdp.foot_clearance_relative(ray_env(1.15, 1., .65), ["foot"])
    torch.testing.assert_close(flat, raised)

  def test_stair_height_is_subtracted(self):
    h, valid = mdp.foot_clearance_relative(ray_env(.25, .20, .55), ["foot"])
    self.assertAlmostEqual(h.item(), .05)
    self.assertTrue(valid.item())

  def test_ray_miss_and_nan_are_masked(self):
    h, valid = mdp.foot_clearance_relative(ray_env(.25, float("nan"), -1.), ["foot"])
    result = mdp.clearance_cost(h, torch.ones_like(h), torch.zeros_like(valid),
                                valid, torch.tensor([True]), .1)
    self.assertEqual(result.item(), 0.)

  def test_stance_and_standing_are_not_penalized(self):
    h = torch.zeros(2, 2)
    cost = mdp.clearance_cost(h, torch.ones_like(h),
      torch.tensor([[True, False], [False, False]]), torch.ones_like(h, dtype=torch.bool),
      torch.tensor([True, False]), .1)
    torch.testing.assert_close(cost, torch.tensor([.1, 0.]))


class CurriculumTests(unittest.TestCase):
  def test_success_failure_hysteresis_and_zero_command(self):
    expected = torch.tensor([5., 5., 5., 0., 5., 5.])
    progress = torch.tensor([4.5, 1., 2.5, 8., 5., 5.])
    exposure = torch.tensor([2., 2., 2., 2., 0., 2.])
    elapsed = torch.full((6,), 20.)
    failed = torch.tensor([False, False, False, False, False, True])
    up, down, _ = mdp.terrain_decisions(expected, progress, exposure, elapsed, failed, 20.)
    self.assertEqual(up.tolist(), [True, False, False, False, False, False])
    self.assertEqual(down.tolist(), [False, True, False, False, False, True])

  def test_short_episode_cannot_promote(self):
    up, _, _ = mdp.terrain_decisions(torch.tensor([3.]), torch.tensor([3.]),
      torch.tensor([2.]), torch.tensor([5.]), torch.tensor([False]), 20.)
    self.assertFalse(up.item())

  def test_velocity_quality_gate_blocks_promotion(self):
    args = (
      torch.tensor([5.]), torch.tensor([4.5]), torch.tensor([2.]),
      torch.tensor([20.]), torch.tensor([False]), 20.,
    )
    up_bad, _, _ = mdp.terrain_decisions(*args, velocity_error=torch.tensor([.9]))
    up_good, _, _ = mdp.terrain_decisions(*args, velocity_error=torch.tensor([.7]))
    self.assertFalse(up_bad.item())
    self.assertTrue(up_good.item())

  def test_progress_ignores_drift_backwards_and_overspeed(self):
    command = torch.tensor([[1., 0., 0.]] * 4)
    env = NS(step_dt=.02, scene={"robot": NS(data=NS(
      root_link_lin_vel_b=torch.tensor([[0., 1., 0.], [-1., 0., 0.], [3., 0., 0.], [.5, 0., 0.]])))},
      command_manager=NS(get_command=lambda _: command))
    torch.testing.assert_close(mdp.tracking_progress(env), torch.tensor([0., 0., .02, .01]))

  def test_command_resampling_integrates_each_segment(self):
    command = torch.tensor([[1., 0., 0.]])
    env = NS(step_dt=1., command_manager=NS(get_command=lambda _: command),
      scene={"robot": NS(data=NS(root_link_lin_vel_b=torch.tensor([[1., 0., 0.]])))})
    expected = mdp.commanded_distance(env)
    progress = mdp.tracking_progress(env)
    command[:] = torch.tensor([[0., 1., 0.]])
    env.scene["robot"].data.root_link_lin_vel_b[:] = torch.tensor([[0., 1., 0.]])
    expected += mdp.commanded_distance(env)
    progress += mdp.tracking_progress(env)
    command.zero_()  # final standing command must not erase previous demand
    expected += mdp.commanded_distance(env)
    self.assertEqual(expected.item(), 2.)
    self.assertEqual(progress.item(), 2.)

  def test_reset_uses_completed_sums_and_only_selected_envs(self):
    selected = []
    terrain = NS(terrain_levels=torch.tensor([1, 2]),
      update_env_origins=lambda ids, up, down: selected.append((ids.tolist(), up.tolist(), down.tolist())))
    env = NS(metrics_manager=NS(_episode_sums={
      "commanded_distance": torch.tensor([5., 0.]), "tracking_progress": torch.tensor([5., 0.]),
      "outside_spawn_platform": torch.tensor([2., 0.])}),
      termination_manager=NS(terminated=torch.tensor([False, False])),
      episode_length_buf=torch.tensor([1000, 0]), step_dt=.02, max_episode_length_s=20.,
      scene=NS(terrain=terrain))
    result = mdp.terrain_levels_tracking(env, torch.tensor([0]))
    self.assertEqual(selected, [([0], [True], [False])])
    self.assertEqual(result["episode_progress_m"].item(), 5.)
    self.assertEqual(env.metrics_manager._episode_sums["tracking_progress"][0].item(), 5.)

  def test_legacy_curriculum_can_disable_velocity_gate(self):
    expected = torch.tensor([5.])
    progress = torch.tensor([4.5])
    exposure = torch.tensor([2.])
    elapsed = torch.tensor([20.])
    failed = torch.tensor([False])
    up, _, _ = mdp.terrain_decisions(
      expected, progress, exposure, elapsed, failed, 20.,
      velocity_error=None,
    )
    self.assertTrue(up.item())


class ContactTests(unittest.TestCase):
  def setUp(self):
    self.force = torch.zeros(2, 2, 4, 3)
    self.env = NS(step_dt=.02, scene={"legs": NS(data=NS(force_history=self.force))})
    self.term = mdp.sustained_leg_contact(NS(params={"duration_s": .1}), self.env)

  def test_brief_scrape_recovers(self):
    self.force[0, 0, :, 2] = 15.
    self.assertFalse(self.term(self.env, "legs")[0].item())
    self.force.zero_()
    self.assertFalse(self.term(self.env, "legs")[0].item())
    self.assertEqual(self.term.count[0, 0].item(), 0)

  def test_persistent_contact_and_partial_reset(self):
    self.force[:, 0, :, 2] = 15.
    for _ in range(4):
      self.assertFalse(self.term(self.env, "legs").any().item())
    self.assertTrue(self.term(self.env, "legs").all().item())
    self.term.reset(torch.tensor([0]))
    self.assertEqual(self.term.count[0].sum().item(), 0)
    self.assertEqual(self.term.count[1, 0].item(), 5)

  def test_severe_substep_hit_terminates_immediately(self):
    self.force[0, 1, 2, 0] = 81.
    self.assertEqual(self.term(self.env, "legs").tolist(), [True, False])


if __name__ == "__main__":
  unittest.main()
