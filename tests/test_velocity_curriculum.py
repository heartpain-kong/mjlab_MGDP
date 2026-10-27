"""CPU tests for actual curriculum accumulation, gates, and resume state."""

from copy import deepcopy
from types import SimpleNamespace as NS
import unittest

import torch

from src.tasks.velocity.mdp.curriculums import RewardBasedVelocityCurriculum


class VelocityCurriculumTests(unittest.TestCase):
  def make_term(self, **overrides):
    ranges = NS(lin_vel_x=(-0.2, 0.4), lin_vel_y=(-0.15, 0.15))
    command = NS(cfg=NS(ranges=ranges))
    weights = {"track_linear_velocity": 2.0, "track_angular_velocity": 1.0}
    env = NS(num_envs=4, device="cpu", common_step_counter=0,
             step_dt=.02, max_episode_length=1000, max_episode_length_s=20.,
             command_manager=NS(get_term=lambda _: command),
             reward_manager=NS(_episode_sums={key: torch.zeros(4) for key in weights},
                               _scale_by_dt=True,
                               get_term_cfg=lambda key: NS(weight=weights[key])),
             metrics_manager=NS(_episode_sums={key: torch.zeros(4) for key in
                                             ("score", "active", "distance", "progress")}),
             episode_length_buf=torch.zeros(4, dtype=torch.long),
             termination_manager=NS(terminated=torch.zeros(4, dtype=torch.bool)))
    params = dict(command_name="twist", score_metric_name="score", active_metric_name="active",
                  progress_metric_name="progress", distance_metric_name="distance",
                  min_progress_ratio=.6, regress_min_progress_ratio=.4,
                  min_commanded_distance=1.,
                  linear_threshold=.78, linear_increment=.025, linear_patience=3,
                  linear_regress_threshold=.62, linear_regress_patience=2,
                  linear_regress_increment=.025, score_ema_alpha=1.,
                  min_episode_fraction=.8, max_failure_fraction=.25,
                  regress_min_episode_fraction=.65, regress_max_failure_fraction=.35,
                  lin_vel_x_limit=(-.4,.8), lin_vel_y_limit=(-.3,.3), check_interval_s=20.)
    params.update(overrides)
    term = RewardBasedVelocityCurriculum(NS(params=params), env)
    return env, command, term

  def feed(self, env, term, *, steps=1000, failed=0, score=.9, active_steps=None, ids=None,
           advance=1000, progress_ratio=.9, commanded_m=None, progress_m=None):
    env.common_step_counter += advance
    env.episode_length_buf[:] = steps
    env.termination_manager.terminated.zero_()
    env.termination_manager.terminated[:failed] = True
    active = steps if active_steps is None else active_steps
    env.metrics_manager._episode_sums["active"][:] = active
    env.metrics_manager._episode_sums["score"][:] = score * active
    distance = env.metrics_manager._episode_sums["distance"]
    distance[:] = active * .2 * env.step_dt if commanded_m is None else torch.as_tensor(commanded_m)
    env.metrics_manager._episode_sums["progress"][:] = (
      distance * progress_ratio if progress_m is None else torch.as_tensor(progress_m)
    )
    env.reward_manager._episode_sums["track_linear_velocity"][:] = 2 * steps * .02 * .9
    env.reward_manager._episode_sums["track_angular_velocity"][:] = steps * .02 * .9
    return term(env, torch.arange(4) if ids is None else ids)

  def test_v23_high_absolute_score_with_low_progress_cannot_expand(self):
    env, command, term = self.make_term()
    for _ in range(6):
      metrics = self.feed(env, term, steps=870, failed=1, score=.825, progress_ratio=.347)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.assertEqual(metrics["score_gate_pass"], 1.)
    self.assertEqual(metrics["progress_gate_pass"], 0.)
    self.assertEqual(metrics["promotion_allowed"], 0.)

  def test_stationary_robot_scoring_above_threshold_cannot_expand(self):
    env, command, term = self.make_term()
    # Absolute tracking at command 0.2 m/s scores 0.852 even at actual speed 0.
    for _ in range(4): self.feed(env, term, score=.8521438, progress_ratio=0.)
    self.assertTrue(term.score_gate_pass)
    self.assertEqual(float(term.progress_ratio), 0.)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))

  def test_progress_is_weighted_by_commanded_meters_not_episode_average(self):
    env, command, term = self.make_term(linear_patience=1)
    metrics = self.feed(env, term, commanded_m=[10.,1.,1.,1.], progress_m=[2.,1.,1.,1.])
    # Mean of episode ratios is 0.8, but actual completed distance is only 5/13.
    self.assertAlmostEqual(float(metrics["progress_ratio"]), 5/13, places=6)
    self.assertFalse(term.promotion_allowed)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))

  def test_poor_progress_regresses_even_with_good_score_and_survival(self):
    env, command, term = self.make_term()
    for _ in range(6): self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.25,.45))
    self.feed(env, term, progress_ratio=.3)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.25,.45))
    self.feed(env, term, progress_ratio=.3)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))

  def test_progress_hysteresis_holds_and_restarts_promotion_streak(self):
    env, command, term = self.make_term()
    for _ in range(3): self.feed(env, term)
    for _ in range(8): self.feed(env, term, progress_ratio=.5)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))
    self.assertEqual(term.promotion_streak, 0)
    self.assertEqual(term.regression_streak, 0)
    for _ in range(2): self.feed(env, term, progress_ratio=.65)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))
    self.feed(env, term, progress_ratio=.65)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.25,.45))

  def test_negligible_moving_distance_cannot_promote(self):
    env, command, term = self.make_term()
    for _ in range(4): self.feed(env, term, commanded_m=.01, progress_ratio=1.)
    self.assertAlmostEqual(float(term.progress_ratio), 1.)
    self.assertFalse(term.progress_gate_pass)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))

  def test_callers_without_progress_gate_keep_legacy_behavior(self):
    env, command, term = self.make_term(progress_metric_name=None, distance_metric_name=None,
                                       min_progress_ratio=0., regress_min_progress_ratio=0.)
    for _ in range(3): self.feed(env, term, progress_ratio=0.)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))

  def test_v21_like_good_score_with_half_failures_does_not_promote(self):
    env, command, term = self.make_term()
    for _ in range(5):
      metrics = self.feed(env, term, steps=600, failed=2, score=.82)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.assertEqual(metrics["promotion_allowed"], 0.)
    self.assertAlmostEqual(float(metrics["linear_score"]), .82, places=6)
    self.assertAlmostEqual(float(metrics["failure_fraction"]), .5)

  def test_safe_tracking_promotes_only_after_consecutive_checks_and_caps_ranges(self):
    env, command, term = self.make_term()
    for _ in range(2): self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))
    for _ in range(90): self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.4,.8))
    self.assertEqual(command.cfg.ranges.lin_vel_y, (-.3,.3))

  def test_short_survival_without_failure_also_blocks_promotion(self):
    env, command, term = self.make_term()
    for _ in range(4): self.feed(env, term, steps=700, failed=0)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.assertFalse(term.promotion_allowed)

  def test_failure_regresses_high_score_and_stops_at_initial_range(self):
    env, command, term = self.make_term()
    for _ in range(6): self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.25,.45))
    self.feed(env, term, steps=900, failed=2)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.25,.45))
    self.feed(env, term, steps=900, failed=2)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))
    for _ in range(10): self.feed(env, term, failed=3)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))

  def test_hysteresis_holds_range_between_safety_gates(self):
    env, command, term = self.make_term()
    for _ in range(3): self.feed(env, term)
    for _ in range(8): self.feed(env, term, steps=750, failed=1)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))

  def test_all_standing_commands_cannot_promote(self):
    env, command, term = self.make_term()
    for _ in range(4): self.feed(env, term, active_steps=0)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.assertFalse(term.promotion_allowed)

  def test_partial_resets_accumulate_actual_episode_count_until_check(self):
    env, _, term = self.make_term(linear_patience=1)
    self.feed(env, term, steps=1000, failed=0, ids=torch.tensor([0,1]), advance=500)
    self.assertEqual(term.episode_count, 2)
    metrics = self.feed(env, term, steps=500, failed=4, ids=torch.tensor([2,3]), advance=500)
    self.assertAlmostEqual(float(metrics["episode_fraction"]), .75)
    self.assertAlmostEqual(float(metrics["failure_fraction"]), .5)
    self.assertFalse(term.promotion_allowed)
    self.assertEqual(term.episode_count, 0)
    self.assertEqual(float(term.episode_steps_sum), 0.)

  def test_interrupted_promotion_streak_must_restart(self):
    env, command, term = self.make_term()
    for _ in range(2): self.feed(env, term)
    self.feed(env, term, steps=750)
    self.assertEqual(term.promotion_streak, 0)
    for _ in range(2): self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.2,.4))
    self.feed(env, term)
    self.assertEqual(command.cfg.ranges.lin_vel_x, (-.225,.425))

  def test_missing_direct_metrics_never_falls_back_to_standing_reward(self):
    env, _, term = self.make_term()
    env.common_step_counter = 1
    env.episode_length_buf.fill_(1)
    del env.metrics_manager._episode_sums["score"]
    with self.assertRaisesRegex(ValueError, "metrics are missing"):
      term(env, torch.arange(env.num_envs))

  def test_checkpoint_roundtrip_preserves_partial_window(self):
    env, command, term = self.make_term(linear_patience=1)
    self.feed(env, term, steps=1000, failed=0, ids=torch.tensor([0,1]), advance=500)
    saved = term.state_dict()
    new_env, new_command, restored = self.make_term(linear_patience=1)
    new_env.common_step_counter = env.common_step_counter
    restored.load_state_dict(saved)
    self.assertEqual(restored.episode_count, 2)
    self.assertEqual(float(restored.episode_steps_sum), 2000.)
    torch.testing.assert_close(restored.commanded_distance_sum, term.commanded_distance_sum)
    torch.testing.assert_close(restored.tracking_progress_sum, term.tracking_progress_sum)
    self.feed(env, term, steps=500, failed=4, ids=torch.tensor([2,3]), advance=500)
    self.feed(new_env, restored, steps=500, failed=4, ids=torch.tensor([2,3]), advance=500)
    self.assertEqual(command.cfg.ranges.lin_vel_x, new_command.cfg.ranges.lin_vel_x)
    torch.testing.assert_close(term.episode_fraction, restored.episode_fraction)
    torch.testing.assert_close(term.failure_fraction, restored.failure_fraction)
    torch.testing.assert_close(term.progress_ratio, restored.progress_ratio)

  def test_v1_checkpoint_discards_unverified_partial_window_and_streak(self):
    env, _, term = self.make_term()
    self.feed(env, term)
    state=term.state_dict()
    state.update(version=1, promotion_streak=2, episode_count=3, last_check_step=0)
    for key in ("episode_steps_sum", "failed_episode_count", "episode_fraction", "failure_fraction"):
      state.pop(key)
    new_env, _, restored = self.make_term()
    new_env.common_step_counter=1000
    restored.load_state_dict(state)
    self.assertEqual(restored.episode_count, 0)
    self.assertEqual(restored.promotion_streak, 0)
    self.assertEqual(restored.last_check_step, 1000)
    self.assertFalse(restored.promotion_allowed)

  def test_v2_checkpoint_cannot_reuse_streak_without_progress_statistics(self):
    env, command, term = self.make_term()
    for _ in range(3): self.feed(env, term)
    state=term.state_dict()
    state.update(version=2, promotion_streak=2, episode_count=3)
    for key in ("commanded_distance_sum", "tracking_progress_sum", "progress_ratio"):
      state.pop(key)
    new_env, new_command, restored = self.make_term()
    new_env.common_step_counter=env.common_step_counter
    restored.load_state_dict(state)
    self.assertEqual(new_command.cfg.ranges.lin_vel_x, command.cfg.ranges.lin_vel_x)
    self.assertEqual(restored.episode_count, 0)
    self.assertEqual(restored.promotion_streak, 0)
    self.assertEqual(float(restored.commanded_distance_sum), 0.)
    self.assertFalse(restored.progress_gate_pass)
    for _ in range(3): self.feed(new_env, restored, progress_ratio=.3)
    self.assertEqual(new_command.cfg.ranges.lin_vel_x, (-.2,.4))

  def test_missing_progress_metrics_are_rejected(self):
    for name in ("distance", "progress"):
      env, _, term = self.make_term()
      env.common_step_counter=1
      env.episode_length_buf.fill_(1)
      del env.metrics_manager._episode_sums[name]
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, "progress metrics are missing"):
        term(env, torch.arange(4))

  def test_invalid_gate_and_checkpoint_statistics_are_rejected(self):
    for kwargs in ({"min_episode_fraction": float('nan')}, {"max_failure_fraction":1.2},
                   {"regress_max_failure_fraction": .2}, {"active_metric_name":None},
                   {"min_progress_ratio": float('nan')}, {"min_progress_ratio": 1.2},
                   {"regress_min_progress_ratio": .7}, {"distance_metric_name": None},
                   {"min_commanded_distance": 0.},
                   {"distance_metric_name": None, "progress_metric_name": None}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.make_term(**kwargs)
    env, _, term = self.make_term()
    saved=term.state_dict()
    for name,value in (("failure_fraction",2.0), ("episode_fraction",float('nan')),
                       ("failed_episode_count",1.0), ("episode_steps_sum",-1.0),
                       ("progress_ratio",1.5), ("commanded_distance_sum",float('nan')),
                       ("tracking_progress_sum",2.)):
      state=deepcopy(saved);state[name]=value
      with self.subTest(name=name), self.assertRaises(ValueError): term.load_state_dict(state)


if __name__ == '__main__': unittest.main()
