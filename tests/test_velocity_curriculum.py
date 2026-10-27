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
             metrics_manager=NS(_episode_sums={"score": torch.zeros(4), "active": torch.zeros(4)}),
             episode_length_buf=torch.zeros(4, dtype=torch.long),
             termination_manager=NS(terminated=torch.zeros(4, dtype=torch.bool)))
    params = dict(command_name="twist", score_metric_name="score", active_metric_name="active",
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
           advance=1000):
    env.common_step_counter += advance
    env.episode_length_buf[:] = steps
    env.termination_manager.terminated.zero_()
    env.termination_manager.terminated[:failed] = True
    active = steps if active_steps is None else active_steps
    env.metrics_manager._episode_sums["active"][:] = active
    env.metrics_manager._episode_sums["score"][:] = score * active
    env.reward_manager._episode_sums["track_linear_velocity"][:] = 2 * steps * .02 * .9
    env.reward_manager._episode_sums["track_angular_velocity"][:] = steps * .02 * .9
    return term(env, torch.arange(4) if ids is None else ids)

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
    self.feed(env, term, steps=500, failed=4, ids=torch.tensor([2,3]), advance=500)
    self.feed(new_env, restored, steps=500, failed=4, ids=torch.tensor([2,3]), advance=500)
    self.assertEqual(command.cfg.ranges.lin_vel_x, new_command.cfg.ranges.lin_vel_x)
    torch.testing.assert_close(term.episode_fraction, restored.episode_fraction)
    torch.testing.assert_close(term.failure_fraction, restored.failure_fraction)

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

  def test_invalid_gate_and_checkpoint_statistics_are_rejected(self):
    for kwargs in ({"min_episode_fraction": float('nan')}, {"max_failure_fraction":1.2},
                   {"regress_max_failure_fraction": .2}, {"active_metric_name":None}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.make_term(**kwargs)
    env, _, term = self.make_term()
    saved=term.state_dict()
    for name,value in (("failure_fraction",2.0), ("episode_fraction",float('nan')),
                       ("failed_episode_count",1.0), ("episode_steps_sum",-1.0)):
      state=deepcopy(saved);state[name]=value
      with self.subTest(name=name), self.assertRaises(ValueError): term.load_state_dict(state)


if __name__ == '__main__': unittest.main()
