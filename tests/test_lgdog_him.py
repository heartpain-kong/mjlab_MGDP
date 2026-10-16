"""Behavioral tests for HIM identification, history and checkpoint contracts."""

import copy
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from src.policy_history import PolicyHistory, validate_policy_history
from src.tasks.velocity.rl.him import HIMActor, HIMPPO
from src.tasks.velocity.rl.him_estimator import balanced_assignments
from mjlab.utils.buffers import CircularBuffer


def make_obs(batch=8):
  return TensorDict({"actor": torch.randn(batch, 6, 47), "critic": torch.randn(batch, 261),
                    "him_target": torch.randn(batch, 47), "him_velocity": torch.randn(batch, 3)},
                   batch_size=[batch])


def make_algorithm():
  obs = make_obs()
  groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = HIMActor(obs, groups, "actor", 12, hidden_dims=(32, 16), obs_normalization=True,
                   distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                     "std_type": "scalar"})
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(32, 16))
  storage = RolloutStorage("rl", 8, 2, obs, [12])
  alg = HIMPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2)
  return alg, obs


def rollout(alg, obs, all_done=False):
  with torch.inference_mode():
    for _ in range(2):
      alg.act(obs)
      successor = make_obs()
      dones = torch.ones(8, dtype=torch.long) if all_done else torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
      alg.process_env_step(successor, torch.ones(8), dones, {})
      obs = successor
    alg.compute_returns(obs)


class HistoryTests(unittest.TestCase):
  def test_numpy_buffer_matches_mjlab_and_backfill(self):
    deploy = PolicyHistory()
    training = CircularBuffer(6, 1, "cpu")
    for step in range(10):
      frame = np.arange(47, dtype=np.float32) + step * 100
      training.append(torch.from_numpy(frame[None]))
      np.testing.assert_array_equal(deploy.push(frame), training.buffer[0].numpy().reshape(-1))
    deploy.reset()
    training.reset()
    frame = np.full(47, 99, dtype=np.float32)
    training.append(torch.from_numpy(frame[None]))
    np.testing.assert_array_equal(deploy.push(frame), training.buffer[0].numpy().reshape(-1))

  def test_contract_rejects_wrong_order_and_old_policy(self):
    metadata = {"policy_architecture": "lgdog_him_v1", "history_length": "6",
                "frame_dim": "47", "history_order": "oldest_to_newest",
                "history_reset": "repeat_first_frame", "control_dt": "0.02"}
    validate_policy_history([1, 282], metadata, 6)
    with self.assertRaises(ValueError):
      validate_policy_history([1, 47], metadata, 6)
    metadata["history_order"] = "newest_to_oldest"
    with self.assertRaises(ValueError):
      validate_policy_history([1, 282], metadata, 6)
    validate_policy_history([1, 47], {}, 1)  # original policies still work

  def test_bad_frame_rejected(self):
    with self.assertRaises(ValueError):
      PolicyHistory().push(np.full(47, np.nan))


class HIMTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(17)

  def test_sinkhorn_finite_probabilities_under_large_scores(self):
    q = balanced_assignments(torch.randn(64, 32) * 1000)
    self.assertTrue(torch.isfinite(q).all())
    torch.testing.assert_close(q.sum(-1), torch.ones(64), atol=2e-3, rtol=2e-3)
    uniform = balanced_assignments(torch.zeros(64, 32))
    torch.testing.assert_close(uniform.sum(0), torch.full((32,), 2.))

  def test_actor_has_no_privileged_dependency(self):
    alg, obs = make_algorithm()
    expected = alg.actor(obs)
    for group in ("critic", "him_target", "him_velocity"):
      obs[group].fill_(1000)
    torch.testing.assert_close(alg.actor(obs), expected)

  def test_policy_backward_does_not_train_estimator(self):
    alg, obs = make_algorithm()
    alg.actor(obs).square().mean().backward()
    self.assertTrue(any(p.grad is not None for p in alg.actor.mlp.parameters()))
    self.assertTrue(all(p.grad is None for p in alg.actor.estimator.parameters()))
    ppo_ids = {id(p) for group in alg.optimizer.param_groups for p in group["params"]}
    self.assertFalse(ppo_ids & {id(p) for p in alg.actor.estimator.parameters()})

  def test_auxiliary_losses_train_encoder_target_and_prototypes(self):
    alg, obs = make_algorithm()
    e = alg.actor.estimator
    v, s = e.losses(obs["actor"], obs["him_target"], obs["him_velocity"])
    (v + s).backward()
    for parameters in (e.encoder.parameters(), e.target.parameters(), [e.prototypes]):
      self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))

  def test_export_parity_with_nontrivial_normalization(self):
    alg, obs = make_algorithm()
    alg.actor.update_normalization(obs)
    torch.testing.assert_close(alg.actor(obs), alg.actor.as_onnx()(obs["actor"].flatten(1)))

  def test_terminal_transitions_masked_and_auxiliary_updates(self):
    alg, obs = make_algorithm()
    before = [p.detach().clone() for p in alg.actor.estimator.parameters()]
    rollout(alg, obs)
    self.assertEqual(alg.identification_valid[:, 0].tolist(), [False, False])
    losses = alg.update()
    self.assertEqual(losses["him_valid_fraction"], 7 / 8)
    self.assertEqual(losses["him_updates"], 2)
    self.assertTrue(all(np.isfinite(x) for x in losses.values()))
    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, alg.actor.estimator.parameters())))
    self.assertTrue(all(p.grad is None for p in alg.actor.estimator.parameters()))

  def test_all_terminal_batch_skips_identification(self):
    alg, obs = make_algorithm()
    before = copy.deepcopy(alg.actor.estimator.state_dict())
    rollout(alg, obs, all_done=True)
    losses = alg.update()
    self.assertEqual(losses["him_updates"], 0)
    for name, tensor in before.items():
      torch.testing.assert_close(tensor, alg.actor.estimator.state_dict()[name])

  def test_checkpoint_restores_estimator_optimizer_and_actor(self):
    alg, obs = make_algorithm()
    rollout(alg, obs)
    alg.update()
    saved = copy.deepcopy(alg.save())
    loaded, _ = make_algorithm()
    self.assertTrue(loaded.load(saved, None, True))
    torch.testing.assert_close(alg.actor(obs), loaded.actor(obs))
    original_state = alg.estimator_optimizer.state_dict()["state"]
    restored_state = loaded.estimator_optimizer.state_dict()["state"]
    self.assertEqual(original_state.keys(), restored_state.keys())
    for key in original_state:
      for field in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(original_state[key][field], restored_state[key][field])
    saved.pop("him_contract")
    with self.assertRaises(ValueError):
      loaded.load(saved, None, True)


if __name__ == "__main__":
  unittest.main()
