"""HIM contract checks that do not require the simulation stack."""

from types import SimpleNamespace
import unittest

from src.him_contract import (
  HIM_FRAME_DIM, HIM_HISTORY_LENGTH, HIM_INPUT_DIM, LGDOG_HIM_LAYOUT,
  apply_him_play_contract, make_him_checkpoint_contract,
)


class HimContractTests(unittest.TestCase):
  def test_expected_frame_and_history_shape(self):
    self.assertEqual((HIM_FRAME_DIM, HIM_HISTORY_LENGTH, HIM_INPUT_DIM), (47, 3, 141))
    LGDOG_HIM_LAYOUT.validate_terms(("base_ang_vel", "projected_gravity", "command",
                                     "phase", "joint_pos", "joint_vel", "actions"))

  def test_reordered_or_extra_terms_are_rejected(self):
    terms = [name for name, _ in LGDOG_HIM_LAYOUT.terms]
    terms[0], terms[1] = terms[1], terms[0]
    with self.assertRaises(ValueError):
      LGDOG_HIM_LAYOUT.validate_terms(terms)
    with self.assertRaises(ValueError):
      LGDOG_HIM_LAYOUT.validate_terms([*terms, "height_scan"])

  def test_legacy_six_frame_checkpoint_playback(self):
    env = SimpleNamespace(observations={"actor": SimpleNamespace(history_length=3)})
    agent = SimpleNamespace(actor=SimpleNamespace(history_length=3))
    contract = make_him_checkpoint_contract(47, 6)
    self.assertEqual(contract, {"frame_dim": 47, "history_length": 6,
                                "order": "oldest_to_newest"})
    self.assertEqual(apply_him_play_contract(env, agent, contract), 6)
    self.assertEqual(env.observations["actor"].history_length, 6)
    self.assertEqual(agent.actor.history_length, 6)
    self.assertEqual(HIM_HISTORY_LENGTH, 3)

  def test_invalid_checkpoint_does_not_modify_configuration(self):
    env = SimpleNamespace(observations={"actor": SimpleNamespace(history_length=3)})
    agent = SimpleNamespace(actor=SimpleNamespace(history_length=3))
    for contract in (None, make_him_checkpoint_contract(46, 6),
                     make_him_checkpoint_contract(47, True),
                     {"frame_dim": 47, "history_length": 6, "order": "newest_to_oldest"}):
      with self.assertRaises(ValueError):
        apply_him_play_contract(env, agent, contract)
      self.assertEqual(env.observations["actor"].history_length, 3)
      self.assertEqual(agent.actor.history_length, 3)


if __name__ == "__main__":
  unittest.main()
