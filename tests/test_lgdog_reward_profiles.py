"""Check profile routing and mutation isolation across real LgDog factories."""

import unittest
from unittest.mock import patch

from src.tasks.velocity.config.lgdog1.env_cfgs import lgdog1_flat_env_cfg, lgdog1_rough_env_cfg
from src.tasks.velocity.config.lgdog1.blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg
from src.tasks.velocity.config.lgdog1.him_env_cfg import lgdog1_him_env_cfg
from src.tasks.velocity.config.lgdog1.reward_cfg import (
  DASHBOARD_REWARD_OVERRIDES, REWARD_PROFILES, make_lgdog_rewards,
)


class RewardProfileTests(unittest.TestCase):
  def test_him_override_reaches_train_and_play_without_changing_other_tasks(self):
    with patch.dict(REWARD_PROFILES["him"]["weights"], {"foot_clearance": -0.8}):
      for play in (False, True):
        self.assertEqual(lgdog1_him_env_cfg(play).rewards["foot_clearance"].weight, -0.8)
        self.assertEqual(lgdog1_blind_terrain_env_cfg(play).rewards["foot_clearance"].weight, -1.0)
        self.assertEqual(lgdog1_flat_env_cfg(play).rewards["foot_clearance"].weight, -1.0)
        self.assertEqual(lgdog1_rough_env_cfg(play).rewards["foot_clearance"].weight, -1.0)

  def test_parameter_override_reaches_relative_clearance(self):
    with patch.dict(REWARD_PROFILES["him"]["params"]["foot_clearance"], {"target_height": 0.12}):
      cfg = lgdog1_him_env_cfg()
      self.assertEqual(cfg.rewards["foot_clearance"].params["target_height"], 0.12)
      self.assertEqual(len(cfg.rewards["foot_clearance"].params["sensor_names"]), 4)
      self.assertEqual(lgdog1_blind_terrain_env_cfg().rewards["foot_clearance"].params["target_height"], 0.1)

  def test_mutating_a_config_does_not_leak_to_later_configs_or_profiles(self):
    first = lgdog1_him_env_cfg()
    first.rewards["pose"].params["std_standing"].clear()
    first.rewards["foot_gait"].params["offset"][0] = 0.25
    first.rewards["foot_clearance"].params["target_height"] = 9.0
    second = lgdog1_him_env_cfg()
    self.assertEqual(len(second.rewards["pose"].params["std_standing"]), 3)
    self.assertEqual(second.rewards["foot_gait"].params["offset"][0], 0.0)
    self.assertEqual(second.rewards["foot_clearance"].params["target_height"], 0.1)

  def test_invalid_profile_reward_name_and_missing_rays_fail_clearly(self):
    kwargs = {"base_body": "trunk", "foot_sites": ("FR_foot", "FL_foot", "RR_foot", "RL_foot")}
    with self.assertRaisesRegex(ValueError, "Unknown LgDog reward profile"):
      make_lgdog_rewards("typo", **kwargs)
    with self.assertRaisesRegex(ValueError, "one terrain ray"):
      make_lgdog_rewards("him", **kwargs)
    with patch.dict(REWARD_PROFILES["him"]["weights"], {"typo_reward": -1.0}):
      with self.assertRaisesRegex(ValueError, "Unknown reward names"):
        make_lgdog_rewards("him", **kwargs)

  def test_positive_clearance_from_old_dashboard_override_fails_before_training(self):
    with patch.dict(DASHBOARD_REWARD_OVERRIDES["him_standard"]["weights"], {"foot_clearance": 1.0}):
      from src.tasks.velocity.config.lgdog1.him_env_cfg import lgdog1_him_standard_env_cfg
      with self.assertRaisesRegex(ValueError, "foot_clearance.*非负代价"):
        lgdog1_him_standard_env_cfg()


if __name__ == "__main__":
  unittest.main()
