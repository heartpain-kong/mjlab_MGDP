"""Standard terrain, blind observation routing and legacy-task isolation."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from src.tasks.velocity.config.lgdog1 import blind_mdp
from src.tasks.velocity.mdp.curriculums import RewardBasedVelocityCurriculum
from src.tasks.velocity.config.lgdog1.him_env_cfg import (
  lgdog1_him_env_cfg,
  lgdog1_him_flat_env_cfg,
  lgdog1_him_standard_env_cfg,
)
from src.tasks.velocity.config.lgdog1.reward_cfg import REWARD_PROFILES
from src.tasks.velocity.rl import PPOHIMRunner


TASK = "LgDog1-Rough-Blind-HIM-Standard"


class StandardHIMTests(unittest.TestCase):
  def test_browser_task_catalog_matches_the_real_training_registry(self):
    from src.dashboard.catalog import TASKS
    for task in TASKS.values():
      self.assertEqual(load_rl_cfg(task["id"]).experiment_name, task["experiment"])
      self.assertIn("actor", load_env_cfg(task["id"]).observations)
      if task['algorithm']=='him':
        self.assertEqual(task['obs'],141)
        self.assertEqual(load_rl_cfg(task['id']).actor.history_length,3)
        for play in (False,True):
          self.assertEqual(load_env_cfg(task['id'],play=play).observations['actor'].history_length,3)

  def test_no_body_grid_scan_in_train_or_play_and_history_remains_proprioceptive(self):
    for play in (False, True):
      cfg = load_env_cfg(TASK, play=play)
      sensors = {sensor.name for sensor in cfg.scene.sensors}
      self.assertNotIn("terrain_scan", sensors)
      for group in cfg.observations.values():
        self.assertNotIn("height_scan", group.terms)
      self.assertEqual(cfg.observations["actor"].history_length, 3)
      self.assertFalse(cfg.observations["actor"].flatten_history_dim)
      self.assertFalse(cfg.observations["him_target"].enable_corruption)
      self.assertEqual(set(cfg.observations["actor"].terms), {
        "base_ang_vel", "projected_gravity", "command", "phase",
        "joint_pos", "joint_vel", "actions",
      })
      self.assertEqual(sum(name.endswith("_ground_ray") for name in sensors), 4)

  def test_standard_generator_uses_unitree_rough_profile_without_polluting_legacy_gentle(self):
    original = deepcopy(ROUGH_TERRAINS_CFG.sub_terrains)
    legacy = lgdog1_him_env_cfg()
    cfg = lgdog1_him_standard_env_cfg()
    terrain = cfg.scene.terrain.terrain_generator
    self.assertNotEqual(terrain.sub_terrains, original)
    self.assertIsNot(terrain.sub_terrains, ROUGH_TERRAINS_CFG.sub_terrains)
    self.assertEqual(terrain.sub_terrains["flat"].proportion, 0.2)
    self.assertEqual(terrain.sub_terrains["pyramid_stairs"].step_height_range, (0.05, 0.15))
    self.assertEqual(terrain.sub_terrains["hf_pyramid_slope"].slope_range, (0.0, 0.4))
    self.assertEqual(legacy.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range, (0.0, 0.05))
    self.assertEqual(ROUGH_TERRAINS_CFG.sub_terrains, original)

  def test_standard_reward_tuning_does_not_change_v7(self):
    with patch.dict(REWARD_PROFILES["him_standard"]["weights"], {"foot_clearance": -0.8}):
      for play in (False, True):
        self.assertEqual(lgdog1_him_standard_env_cfg(play).rewards["foot_clearance"].weight, -0.8)
        old = lgdog1_him_env_cfg(play)
        self.assertEqual(old.rewards["foot_clearance"].weight, -1.0)
        self.assertIn("height_scan", old.observations["critic"].terms)
        self.assertIn("terrain_scan", {sensor.name for sensor in old.scene.sensors})

  def test_registered_runner_and_checkpoint_directory_are_independent(self):
    cfg = load_rl_cfg(TASK)
    self.assertIs(load_runner_cls(TASK), PPOHIMRunner)
    self.assertEqual(cfg.experiment_name, "lgdog1_velocity_blind_him_standard_v1")
    self.assertNotEqual(cfg.experiment_name, load_rl_cfg("LgDog1-Rough-Blind-HIM").experiment_name)
    self.assertTrue(cfg.actor.class_name.endswith(":HIMActor"))
    self.assertTrue(cfg.algorithm.class_name.endswith(":HIMPPO"))

  def test_stability_training_defaults_are_recorded_in_agent_config(self):
    cfg = load_rl_cfg(TASK)
    self.assertEqual(cfg.clip_actions, 1.0)
    self.assertEqual(cfg.num_steps_per_env, 50)
    self.assertEqual(cfg.algorithm.estimator_epochs, 5)
    env = lgdog1_him_standard_env_cfg()
    self.assertEqual(env.commands["twist"].resampling_time_range, (3.0, 5.0))

  def test_improved_curriculum_and_relative_clearance_are_used_on_standard_terrain(self):
    cfg = lgdog1_him_standard_env_cfg()
    self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 1)
    self.assertIn("command_vel", cfg.curriculum)
    self.assertIs(cfg.curriculum["command_vel"].func, RewardBasedVelocityCurriculum)
    self.assertIs(cfg.curriculum["terrain_levels"].func, blind_mdp.terrain_levels_tracking)
    self.assertEqual(cfg.curriculum["terrain_levels"].params, {"max_velocity_error": 0.8})
    self.assertEqual(cfg.curriculum["command_vel"].params["linear_threshold"], 0.78)
    self.assertEqual(cfg.curriculum["command_vel"].params["linear_patience"], 3)
    self.assertEqual(
      cfg.curriculum["command_vel"].params["score_metric_name"],
      "velocity_tracking_score",
    )
    self.assertEqual(cfg.commands["twist"].rel_standing_envs, 0.25)
    self.assertEqual(cfg.rewards["track_linear_velocity"].weight, 2.0)
    self.assertEqual(cfg.rewards["track_linear_velocity"].params["relative_weight"], 0.0)
    self.assertEqual(cfg.rewards["feet_stumble"].weight, -0.5)
    self.assertEqual(cfg.rewards["feet_air_time"].weight, 1.0)
    self.assertEqual(cfg.rewards["foot_clearance"].weight, 0.0)
    self.assertEqual(cfg.rewards["pose"].weight, 1.5)
    self.assertEqual(cfg.rewards["soft_landing"].weight, -0.002)
    self.assertEqual(cfg.curriculum["command_vel"].params["min_episode_fraction"], 0.8)
    self.assertEqual(cfg.curriculum["command_vel"].params["max_failure_fraction"], 0.25)
    self.assertEqual(cfg.curriculum["command_vel"].params["progress_metric_name"], "tracking_progress")
    self.assertEqual(cfg.curriculum["command_vel"].params["distance_metric_name"], "commanded_distance")
    self.assertEqual(cfg.curriculum["command_vel"].params["min_progress_ratio"], 0.6)
    self.assertEqual(cfg.curriculum["command_vel"].params["regress_min_progress_ratio"], 0.4)
    self.assertIs(cfg.rewards["foot_clearance"].func, blind_mdp.feet_clearance_swing)
    self.assertNotIn('base_height',cfg.rewards)
    self.assertEqual(cfg.rewards["feet_air_time"].params['threshold'],0.4)
    clearance=cfg.rewards['foot_clearance'].params
    gait=cfg.rewards['foot_gait'].params
    self.assertEqual(clearance['period'],gait['period'])
    self.assertEqual(clearance['offset'],gait['offset'])
    self.assertEqual(clearance['stance_fraction'],gait['threshold'])
    self.assertEqual(cfg.rewards["leg_contact_force"].params["sensor_name"], "legs_ground_contact")
    self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-0.2, 0.4))

  def test_flat_him_keeps_speed_curriculum_but_removes_terrain_curriculum(self):
    cfg = lgdog1_him_flat_env_cfg()
    self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
    self.assertNotIn("terrain_levels", cfg.curriculum)
    self.assertIs(cfg.curriculum["command_vel"].func, RewardBasedVelocityCurriculum)
    self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (-0.2, 0.4))
    self.assertEqual(cfg.curriculum["command_vel"].params["min_progress_ratio"], 0.6)


if __name__ == "__main__":
  unittest.main()
