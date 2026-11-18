"""CPU tests for the four opt-in passage-curriculum improvements.

The fake environment mirrors ``tests/test_speed_groups.py``: four environments
live on four terrain columns split into a ``flat`` and a ``high_wall`` speed
group.  Every new feature must be off by default, so the legacy assertions stay
in the file as regression guards.
"""

import hashlib
from dataclasses import asdict
import unittest
from types import SimpleNamespace as NS

import torch

from mjlab.terrains import TerrainGeneratorCfg
from src.tasks.velocity.mdp import auto_curriculum, passage
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg


GROUPS = {'flat': ('flat',), 'wall': ('high_wall',)}


class FakeCommand:
  """Minimal PassageCommand stand-in with per-environment speed/command bands."""

  def __init__(self, cfg, terrain_types):
    self.cfg = cfg
    self.terrain_types = terrain_types
    self.xy = torch.zeros(len(terrain_types), 2)
    lower, upper = (float(value) for value in cfg.ranges.lin_vel_x)
    self.speed = torch.zeros(len(terrain_types))
    self.speed_floors = torch.full((len(terrain_types),), lower)
    self.speed_limits = torch.full((len(terrain_types),), upper)
    self.yaw_limits = torch.full((len(terrain_types),), .8)
    self.lateral_limits = None

  def local_xy(self):
    return self.xy

  def column_mask(self, columns):
    columns = torch.as_tensor(columns, dtype=torch.long).flatten()
    if len(columns) == 0:
      return torch.zeros(len(self.terrain_types), dtype=torch.bool)
    return torch.isin(self.terrain_types, columns)

  def set_speed_limits(self, columns, low, high):
    mask = self.column_mask(columns)
    self.speed_floors[mask] = float(low)
    self.speed_limits[mask] = float(high)

  def set_global_speeds(self, low, high):
    self.speed_floors.fill_(float(low))
    self.speed_limits.fill_(float(high))

  def set_command_limits(self, columns, yaw_max=None, vy_max=None):
    mask = self.column_mask(columns)
    if yaw_max is not None:
      self.yaw_limits[mask] = float(yaw_max)
    if vy_max is not None:
      if self.lateral_limits is None:
        self.lateral_limits = torch.full((len(self.terrain_types),), torch.inf)
      self.lateral_limits[mask] = float(vy_max)


def fake_env(speed_groups=None, ranges=(.25, .35), floor=.35, ceiling=2.,
             command_cfg=None, failure=None, yaw_max=None, yaw_max_limit=None,
             vy_max=None, vy_max_limit=None):
  """Four environments on four columns: 0-1 flat, 2-3 high_wall."""
  terrain_types = torch.tensor([0, 1, 2, 3])
  num_envs = len(terrain_types)
  band = (yaw_max, yaw_max_limit, vy_max, vy_max_limit)
  if command_cfg is None:
    command_cfg = NS(
      goal_x=5.3, lane_half_width=.28, waypoint_radius=.15,
      ranges=NS(lin_vel_x=tuple(ranges), lin_vel_y=(-.1, .1), ang_vel_z=(-.8, .8)),
      speed_groups=speed_groups, speed_group_floor=floor,
      speed_group_ceiling=ceiling, yaw_max=yaw_max, yaw_max_limit=yaw_max_limit,
      vy_max=vy_max, vy_max_limit=vy_max_limit)
  elif any(value is not None for value in band):
    raise ValueError('pass band fields only with the plain (NS) command cfg')
  command = FakeCommand(command_cfg, terrain_types)
  terrain = NS(terrain_types=terrain_types,
               terrain_levels=torch.zeros(num_envs, dtype=torch.long),
               max_terrain_level=3, terrain_origins=torch.zeros(3, 4, 3),
               env_origins=torch.zeros(num_envs, 3),
               cfg=NS(terrain_generator=TerrainGeneratorCfg(
                 size=(8., 8.), num_cols=4, num_rows=3,
                 sub_terrains={'flat': PassageTerrainCfg(kind='start_flat', proportion=.5),
                               'high_wall': PassageTerrainCfg(kind='high_wall', proportion=.5)})))

  def update_env_origins(ids, up, down):
    terrain.terrain_levels[ids] = (terrain.terrain_levels[ids]
                                   + up.long() - down.long()).clamp(0, 2)

  terrain.update_env_origins = update_env_origins
  if failure is None:
    failure = torch.zeros(num_envs, dtype=torch.bool)
  env = NS(num_envs=num_envs, device='cpu', step_dt=.02, scene=NS(terrain=terrain),
           command_manager=NS(get_term=lambda name: command),
           episode_length_buf=torch.full((num_envs,), 100),
           reset_buf=torch.ones(num_envs, dtype=torch.bool),
           cfg=NS(terminations={'impact': NS(time_out=False)}),
           termination_manager=NS(get_term=lambda name: failure),
           metrics_manager=NS(_step_count=torch.full((num_envs,), 100),
                              _episode_sums={'velocity_error_xy_mps': torch.zeros(num_envs)}))
  return env, command, failure


def curriculum_of(env, params=None):
  return passage.PassageCurriculum(NS(params=dict(params or {})), env)


class DefaultOffTests(unittest.TestCase):
  def test_every_feature_is_off_by_default_and_keeps_legacy_behavior(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env)
    self.assertFalse(curriculum.command_curriculum)
    self.assertFalse(curriculum.trajectory_curriculum)
    self.assertIsNone(curriculum.trajectory_scheduler)
    self.assertEqual(curriculum.resample_windows, 0)
    self.assertIsNone(curriculum.safety_margin_fn)
    state = curriculum.state_dict()
    for key in ('command_bands', 'trajectory_scheduler', 'resample', 'safety'):
      self.assertNotIn(key, state)
    command.xy[:, 0] = 5.4
    for _ in range(3):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    self.assertTrue(torch.all(command.yaw_limits == .8))
    self.assertIsNone(command.lateral_limits)

  def test_default_command_cfg_keeps_the_legacy_fingerprint(self):
    # A checkpoint written before the yaw/lateral fields existed must still
    # resume: None fields are removed from the command contract before hashing.
    cfg = passage.PassageCommandCfg(
      entity_name='robot', resampling_time_range=(76., 76.), heading_command=True,
      ranges=passage.PassageCommandCfg.Ranges(
        lin_vel_x=(.25, .35), lin_vel_y=(-.1, .1), ang_vel_z=(-.8, .8),
        heading=(-3.141593, 3.141593)),
      waypoints={})
    env, _, _ = fake_env(GROUPS, command_cfg=cfg)
    curriculum = curriculum_of(env)
    legacy = asdict(cfg)
    for key in ('speed_groups', 'speed_group_floor', 'speed_group_ceiling',
                'yaw_max', 'yaw_max_limit', 'vy_max', 'vy_max_limit',
                'skill_training_profile', 'slalom_use_lateral',
                'obstacle_speed_limits', 'staged_waypoints'):
      legacy.pop(key, None)
    generator = env.scene.terrain.cfg.terrain_generator
    expected = hashlib.sha256(
      repr((asdict(generator), env.step_dt, None, legacy)).encode()).hexdigest()
    self.assertEqual(curriculum.fingerprint, expected)


class CommandBandwidthTests(unittest.TestCase):
  def make(self):
    env, command, _ = fake_env(GROUPS, yaw_max=.4, yaw_max_limit=.8,
                               vy_max=.1, vy_max_limit=.3)
    return env, command, curriculum_of(env)

  def test_bandwidth_grows_proportionally_with_each_group(self):
    env, command, curriculum = self.make()
    self.assertTrue(curriculum.command_curriculum)
    self.assertTrue(torch.allclose(command.yaw_limits, torch.full((4,), .4)))
    self.assertTrue(torch.allclose(command.lateral_limits, torch.full((4,), .1)))
    command.xy[:, 0] = 5.4
    command.xy[2:, 0] = 0.  # high_wall never crosses
    logs = None
    for _ in range(3):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    self.assertGreater(float(command.yaw_limits[0]), .4)
    self.assertGreater(float(command.lateral_limits[0]), .1)
    # The stuck group owns its own bandwidth and has not expanded.
    self.assertAlmostEqual(float(command.yaw_limits[2]), .4)
    self.assertAlmostEqual(float(command.lateral_limits[2]), .1)
    self.assertGreater(float(logs['yaw_max/flat']), .4)
    self.assertAlmostEqual(float(logs['yaw_max/wall']), .4)
    self.assertGreater(float(logs['vy_max/flat']), .1)
    # Repeated success opens the band all the way to the configured limit.
    for _ in range(200):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], 2.)
    self.assertAlmostEqual(float(command.yaw_limits[0]), .8)
    self.assertAlmostEqual(float(command.yaw_limits[1]), .8)
    self.assertAlmostEqual(float(command.lateral_limits[0]), .3)
    self.assertAlmostEqual(float(command.yaw_limits[2]), .4)
    self.assertAlmostEqual(float(command.lateral_limits[2]), .1)

  def test_band_state_round_trips_and_old_checkpoints_still_load(self):
    env, command, curriculum = self.make()
    command.xy[:, 0] = 5.4
    command.xy[2:, 0] = 0.
    for _ in range(6):
      curriculum(env, torch.arange(4), min_episodes=1)
    expected_high = curriculum.groups['flat']['high']
    expected_yaw = float(command.yaw_limits[0])
    state = curriculum.state_dict()
    self.assertIn('command_bands', state)

    other_env, other_command, _ = fake_env(GROUPS, yaw_max=.4, yaw_max_limit=.8,
                                           vy_max=.1, vy_max_limit=.3)
    restored = curriculum_of(other_env)
    restored.load_state_dict(state)
    self.assertAlmostEqual(restored.groups['flat']['high'], expected_high)
    self.assertAlmostEqual(float(other_command.yaw_limits[0]), expected_yaw)

    # A checkpoint written with the feature disabled has no optional keys; the
    # bands are re-derived from the restored speed progress instead.
    legacy = {key: value for key, value in state.items()
              if key not in ('command_bands', 'trajectory_scheduler',
                             'resample', 'safety')}
    restored.load_state_dict(legacy)
    self.assertAlmostEqual(float(other_command.yaw_limits[0]), expected_yaw)

  def test_passage_command_delivers_per_environment_bandwidth(self):
    # Exercise the real PassageCommand setter without building a physics scene.
    command = object.__new__(passage.PassageCommand)
    command.column_kinds = torch.tensor([0, 0, 1, 1])
    command._env = NS(num_envs=4, device='cpu',
                      scene=NS(terrain=NS(terrain_types=torch.tensor([0, 1, 2, 3]))))
    command.vel_command_b = torch.tensor([[.5, .5, .7]]*4)
    command.yaw_limits = torch.full((4,), .8)
    command.lateral_limits = None
    command.set_command_limits(torch.tensor([0, 1]), yaw_max=.4, vy_max=.1)
    self.assertTrue(torch.allclose(command.yaw_limits, torch.tensor([.4, .4, .8, .8])))
    self.assertTrue(torch.allclose(command.lateral_limits, torch.tensor([.1, .1, torch.inf, torch.inf])))
    self.assertTrue(torch.allclose(command.vel_command_b[:, 2], torch.tensor([.4, .4, .7, .7])))
    self.assertTrue(torch.allclose(command.vel_command_b[:, 1], torch.tensor([.1, .1, .5, .5])))
    command.set_global_command_limits(yaw_max=.2)
    self.assertTrue(torch.allclose(command.yaw_limits, torch.full((4,), .2)))
    self.assertTrue(torch.all(command.vel_command_b[:, 2] == .2))

  def test_invalid_band_configuration_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'yaw_max'):
      curriculum_of(fake_env(GROUPS, yaw_max=.8, yaw_max_limit=.4)[0])
    with self.assertRaisesRegex(ValueError, 'yaw_max_limit requires'):
      curriculum_of(fake_env(GROUPS, yaw_max_limit=.8)[0])


class TrajectoryQualityTests(unittest.TestCase):
  def test_quality_score_rewards_progress_tracking_and_safe_termination(self):
    good = auto_curriculum.trajectory_quality_score(1., 0., failed=0.)
    bad = auto_curriculum.trajectory_quality_score(0., .5, failed=1.)
    middle = auto_curriculum.trajectory_quality_score(.5, .25, failed=0.)
    self.assertAlmostEqual(float(good), 1.)
    self.assertAlmostEqual(float(bad), 0.)
    self.assertTrue(float(bad) < float(middle) < float(good))
    # A timeout without success is mildly penalised by its low progress only.
    timeout = auto_curriculum.trajectory_quality_score(.2, .1, failed=0.)
    self.assertTrue(float(bad) < float(timeout) < float(good))
    with self.assertRaises(ValueError):
      auto_curriculum.trajectory_quality_score(0., 0., progress_weight=-1.)
    with self.assertRaises(ValueError):
      auto_curriculum.trajectory_quality_score(0., 0., progress_weight=0.,
                                               error_weight=0., failure_weight=0.)
    with self.assertRaises(ValueError):
      auto_curriculum.trajectory_quality_score(0., 0., error_ref=0.)
    with self.assertRaises(ValueError):
      auto_curriculum.trajectory_quality_score(float('nan'), 0.)

  def test_bandwidth_interpolation_is_bounded(self):
    self.assertAlmostEqual(auto_curriculum.interpolate_bandwidth(.4, .8, 0.), .4)
    self.assertAlmostEqual(auto_curriculum.interpolate_bandwidth(.4, .8, 1.), .8)
    self.assertAlmostEqual(auto_curriculum.interpolate_bandwidth(.4, .8, .5), .6)
    self.assertAlmostEqual(auto_curriculum.interpolate_bandwidth(.4, .8, 2.), .8)
    self.assertAlmostEqual(auto_curriculum.interpolate_bandwidth(.4, .8, -1.), .4)
    with self.assertRaises(ValueError):
      auto_curriculum.interpolate_bandwidth(-.1, .8, 0.)

  def test_survival_margin_prefers_successful_and_long_episodes(self):
    env, command, failure = fake_env(GROUPS)
    command.xy[:, 0] = 5.4
    margin = auto_curriculum.survival_safety_margin(env, torch.arange(4))
    self.assertTrue((margin > .5).all())
    failure[2:] = True
    margin = auto_curriculum.survival_safety_margin(env, torch.arange(4))
    self.assertTrue(torch.all(margin[2:] == 0))
    self.assertTrue((margin[:2] > .5).all())


class TrajectorySchedulerTests(unittest.TestCase):
  def scheduler(self, **kwargs):
    return auto_curriculum.TrajectoryDifficultyScheduler(
      ['flat', 'wall'], up_threshold=.7, down_threshold=.3, patience=3, **kwargs)

  def test_promotes_only_after_consecutive_high_quality_windows(self):
    scheduler = self.scheduler()
    for _ in range(2):
      scheduler.observe('flat', torch.tensor([.9, .8]))
      scheduler.observe('wall', torch.tensor([.9]))
      self.assertEqual(scheduler.update(), {})
    scheduler.observe('flat', torch.tensor([.85]))
    scheduler.observe('wall', torch.tensor([.8]))
    self.assertEqual(scheduler.update(), {'flat': 1, 'wall': 1})
    self.assertEqual(scheduler.up_streak, {'flat': 0, 'wall': 0})
    self.assertAlmostEqual(scheduler.last_window_quality['flat'], .85)

    # A medium window breaks the streak: two highs + medium + two highs is not
    # three consecutive high-quality windows.
    for value in (.9, .9, .5, .9, .9):
      scheduler.observe('flat', torch.tensor([value]))
      scheduler.update()
    self.assertAlmostEqual(scheduler.last_window_quality['flat'], .9)
    self.assertEqual(scheduler.up_streak['flat'], 2)
    scheduler.observe('flat', torch.tensor([.9]))
    self.assertEqual(scheduler.update(), {'flat': 1})

  def test_demotes_after_consecutive_low_quality_windows(self):
    scheduler = self.scheduler()
    for window in range(3):
      scheduler.observe('flat', torch.tensor([.1, .2]))
      decisions = scheduler.update()
      if window < 2:
        self.assertEqual(decisions, {})
    self.assertEqual(decisions, {'flat': -1})
    self.assertEqual(scheduler.down_streak['flat'], 0)

  def test_min_samples_keeps_a_window_open(self):
    scheduler = self.scheduler(min_samples=2)
    scheduler.observe('flat', torch.tensor([.9]))
    self.assertEqual(scheduler.update(), {})
    self.assertEqual(scheduler.pending_samples('flat'), 1)
    scheduler.observe('flat', torch.tensor([.9]))
    scheduler.update()
    self.assertEqual(scheduler.pending_samples('flat'), 0)
    self.assertEqual(scheduler.up_streak['flat'], 1)

  def test_state_round_trip_and_validation(self):
    scheduler = self.scheduler()
    for _ in range(2):
      scheduler.observe('flat', torch.tensor([.9]))
      scheduler.observe('wall', torch.tensor([.1]))
      scheduler.update()
    state = scheduler.state_dict()
    restored = self.scheduler()
    restored.load_state_dict(state)
    self.assertEqual(restored.up_streak, scheduler.up_streak)
    self.assertEqual(restored.down_streak, scheduler.down_streak)
    for _ in range(2):
      for name, values in (('flat', [.9]), ('wall', [.1])):
        scheduler.observe(name, torch.tensor(values))
        restored.observe(name, torch.tensor(values))
      self.assertEqual(scheduler.update(), restored.update())

    for kwargs in ({'up_threshold': .3, 'down_threshold': .3},
                   {'patience': 0}, {'down_threshold': .9},
                   {'up_threshold': 0.}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        auto_curriculum.TrajectoryDifficultyScheduler(['flat'], **kwargs)
    with self.assertRaises(ValueError):
      auto_curriculum.TrajectoryDifficultyScheduler([])
    with self.assertRaises(KeyError):
      scheduler.observe('nope', torch.tensor([.5]))
    with self.assertRaisesRegex(ValueError, 'Incompatible'):
      restored.load_state_dict({**state, 'groups': ['flat']})
    with self.assertRaisesRegex(ValueError, 'Incompatible'):
      restored.load_state_dict({**state, 'up_threshold': .6})
    with self.assertRaisesRegex(ValueError, 'Invalid'):
      restored.load_state_dict({**state, 'up_streak': {'flat': 9, 'wall': 0}})


class TrajectoryCurriculumIntegrationTests(unittest.TestCase):
  def params(self):
    return {'trajectory_curriculum': True, 'trajectory_patience': 2,
            'trajectory_up_threshold': .6, 'trajectory_down_threshold': .4,
            'trajectory_min_samples': 1}

  def test_high_quality_windows_promote_and_low_quality_windows_demote(self):
    env, command, failure = fake_env(GROUPS)
    curriculum = curriculum_of(env, self.params())
    self.assertIsNotNone(curriculum.trajectory_scheduler)
    command.xy[:, :] = 0.
    command.xy[:, 0] = 5.4  # every group reaches the goal cleanly
    logs = None
    for _ in range(2):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [1, 1, 1, 1])
    self.assertAlmostEqual(float(logs['quality/flat']), 1.)
    self.assertAlmostEqual(float(logs['quality/wall']), 1.)
    self.assertAlmostEqual(float(logs['trajectory_up/flat']), 1.)

    # Low quality: no progress, bad tracking and an unsafe termination.
    command.xy[:, 0] = 0.
    env.metrics_manager._episode_sums['velocity_error_xy_mps'][:] = 50.
    failure[:] = True
    for _ in range(2):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [0, 0, 0, 0])
    self.assertAlmostEqual(float(logs['trajectory_down/flat']), 1.)
    self.assertAlmostEqual(float(logs['quality/flat']), 0.)

  def test_trajectory_state_is_optional_in_old_checkpoints(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, self.params())
    command.xy[:, 0] = 5.4
    for _ in range(2):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [1, 1, 1, 1])
    state = curriculum.state_dict()
    self.assertIn('trajectory_scheduler', state)
    legacy = {key: value for key, value in state.items()
              if key != 'trajectory_scheduler'}

    other_env, _, _ = fake_env(GROUPS)
    restored = curriculum_of(other_env, self.params())
    restored.load_state_dict(legacy)  # must not require the new key
    self.assertEqual(restored.trajectory_scheduler.updates, 0)
    self.assertEqual(restored.terrain.terrain_levels.tolist(), [1, 1, 1, 1])


class ResampleTests(unittest.TestCase):
  def test_zero_success_columns_are_demoted_after_m_windows(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {'resample_zero_success_windows': 2})
    self.assertEqual(curriculum.resample_windows, 2)
    env.scene.terrain.terrain_levels[:] = 2
    command.xy[:2, 0] = 5.4
    logs = None
    for _ in range(2):
      logs = curriculum(env, torch.tensor([0, 1]), min_episodes=1)
    # The wall columns never received an environment: zero samples also count.
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 1, 1])
    self.assertEqual(curriculum.resampled_counts.tolist(), [0, 0, 1, 1])
    self.assertEqual(curriculum.zero_success_windows.tolist(), [0, 0, 0, 0])
    self.assertAlmostEqual(float(logs['resampled/high_wall']), 1.)
    self.assertAlmostEqual(float(logs['resampled/flat']), 0.)
    self.assertIn('resample', curriculum.state_dict())
    # The wall environments were mid-episode, so their new origin is deferred
    # until their next reset instead of teleporting them under the robot.
    self.assertTrue(bool(curriculum._resample_origin_pending[2:].all()))
    command.xy[2:, 0] = 5.4
    curriculum(env, torch.arange(4), min_episodes=1)
    self.assertFalse(bool(curriculum._resample_origin_pending.any()))

  def test_zero_success_after_real_attempts_is_counted(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {'resample_zero_success_windows': 1})
    env.scene.terrain.terrain_levels[:] = 2
    command.xy[:, 0] = 5.4
    command.xy[2:, 0] = 0.  # wall attempts but never succeeds
    # Keep the legacy outcome gate out of this call so only resampling acts.
    logs = curriculum(env, torch.arange(4), min_episodes=1, level_down_streak=2)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 1, 1])
    self.assertEqual(curriculum.resampled_counts.tolist(), [0, 0, 1, 1])
    self.assertAlmostEqual(float(logs['resampled/high_wall']), 1.)

  def test_resample_method_accepts_explicit_window_statistics(self):
    env, _, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {'resample_zero_success_windows': 1})
    env.scene.terrain.terrain_levels[:] = 2
    result = curriculum.resample_zero_success_columns(
      window_samples=torch.tensor([1., 1., 0., 0.]),
      window_success=torch.tensor([1., 1., 0., 0.]))
    self.assertEqual(result, {'high_wall': True})
    self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [2, 2, 1, 1])
    with self.assertRaises(ValueError):
      curriculum.resample_zero_success_columns(torch.zeros(2), torch.zeros(2))

  def test_resampling_is_disabled_by_default(self):
    env, _, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env)
    self.assertEqual(curriculum.resample_zero_success_columns(), {})
    self.assertNotIn('resample', curriculum.state_dict())


class SafetyMarginTests(unittest.TestCase):
  def test_low_margin_vetoes_speed_and_terrain_promotions(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {
      'safety_margin_fn': lambda env, ids: torch.full((len(ids),), .2),
      'safety_margin_threshold': .5})
    self.assertIsNotNone(curriculum.safety_margin_fn)
    command.xy[:, 0] = 5.4  # everyone succeeds perfectly
    for _ in range(6):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .35)
    self.assertAlmostEqual(curriculum.groups['wall']['high'], .35)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [0, 0, 0, 0])

  def test_safe_margin_allows_promotions(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {
      'safety_margin_fn': lambda env, ids: torch.full((len(ids),), .9),
      'safety_margin_threshold': .5})
    command.xy[:, 0] = 5.4
    for _ in range(3):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    # Level up is one step per successful window, capped below the top row.
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 2, 2])

  def test_safety_state_is_optional_and_invalid_margins_are_rejected(self):
    env, command, _ = fake_env(GROUPS)
    curriculum = curriculum_of(env, {
      'safety_margin_fn': lambda env, ids: torch.full((len(ids),), .9),
      'safety_margin_threshold': .5})
    command.xy[:, 0] = 5.4
    curriculum(env, torch.arange(4), min_episodes=1)
    state = curriculum.state_dict()
    self.assertIn('safety', state)
    legacy = {key: value for key, value in state.items() if key != 'safety'}

    other_env, _, _ = fake_env(GROUPS)
    restored = curriculum_of(other_env, {
      'safety_margin_fn': lambda env, ids: torch.full((len(ids),), .9),
      'safety_margin_threshold': .5})
    restored.load_state_dict(legacy)
    self.assertEqual(restored.safety_count.get('flat', 0), 0)

    bad_env, _, _ = fake_env(GROUPS)
    bad = curriculum_of(bad_env, {
      'safety_margin_fn': lambda env, ids: torch.full((len(ids),), float('nan'))})
    with self.assertRaises(ValueError):
      bad(bad_env, torch.arange(4), min_episodes=1)


if __name__ == '__main__':
  unittest.main()
