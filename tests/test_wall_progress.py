"""World-frame wall-waypoint progress reward and relative-progress curriculum.

The reward tests check the CMU-Extreme-Parkour-style tracking term
``r = clamp(<v_world, d_next>, 0, v_cmd)``: only forward velocity towards the
next uncleared wall midpoint may pay. The curriculum tests check the opt-in
per-column ``route_progress_m / route_length_m`` terrain gate and that the
legacy success-rate curriculum is untouched when the mode stays off.
"""

import unittest
from types import SimpleNamespace as NS

import torch

from mjlab.terrains import TerrainGeneratorCfg
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.mdp.wall_training import WallApproachReward
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import (
  START_X, WALL_CENTRES, PassageTerrainCfg, WallSkillTerrainCfg,
)


class Scene(dict):
  @property
  def terrain(self):
    return self["terrain"]


def wall_env(num_envs=4, levels=7, types=None):
  """Four environments on wall columns, wall centres relative to the spawn."""
  generator = TerrainGeneratorCfg(
    size=(12., 8.), num_rows=8, num_cols=4,
    sub_terrains={
      'flat': PassageTerrainCfg(kind='start_flat', proportion=.5),
      'high_wall': WallSkillTerrainCfg(size=(12., 8.), proportion=.5)})
  if types is None:
    types = torch.tensor([2, 3, 2, 3])  # columns 2-3 own the high_wall lesson
  types = torch.as_tensor(types, dtype=torch.long)
  terrain = NS(cfg=NS(terrain_generator=generator), terrain_types=types,
               terrain_levels=torch.full((num_envs,), levels, dtype=torch.long),
               env_origins=torch.zeros(num_envs, 3))
  xy = torch.zeros(num_envs, 2)
  velocity = torch.zeros(num_envs, 3)
  command = NS(command=torch.zeros(num_envs, 3), local_xy=lambda: xy,
               wall_centres=torch.tensor(WALL_CENTRES)-START_X,
               cfg=NS(goal_x=5.3, lane_half_width=.28))
  robot = NS(data=NS(root_link_lin_vel_w=velocity))
  scene = Scene(terrain=terrain, robot=robot)
  env = NS(num_envs=num_envs, device='cpu', step_dt=.02, scene=scene,
           command_manager=NS(get_term=lambda name: command))
  return env, xy, velocity, command


def wall_reward(env):
  return WallApproachReward(NS(params={}), env)


def approach_env(x, y, velocity_xy, levels=7, types=None):
  env, xy, velocity, command = wall_env(levels=levels, types=types)
  xy[:, 0], xy[:, 1] = x, y
  velocity[:, :2] = torch.as_tensor(velocity_xy, dtype=torch.float32)
  return env, xy, velocity


class WallApproachRewardTests(unittest.TestCase):
  def reward(self, x, y, velocity_xy, levels=7, types=None, **kwargs):
    env, _, _ = approach_env(x, y, velocity_xy, levels=levels, types=types)
    return wall_reward(env)(env, **kwargs)

  def test_velocity_towards_the_next_wall_pays_its_projection(self):
    reward = self.reward(.5, 0., [.4, 0.])
    self.assertTrue(torch.all(reward > 0.))
    self.assertAlmostEqual(float(reward[0]), .4, places=6)
    # The commanded tracking speed caps the paid projection.
    capped = self.reward(.5, 0., [1.5, 0.], v_cmd=.6)
    self.assertAlmostEqual(float(capped[0]), .6, places=6)
    small = self.reward(.5, 0., [1.5, 0.], v_cmd=.25)
    self.assertAlmostEqual(float(small[0]), .25, places=6)

  def test_lateral_and_backward_world_velocity_pay_nothing(self):
    lateral = self.reward(.5, 0., [0., .6])
    backward = self.reward(.5, 0., [-.6, 0.])
    zeros = torch.zeros_like(lateral)
    torch.testing.assert_close(lateral, zeros)
    torch.testing.assert_close(backward, zeros)

  def test_far_from_the_next_wall_pays_nothing(self):
    # Wall 1 is at local x = 1.3; 1.8 m behind the wall is outside 1.6 m.
    far = self.reward(-.5, 0., [.6, 0.])
    torch.testing.assert_close(far, torch.zeros_like(far))
    near = self.reward(-.2, 0., [.6, 0.], max_distance_m=1.6)
    self.assertGreater(float(near[0]), 0.)

  def test_target_switches_to_the_next_wall_after_crossing(self):
    env, xy, velocity = approach_env(1.4, -.8, [0., 0.])
    term = wall_reward(env)
    # Wall 1 (1.3) is behind; aim exactly at the second wall midpoint (2.7, 0).
    first_target = torch.tensor([2.7-1.4, .8])
    first_target /= first_target.norm()
    velocity[:, :2] = .5*first_target
    reward = term(env)
    self.assertAlmostEqual(float(reward[0]), .5, places=5)
    # The old (first-wall) target is behind the robot: aiming there must not pay
    # the same rate as aiming at the actual next wall.
    stale_target = torch.tensor([1.3-1.4, .8])
    stale_target /= stale_target.norm()
    velocity[:, :2] = .5*stale_target
    stale = float(term(env)[0])
    self.assertLess(stale, .35)
    # All three walls crossed: the term stops paying entirely.
    xy[:] = torch.tensor([4.2, 0.])
    velocity[:, :2] = torch.tensor([.5, 0.])
    torch.testing.assert_close(term(env), torch.zeros(4))

  def test_only_wall_columns_and_active_walls_are_rewarded(self):
    flat = self.reward(.5, 0., [.6, 0.], types=torch.tensor([0, 1, 0, 1]))
    torch.testing.assert_close(flat, torch.zeros_like(flat))
    # Level 0 has a single wall: once it is behind the robot the term is zero,
    # even though the full course still contains walls two and three.
    single = self.reward(2.0, 0., [.6, 0.], levels=0)
    torch.testing.assert_close(single, torch.zeros_like(single))
    before = self.reward(.5, 0., [.6, 0.], levels=0)
    self.assertGreater(float(before[0]), 0.)

  def test_missing_velocity_interface_falls_back_to_position_differences(self):
    env, xy, _ = approach_env(.2, 0., [0., 0.])
    env.scene['robot'].data.root_link_lin_vel_w = None
    term = wall_reward(env)
    term.reset()
    xy[:, 0] = .3
    first = term(env)
    # No finite-difference baseline yet after a reset: no income and no spike.
    torch.testing.assert_close(first, torch.zeros_like(first))
    xy[:, 0] = .5
    speed_up = term(env)
    # +0.2 m in one 20 ms step would be 10 m/s; the term caps at v_cmd.
    self.assertAlmostEqual(float(speed_up[0]), .6, places=5)
    # Holding still pays nothing.
    torch.testing.assert_close(term(env), torch.zeros_like(speed_up))

  def test_invalid_parameters_are_rejected(self):
    env, _, _ = approach_env(.5, 0., [.5, 0.])
    term = wall_reward(env)
    with self.assertRaisesRegex(ValueError, 'v_cmd'):
      term(env, v_cmd=0.)
    with self.assertRaisesRegex(ValueError, 'max_distance_m'):
      term(env, max_distance_m=0.)


GROUPS = {'flat': ('flat',), 'wall': ('high_wall',)}


class FakeCommand:
  """Minimal directed-passage command stand-in for the curriculum tests."""

  def __init__(self, cfg, terrain_types, num_columns=4):
    self.cfg = cfg
    self.terrain_types = terrain_types
    self.xy = torch.zeros(len(terrain_types), 2)
    # Final route progress in metres and the matching route lengths.
    self.progress = torch.zeros(len(terrain_types))
    self.cumulative = torch.full((num_columns, 2), 5.3)
    self.last_waypoint = torch.ones(num_columns, dtype=torch.long)
    lower, upper = (float(value) for value in cfg.ranges.lin_vel_x)
    self.speed = torch.zeros(len(terrain_types))
    self.speed_floors = torch.full((len(terrain_types),), lower)
    self.speed_limits = torch.full((len(terrain_types),), upper)

  def route_keys(self):
    return self.terrain_types

  def route_state(self):
    return None, None, None, self.progress

  def route_success(self):
    return self.progress >= 5.3

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


def curriculum_env(levels=None, failure=None, speed_groups=GROUPS):
  """Four environments on four columns: 0-1 flat, 2-3 high_wall."""
  terrain_types = torch.tensor([0, 1, 2, 3])
  num_envs = len(terrain_types)
  cfg = NS(goal_x=5.3, lane_half_width=.28, waypoint_radius=.15,
           ranges=NS(lin_vel_x=(.25, .35), lin_vel_y=(-.1, .1), ang_vel_z=(-.8, .8)),
           speed_groups=speed_groups, speed_group_floor=.35, speed_group_ceiling=2.)
  command = FakeCommand(cfg, terrain_types)
  level_buffer = (torch.zeros(num_envs, dtype=torch.long) if levels is None
                  else torch.as_tensor(levels, dtype=torch.long))
  terrain = NS(terrain_types=terrain_types, terrain_levels=level_buffer.clone(),
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
  return env, command


def curriculum_of(env, params=None):
  return passage.PassageCurriculum(NS(params=dict(params or {})), env)


class ProgressCurriculumTests(unittest.TestCase):
  def params(self, **overrides):
    base = {'progress_curriculum': True}
    base.update(overrides)
    return base

  def test_progress_mode_is_off_by_default_and_legacy_stays(self):
    env, command = curriculum_env()
    curriculum = curriculum_of(env)
    self.assertFalse(curriculum.progress_curriculum)
    self.assertNotIn('progress', curriculum.state_dict())
    command.xy[:, 0] = 5.4  # every column crosses cleanly
    command.progress[:] = 5.3
    logs = None
    for _ in range(3):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    # The legacy success-rate gate is unchanged and still expands both groups.
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    self.assertAlmostEqual(curriculum.groups['wall']['high'], .375)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 2, 2])
    self.assertFalse(any(key.startswith('progress_') for key in logs))
    self.assertIn('success_rate/flat', logs)
    for _ in range(2):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertNotIn('progress', curriculum.state_dict())

  def test_high_progress_columns_promote_and_low_columns_demote(self):
    env, command = curriculum_env(levels=[1, 1, 1, 1])
    curriculum = curriculum_of(env, self.params())
    self.assertTrue(curriculum.progress_curriculum)
    command.progress[:] = torch.tensor([5.3, 0., 5.3, 0.])
    logs = curriculum(env, torch.arange(4), min_episodes=1, level_down_streak=1)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 0, 2, 0])
    # Per-terrain logs average column ratios: one goal + one stall = 0.5.
    self.assertAlmostEqual(float(logs['progress_ratio/flat']), .5)
    self.assertAlmostEqual(float(logs['progress_ratio/high_wall']), .5)
    self.assertAlmostEqual(float(logs['progress_up/flat']), 1.)
    self.assertAlmostEqual(float(logs['progress_down/flat']), 1.)
    self.assertAlmostEqual(float(logs['progress_up/high_wall']), 1.)
    self.assertAlmostEqual(float(logs['progress_down/high_wall']), 1.)

  def test_neutral_progress_keeps_levels_and_consumes_the_window(self):
    env, command = curriculum_env(levels=[1, 1, 1, 1])
    curriculum = curriculum_of(env, self.params())
    command.progress[:] = 2.65  # exactly half of the 5.3 m route
    logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [1, 1, 1, 1])
    self.assertAlmostEqual(float(logs['progress_ratio/flat']), .5)
    self.assertAlmostEqual(float(logs['progress_up/flat']), 0.)
    self.assertAlmostEqual(float(logs['progress_down/flat']), 0.)
    # A finished window always starts from zero.
    self.assertEqual(curriculum.progress_count.tolist(), [0., 0., 0., 0.])
    self.assertEqual(curriculum.progress_sum.tolist(), [0., 0., 0., 0.])

  def test_low_progress_must_persist_before_demotion(self):
    env, command = curriculum_env(levels=[1, 1, 1, 1])
    curriculum = curriculum_of(env, self.params())
    command.progress[:] = 0.
    curriculum(env, torch.arange(4), min_episodes=1, level_down_streak=2)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [1, 1, 1, 1])
    logs = curriculum(env, torch.arange(4), min_episodes=1, level_down_streak=2)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [0, 0, 0, 0])
    self.assertAlmostEqual(float(logs['progress_down/flat']), 1.)
    # A single high window breaks the demotion streak.
    env2, command2 = curriculum_env(levels=[1, 1, 1, 1])
    other = curriculum_of(env2, self.params())
    command2.progress[:] = 0.
    other(env2, torch.arange(4), min_episodes=1, level_down_streak=2)
    command2.progress[:] = 5.3
    other(env2, torch.arange(4), min_episodes=1, level_down_streak=2)
    command2.progress[:] = 0.
    other(env2, torch.arange(4), min_episodes=1, level_down_streak=2)
    self.assertEqual(other.terrain.terrain_levels.tolist(), [2, 2, 2, 2])

  def test_progress_mode_keeps_the_speed_gates_working(self):
    env, command = curriculum_env()
    curriculum = curriculum_of(env, self.params())
    command.xy[:, 0] = 5.4
    command.progress[:] = 5.3
    logs = None
    for _ in range(3):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    self.assertAlmostEqual(curriculum.groups['wall']['high'], .375)
    self.assertAlmostEqual(float(logs['speed_max/flat']), .375)
    # The progress gate still moved the terrain levels as configured.
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 2, 2])

  def test_progress_window_state_round_trips_and_is_optional(self):
    env, command = curriculum_env(levels=[1, 1, 1, 1])
    curriculum = curriculum_of(env, self.params())
    command.progress[:] = 5.3
    curriculum(env, torch.arange(4), min_episodes=2)  # half a window
    self.assertEqual(curriculum.progress_count.tolist(), [1., 1., 1., 1.])
    self.assertEqual(curriculum.progress_sum.tolist(), [1., 1., 1., 1.])
    state = curriculum.state_dict()
    self.assertIn('progress', state)

    other_env, _ = curriculum_env(levels=[0, 0, 0, 0])
    restored = curriculum_of(other_env, self.params())
    restored.load_state_dict(state)
    self.assertEqual(restored.progress_count.tolist(), [1., 1., 1., 1.])
    self.assertEqual(restored.progress_sum.tolist(), [1., 1., 1., 1.])
    self.assertEqual(restored.terrain.terrain_levels.tolist(), [1, 1, 1, 1])

    # A checkpoint written without the feature's key loads as a fresh window.
    legacy = {key: value for key, value in state.items() if key != 'progress'}
    fresh_env, _ = curriculum_env(levels=[0, 0, 0, 0])
    fresh = curriculum_of(fresh_env, self.params())
    fresh.load_state_dict(legacy)
    self.assertEqual(fresh.progress_count.tolist(), [0., 0., 0., 0.])

  def test_progress_mode_coexists_with_the_trajectory_curriculum(self):
    env, command = curriculum_env(levels=[1, 1, 1, 1])
    params = self.params(trajectory_curriculum=True, trajectory_patience=1,
                         trajectory_min_samples=1, trajectory_up_threshold=.6,
                         trajectory_down_threshold=.4)
    curriculum = curriculum_of(env, params)
    self.assertIsNotNone(curriculum.trajectory_scheduler)
    command.xy[:, 0] = 5.4
    command.progress[:] = 5.3
    # Bad tracking plus a failure pushes the trajectory quality to 0.5: the
    # scheduler stays neutral while relative progress still promotes the column.
    env.metrics_manager._episode_sums['velocity_error_xy_mps'][:] = 5000.
    env.termination_manager.get_term('impact')[:] = True
    logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(float(logs['quality/flat']), .5, places=4)
    self.assertAlmostEqual(float(logs['trajectory_up/flat']), 0.)
    self.assertAlmostEqual(float(logs['progress_ratio/flat']), 1., places=4)
    self.assertAlmostEqual(float(logs['progress_up/flat']), 1.)
    self.assertEqual(curriculum.terrain.terrain_levels.tolist(), [2, 2, 2, 2])


if __name__ == '__main__':
  unittest.main()
