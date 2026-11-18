"""Terrain-grouped passage speed curriculum: one stuck lesson freezes itself only."""

import unittest
from types import SimpleNamespace as NS
import torch
from mjlab.terrains import TerrainGeneratorCfg
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg


class FakeCommand:
  """Minimal PassageCommand stand-in that keeps the per-environment bands."""

  def __init__(self, cfg, terrain_types):
    self.cfg = cfg
    self.terrain_types = terrain_types
    self.xy = torch.zeros(len(terrain_types), 2)
    lower, upper = (float(value) for value in cfg.ranges.lin_vel_x)
    self.speed = torch.zeros(len(terrain_types))
    self.speed_floors = torch.full((len(terrain_types),), lower)
    self.speed_limits = torch.full((len(terrain_types),), upper)

  def local_xy(self):
    return self.xy

  def column_mask(self, columns):
    columns = torch.as_tensor(columns, dtype=torch.long).flatten()
    if len(columns) == 0:
      return torch.zeros(len(self.terrain_types), dtype=torch.bool)
    return torch.isin(self.terrain_types, columns)

  def set_speed_limits(self, columns, low, high):
    self.speed_floors[self.column_mask(columns)] = float(low)
    self.speed_limits[self.column_mask(columns)] = float(high)

  def set_global_speeds(self, low, high):
    self.speed_floors.fill_(float(low))
    self.speed_limits.fill_(float(high))


def fake_env(speed_groups=None, ranges=(.25, .35), floor=.35, ceiling=2.):
  """Four environments on four columns: 0-1 flat, 2-3 high_wall."""
  terrain_types = torch.tensor([0, 1, 2, 3])
  num_envs = len(terrain_types)
  cfg = NS(goal_x=5.3, lane_half_width=.28, ranges=NS(lin_vel_x=tuple(ranges)),
           speed_groups=speed_groups, speed_group_floor=floor, speed_group_ceiling=ceiling)
  command = FakeCommand(cfg, terrain_types)
  terrain = NS(terrain_types=terrain_types, terrain_levels=torch.zeros(num_envs, dtype=torch.long),
               max_terrain_level=3, terrain_origins=torch.zeros(3, 4, 3),
               env_origins=torch.zeros(num_envs, 3),
               cfg=NS(terrain_generator=TerrainGeneratorCfg(size=(8., 8.), num_cols=4, num_rows=3,
                 sub_terrains={'flat': PassageTerrainCfg(kind='start_flat', proportion=.5),
                               'high_wall': PassageTerrainCfg(kind='high_wall', proportion=.5)})))

  def update_env_origins(ids, up, down):
    terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + up.long() - down.long()).clamp(0, 2)

  terrain.update_env_origins = update_env_origins
  env = NS(num_envs=num_envs, device='cpu', step_dt=.02, scene=NS(terrain=terrain),
           command_manager=NS(get_term=lambda name: command),
           episode_length_buf=torch.full((num_envs,), 100),
           reset_buf=torch.ones(num_envs, dtype=torch.bool),
           cfg=NS(terminations={'impact': NS(time_out=False)}),
           termination_manager=NS(get_term=lambda name: torch.zeros(num_envs, dtype=torch.bool)),
           metrics_manager=NS(_step_count=torch.full((num_envs,), 100),
                              _episode_sums={'velocity_error_xy_mps': torch.zeros(num_envs)}))
  return env, command.xy, command


GROUPS = {'flat': ('flat',), 'wall': ('high_wall',)}


class SpeedGroupTests(unittest.TestCase):
  def test_stuck_wall_group_only_freezes_itself(self):
    env, xy, command = fake_env(GROUPS)
    curriculum = passage.PassageCurriculum(NS(params={}), env)
    self.assertTrue(curriculum.grouped)
    xy[:, 0] = 5.4
    xy[2:, 0] = 0.  # high_wall never crosses
    logs = None
    for _ in range(3):
      logs = curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .375)
    self.assertAlmostEqual(curriculum.groups['wall']['high'], .35)
    self.assertEqual(curriculum.groups['flat']['streak'], 0)  # spent on the upgrade
    self.assertEqual(curriculum.groups['wall']['streak'], 0)
    self.assertAlmostEqual(logs['speed_max/flat'].item(), .375)
    self.assertAlmostEqual(logs['speed_max/wall'].item(), .35)
    self.assertAlmostEqual(logs['speed_min/flat'].item(), .35)
    self.assertAlmostEqual(logs['speed_min/wall'].item(), .35)
    # The batch is no longer held at the slowest lesson.
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x, (.35, .375))
    self.assertTrue(torch.all(command.speed_limits[:2] == .375))
    self.assertTrue(torch.all(command.speed_limits[2:] == .35))
    # One extra successful window per group keeps expanding independently.
    for _ in range(3):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .4)
    self.assertAlmostEqual(curriculum.groups['wall']['high'], .35)
    self.assertAlmostEqual(curriculum.command.cfg.ranges.lin_vel_x[1], .4)

  def test_group_ceiling_and_floor_bound_the_band(self):
    env, xy, command = fake_env(GROUPS, ranges=(.2, .3), floor=.3, ceiling=.34)
    curriculum = passage.PassageCurriculum(NS(params={}), env)
    self.assertAlmostEqual(curriculum.groups['flat']['low'], .3)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .3)
    xy[:, 0] = 5.4
    for _ in range(30):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(curriculum.groups['flat']['high'], .34)

  def test_legacy_single_gate_is_unchanged(self):
    env, xy, command = fake_env(None)
    curriculum = passage.PassageCurriculum(NS(params={}), env)
    self.assertFalse(curriculum.grouped)
    self.assertEqual(list(curriculum.groups), ['all'])
    xy[:, 0] = 5.4
    for i in range(3):
      curriculum(env, torch.arange(4), min_episodes=1)
      self.assertAlmostEqual(curriculum.command.cfg.ranges.lin_vel_x[1], .375 if i == 2 else .35)
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x[0], .25)
    # A single failing column still blocks the whole batch, as before.
    curriculum.command.cfg.ranges.lin_vel_x = (.25, .35)
    curriculum.count.zero_(); curriculum.success.zero_(); curriculum.error.zero_()
    curriculum.streak = 0
    xy[3, 0] = 0.
    for _ in range(3):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x[1], .35)
    self.assertNotIn('speed_groups', curriculum.state_dict())

  def test_state_round_trip_keeps_each_group_speed(self):
    env, xy, command = fake_env(GROUPS)
    xy[:, 0] = 5.4
    xy[2:, 0] = 0.
    curriculum = passage.PassageCurriculum(NS(params={}), env)
    for _ in range(2):
      curriculum(env, torch.arange(4), min_episodes=1)
    self.assertEqual(curriculum.groups['flat']['streak'], 2)
    state = curriculum.state_dict()
    self.assertEqual(state['speed_groups']['flat']['high'], .35)
    self.assertEqual(state['speed_groups']['wall']['high'], .35)

    other_env, other_xy, other_command = fake_env(GROUPS)
    other_xy[:, 0] = 5.4
    other_xy[2:, 0] = 0.
    restored = passage.PassageCurriculum(NS(params={}), other_env)
    restored.load_state_dict(state)
    self.assertEqual(restored.groups['flat']['streak'], 2)
    self.assertEqual(restored.groups['wall']['streak'], 0)
    restored(other_env, torch.arange(4), min_episodes=1)
    self.assertAlmostEqual(restored.groups['flat']['high'], .375)
    self.assertAlmostEqual(restored.groups['wall']['high'], .35)
    self.assertTrue(torch.all(other_command.speed_limits[:2] == .375))
    self.assertTrue(torch.all(other_command.speed_limits[2:] == .35))

    # Checkpoints from before per-group speeds fall back to the global band.
    legacy = {key: value for key, value in state.items() if key != 'speed_groups'}
    restored.load_state_dict(legacy)
    self.assertAlmostEqual(restored.groups['flat']['high'], .35)
    self.assertAlmostEqual(restored.groups['wall']['high'], .35)

    with self.assertRaisesRegex(ValueError, 'speed groups'):
      restored.load_state_dict({**state, 'speed_groups': {'flat': state['speed_groups']['flat']}})
    with self.assertRaisesRegex(ValueError, 'Invalid passage speed range'):
      restored.load_state_dict({**state, 'speed_groups': {
        **state['speed_groups'], 'wall': {'low': .35, 'high': 2.5, 'streak': 0}}})

  def test_group_configuration_is_validated(self):
    kinds = torch.tensor([0, 0, 1, 1])
    names = ['flat', 'high_wall']
    with self.assertRaisesRegex(ValueError, 'unknown terrain'):
      passage.resolve_speed_groups({'a': ('nope',)}, names, kinds)
    with self.assertRaisesRegex(ValueError, 'two speed groups'):
      passage.resolve_speed_groups({'a': ('flat',), 'b': ('flat',)}, names, kinds)
    with self.assertRaisesRegex(ValueError, 'no terrains'):
      passage.resolve_speed_groups({'a': ()}, names, kinds)
    defaulted = passage.resolve_speed_groups({'a': ('flat',)}, names, kinds)
    self.assertEqual(set(defaulted), {'a', 'default'})


if __name__ == '__main__':
  unittest.main()
