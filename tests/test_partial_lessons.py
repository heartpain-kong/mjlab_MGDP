"""Graceful degradation for trimmed wall-training terrains.

``LGDOG_WALL_LESSONS`` (and the distillation single-lesson cfgs built by
``filter_terrain_lessons``) shrink the wall course to a subset of its ten
lessons.  Managers used to hard-code the full roster and crashed at env
construction (``list.index``/``sub_terrains[kind]``); these gates pin the
degradation contract: no exception, wall-only logic disabled, and the full
ten-lesson course unchanged.
"""
import os
import unittest
import warnings
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.mdp.course_v3 import (
  BodyAwarePassageCommand, SlalomBodyProjection, StoneGroundContact,
)
from src.tasks.velocity.mdp.wall_training import (
  WallApproachReward, WallCrossingBonus, WallProgressReward, WallSwingClearance,
  WallWaitingPenalty,
)
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import START_X, WALL_CENTRES


FULL_SPEED_GROUPS = {
  'open': ('start_flat', 'random_hills', 'mjlab_rough'),
  'technical': ('stepping_stones', 'balance_beam', 'gap_steps',
                'slalom_brick', 'clearance_frame', 'tire_array'),
  'wall': ('high_wall',),
}


class Scene(dict):
  @property
  def terrain(self):
    return self['terrain']


def wall_training_cfg(lessons=None):
  """Build the wall-training cfg the same way the task/registry does.

  ``lessons`` is the ``LGDOG_WALL_LESSONS`` value (``None`` keeps all ten).
  The cfg builder reads the variable at call time, so the test is deterministic
  even when the registry was imported earlier without a trimmed course.
  """
  from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg
  value = '' if lessons is None else ','.join(lessons)
  with patch.dict(os.environ, {'LGDOG_WALL_LESSONS': value}):
    return lgdog1_wall_training_env_cfg()


def lesson_cfg(lesson):
  """The distillation single-lesson cfg path (``filter_terrain_lessons``)."""
  from src.tasks.velocity.rl.distill import filter_terrain_lessons
  return filter_terrain_lessons(wall_training_cfg(), lesson)


def column_kinds(generator):
  """Expected column -> sub-terrain index, computed independently of the mdp."""
  proportions = torch.tensor([t.proportion for t in generator.sub_terrains.values()])
  return torch.searchsorted((proportions/proportions.sum()).cumsum(0),
    torch.arange(generator.num_cols)/generator.num_cols+.001, right=True)


def terrain_ns(cfg, terrain_types, terrain_levels=None):
  generator = cfg.scene.terrain.terrain_generator
  n = len(terrain_types)
  return NS(cfg=NS(terrain_generator=generator), terrain_types=torch.as_tensor(terrain_types),
            terrain_levels=(torch.zeros(n, dtype=torch.long) if terrain_levels is None
                            else torch.as_tensor(terrain_levels)),
            env_origins=torch.zeros(n, 3), max_terrain_level=8)


def command_env(cfg, terrain_types):
  """Minimal env that builds the directed passage command (no simulator)."""
  n = len(terrain_types)
  robot = NS(data=NS(root_link_pos_w=torch.zeros(n, 3), heading_w=torch.zeros(n)))
  scene = Scene(robot=robot, terrain=terrain_ns(cfg, terrain_types))
  scene.env_origins = torch.zeros(n, 3)
  env = NS(scene=scene, num_envs=n, device='cpu', step_dt=.02, cfg=cfg)
  command = BodyAwarePassageCommand(cfg.commands['twist'], env)
  env.command_manager = NS(get_term=lambda name: command)
  command._resample_command(torch.arange(n))
  return env, command, robot


def wall_env(cfg, terrain_types, levels=None):
  """Fake env for the wall rewards, mirroring tests/test_mgdp_wall_training.py."""
  n = len(terrain_types)
  terrain = terrain_ns(cfg, terrain_types, levels)
  xy = torch.zeros(n, 2)
  progress = torch.zeros(n)
  command = NS(command=torch.tensor([[.3, 0., 0.]]).repeat(n, 1), local_xy=lambda: xy,
               route_state=lambda: (None, None, xy[:, 1].abs(), progress),
               route_success=lambda: progress >= 5.3,
               cfg=NS(goal_x=5.3, lane_half_width=.28, ranges=NS(lin_vel_x=(.6, .6))))
  robot = NS(data=NS(site_pos_w=torch.zeros(n, 4, 3), site_lin_vel_w=torch.zeros(n, 4, 3),
                     root_link_lin_vel_w=torch.zeros(n, 3)))
  scene = Scene(terrain=terrain, robot=robot,
                feet_ground_contact=NS(data=NS(found=torch.zeros(n, 4))))
  env = NS(num_envs=n, device='cpu', step_dt=.02, scene=scene,
           command_manager=NS(get_term=lambda name: command), cfg=cfg,
           termination_manager=NS(get_term=lambda name: torch.zeros(n, dtype=torch.bool)))
  return env, xy, command


def stone_env(cfg, terrain_types):
  """Fake env for StoneGroundContact with the position-history contact sensor."""
  n = len(terrain_types)
  data = NS(pos_history=torch.zeros(n, 16, 1, 3), normal_history=torch.zeros(n, 16, 1, 3),
            force_history=torch.zeros(n, 16, 1, 3), found_history=torch.zeros(n, 16, 1))
  scene = Scene(terrain=terrain_ns(cfg, terrain_types),
                stone_support_contact=NS(data=data))
  return NS(scene=scene, num_envs=n, device='cpu', step_dt=.02, cfg=cfg)


class PartialLessonTests(unittest.TestCase):
  def test_only_high_wall_cfg_constructs_command_rewards_and_terminations(self):
    cfg = wall_training_cfg(('high_wall',))
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(list(generator.sub_terrains), ['high_wall'])
    n = 6
    types = torch.zeros(n, dtype=torch.long)
    env, command, _ = command_env(cfg, types)
    self.assertTrue(command.wall_available)
    self.assertTrue(bool(command.wall_columns.all()))
    self.assertFalse(bool(command.slalom_columns.any()))
    self.assertEqual(set(command.speed_group_columns), {'wall'})
    # Wall route clearance + approach speed limit run without the slalom lesson.
    command._update_command()
    self.assertTrue(torch.isfinite(command.command).all())
    # Every wall reward constructs; those that only need the wall geometry run.
    reward_env, _, _ = wall_env(cfg, types, levels=[0, 0, 7, 2, 0, 0])
    asset = SceneEntityCfg('robot', site_ids=[0, 1, 2, 3])
    crossing = WallCrossingBonus(NS(), reward_env)
    swing = WallSwingClearance(NS(), reward_env)
    waiting = WallWaitingPenalty(NS(weight=-.6), reward_env)
    approach = WallApproachReward(NS(params={}), reward_env)
    self.assertTrue(crossing.available)
    self.assertEqual(len(crossing.centres), 3)
    self.assertEqual(crossing.counts.tolist(), list(
      generator.sub_terrains['high_wall'].counts))
    for value in (crossing(reward_env, asset), swing(reward_env, asset),
                  waiting(reward_env, asset), approach(reward_env)):
      self.assertEqual(value.shape, (n,))
      self.assertTrue(torch.isfinite(value).all())
    # Terminations that reference the cut lessons must construct and stay quiet.
    stone_in_env = stone_env(cfg, types)
    stone = StoneGroundContact(NS(), stone_in_env)
    self.assertFalse(bool(stone.columns.any()))
    self.assertFalse(stone(stone_in_env, min_feet=2).any())
    slalom = SlalomBodyProjection(NS(), stone_in_env)
    self.assertIsNone(slalom.half_size)
    self.assertFalse(slalom(stone_in_env, None).any())

  def test_only_stepping_stones_cfg_degrades_wall_terms_to_zero(self):
    cfg = lesson_cfg('stepping_stones')
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(list(generator.sub_terrains), ['stepping_stones'])
    n = 6
    types = torch.zeros(n, dtype=torch.long)  # every column is the stone lesson
    env = stone_env(cfg, types)
    stone = StoneGroundContact(NS(), env)
    self.assertTrue(bool(stone.columns.all()))
    # No floor contact recorded -> no violation, but the rule is active.
    self.assertFalse(stone(env, sensor_name='stone_support_contact', min_feet=2).any())
    # The wall lesson is gone: empty geometry and every wall term is zero.
    reward_env, _, _ = wall_env(cfg, types)
    asset = SceneEntityCfg('robot', site_ids=[0, 1, 2, 3])
    crossing = WallCrossingBonus(NS(), reward_env)
    self.assertFalse(crossing.available)
    self.assertEqual(len(crossing.centres), 0)
    self.assertEqual(int(crossing.counts), 0)
    self.assertEqual(crossing(reward_env, asset).sum().item(), 0.)
    self.assertEqual(WallSwingClearance(NS(), reward_env)(reward_env, asset).sum().item(), 0.)
    self.assertEqual(WallWaitingPenalty(NS(weight=-.6), reward_env)(
      reward_env, asset).sum().item(), 0.)
    self.assertEqual(WallApproachReward(NS(params={}), reward_env)(reward_env).sum().item(), 0.)
    # Shaping wrappers only lose their wall gate; positive shaping is untouched.
    base = lambda _: torch.full((n,), 2.)
    wrapper = WallProgressReward(RewardTermCfg(func=WallProgressReward, weight=1.,
                                              params={'base_func': base}), reward_env)
    torch.testing.assert_close(wrapper(reward_env, base_func=base), torch.full((n,), 2.))
    # The directed command disables wall-only branches and keeps the stone route.
    cmd_env, command, _ = command_env(cfg, types)
    self.assertFalse(command.wall_available)
    self.assertEqual(len(command.wall_centres), 0)
    self.assertEqual(int(command.wall_counts), 0)
    self.assertFalse(bool(command.wall_columns.any()))
    self.assertFalse(bool(command.slalom_columns.any()))
    self.assertEqual(set(command.speed_group_columns), {'technical'})
    command._update_command()
    self.assertTrue(torch.isfinite(command.command).all())
    # The pole-penetration termination is off without the slalom lesson.
    slalom = SlalomBodyProjection(NS(), env)
    self.assertFalse(slalom(env, None).any())

  def test_speed_groups_ignore_lessons_cut_from_the_terrain(self):
    names = ['stepping_stones', 'slalom_brick']
    kinds = torch.tensor([0, 0, 1, 1])
    passage._CUT_SPEED_TERRAIN_WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      groups = passage.resolve_speed_groups(FULL_SPEED_GROUPS, names, kinds)
    # open/wall had only cut lessons; technical keeps its two present lessons.
    self.assertEqual(set(groups), {'technical'})
    self.assertEqual(groups['technical'].tolist(), [0, 1, 2, 3])
    self.assertTrue(any('cut from this terrain' in str(w.message) for w in caught))
    # A name that is not a course lesson at all remains a configuration error.
    with self.assertRaisesRegex(ValueError, 'unknown terrain'):
      passage.resolve_speed_groups({'a': ('not_a_lesson',)}, names, kinds)
    # The full roster resolves to exactly the configured groups (no default).
    all_names = list(FULL_SPEED_GROUPS['open'] + FULL_SPEED_GROUPS['technical']
                     + FULL_SPEED_GROUPS['wall'])
    all_kinds = column_kinds(NS(sub_terrains={
      name: NS(proportion=.1) for name in all_names}, num_cols=20))
    resolved = passage.resolve_speed_groups(FULL_SPEED_GROUPS, all_names, all_kinds)
    self.assertEqual(set(resolved), set(FULL_SPEED_GROUPS))
    covered = torch.zeros(20, dtype=torch.bool)
    for name, members in FULL_SPEED_GROUPS.items():
      expected = torch.isin(all_kinds,
        torch.tensor([all_names.index(member) for member in members])).nonzero().flatten()
      torch.testing.assert_close(resolved[name], expected)
      covered[resolved[name]] = True
    self.assertTrue(bool(covered.all()))

  def test_full_ten_lesson_course_keeps_its_columns_and_geometry(self):
    cfg = wall_training_cfg()
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(len(generator.sub_terrains), 10)
    names = list(generator.sub_terrains)
    expected = column_kinds(generator)
    types = torch.arange(generator.num_cols, dtype=torch.long)
    env, command, _ = command_env(cfg, types)
    self.assertTrue(command.wall_available)
    torch.testing.assert_close(command.wall_columns, expected == names.index('high_wall'))
    torch.testing.assert_close(command.slalom_columns, expected == names.index('slalom_brick'))
    torch.testing.assert_close(command.wall_centres,
      torch.tensor(WALL_CENTRES) - START_X)
    self.assertEqual(command.wall_counts.tolist(),
                     list(generator.sub_terrains['high_wall'].counts))
    slalom = expected == names.index('slalom_brick')
    expected_lane = torch.where(slalom,
      torch.full((generator.num_cols,), cfg.commands['twist'].slalom_lane_half_width),
      torch.full((generator.num_cols,), cfg.commands['twist'].lane_half_width))
    torch.testing.assert_close(command.lane_limits, expected_lane)
    # Full roster: one speed group per present lesson, no automatic default.
    self.assertEqual(set(command.speed_group_columns), set(FULL_SPEED_GROUPS))
    # Wall geometry keeps the complete ladder of centres/counts/heights.
    reward_env, _, _ = wall_env(cfg, types)
    geometry = WallCrossingBonus(NS(), reward_env)
    self.assertTrue(geometry.available)
    torch.testing.assert_close(geometry.wall_columns, expected == names.index('high_wall'))
    torch.testing.assert_close(geometry.centres, torch.tensor(WALL_CENTRES) - START_X)
    active = geometry.active_walls(reward_env)
    levels = torch.zeros(len(types), dtype=torch.long)
    torch.testing.assert_close(active,
      (geometry.wall_columns[types, None] &
       (torch.arange(3)[None, :] < geometry.counts[levels, None])))
    # The helper itself is a drop-in replacement when the lesson is present.
    helper = passage.kind_columns_or_empty(env, 'stepping_stones')
    torch.testing.assert_close(helper, expected == names.index('stepping_stones'))
    self.assertFalse(passage.kind_columns_or_empty(env, 'missing_lesson').any())


if __name__ == '__main__':
  unittest.main()
