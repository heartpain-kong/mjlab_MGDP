"""三段式上墙引导：抬脚给分、越过前沿给一次性分、缺课目时恒 0。"""
import unittest
from types import SimpleNamespace as NS

import torch

import src.tasks  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from src.tasks.velocity.mdp.wall_training import WallClimbProgress

TASK = "LgDog1-Race-MGDP-WallTraining"


class Scene(dict):
  @property
  def terrain(self):
    return self["terrain"]


class WallClimbProgressTests(unittest.TestCase):
  def fake(self, levels=(0, 0, 0)):
    n = len(levels)
    generator = load_env_cfg(TASK).scene.terrain.terrain_generator
    generator.num_cols = 10
    wall_index = list(generator.sub_terrains).index("high_wall")
    terrain = NS(cfg=NS(terrain_generator=generator),
                 terrain_types=torch.tensor([wall_index] * n),
                 terrain_levels=torch.tensor(levels),
                 env_origins=torch.zeros(n, 3), max_terrain_level=8)
    robot = NS(data=NS(site_pos_w=torch.zeros(n, 4, 3)))
    xy = torch.zeros(n, 2)
    command = NS(local_xy=lambda: xy)
    env = NS(num_envs=n, device="cpu", scene=Scene(terrain=terrain, robot=robot),
             command_manager=NS(get_term=lambda name: command))
    return env, robot, xy, generator

  def term(self):
    env, robot, xy, generator = self.fake()
    cfg = NS(params={})
    return WallClimbProgress(cfg, env), env, robot, xy, generator

  def test_lifting_a_foot_onto_the_wall_pays_once_per_foot(self):
    term, env, robot, xy, generator = self.term()
    top = generator.sub_terrains["high_wall"].heights[0]
    centre = term.centres[0].item()
    robot.data.site_pos_w[0, :, 0] = centre
    robot.data.site_pos_w[0, :, 2] = top + .05
    cfg = SceneEntityCfg("robot", site_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"))
    value = term(env, cfg)
    self.assertAlmostEqual(float(value[0]), 4 * .4, places=6)
    self.assertAlmostEqual(float(value[1]), 0., places=6)

  def test_feet_below_the_top_or_far_from_the_wall_earn_nothing(self):
    term, env, robot, xy, generator = self.term()
    top = generator.sub_terrains["high_wall"].heights[0]
    robot.data.site_pos_w[0, :, 0] = term.centres[0]
    robot.data.site_pos_w[0, :, 2] = top - .01      # below the top
    robot.data.site_pos_w[1, :, 0] = term.centres[0] + 5.  # far away
    robot.data.site_pos_w[1, :, 2] = top + .2
    cfg = SceneEntityCfg("robot", site_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"))
    value = term(env, cfg)
    self.assertAlmostEqual(float(value.sum()), 0., places=6)

  def test_crest_bonus_requires_a_prior_lift_and_is_paid_once(self):
    term, env, robot, xy, generator = self.term()
    top = generator.sub_terrains["high_wall"].heights[0]
    cfg = SceneEntityCfg("robot", site_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"))
    # Walk past the wall without ever lifting a foot: no crest bonus.
    xy[0, 0] = term.centres[0] + .5
    self.assertAlmostEqual(float(term(env, cfg)[0]), 0., places=6)
    # Lift a foot on the wall, then push the root past the front edge.
    xy[0, 0] = term.centres[0].item()
    robot.data.site_pos_w[0, :, 0] = term.centres[0]
    robot.data.site_pos_w[0, :, 2] = top + .05
    term(env, cfg)
    xy[0, 0] = term.centres[0] + .5
    robot.data.site_pos_w[0, :, 2] = 0.
    self.assertAlmostEqual(float(term(env, cfg)[0]), 1.2, places=6)
    self.assertAlmostEqual(float(term(env, cfg)[0]), 0., places=6)  # paid once


if __name__ == "__main__":
  unittest.main()
