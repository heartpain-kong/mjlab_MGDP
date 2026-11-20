import unittest
from types import SimpleNamespace as NS
import numpy as np,torch,mujoco
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.config.mgdp_camera import D435I20_TASK
from src.config.mgdp_skill_training import configure_skill_training,restore_skill_for_play
from src.tasks.velocity.mdp.wall_training import WallGeometry

class SkillV4Tests(unittest.TestCase):
  def test_two_walls_all_stages_and_reward_positions_match(self):
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c,4)
    wall=c.scene.terrain.terrain_generator.sub_terrains['high_wall']
    for i in range(8):
      spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
      wall.function((i+.5)/8,spec,np.random.default_rng(42));m=spec.compile()
      ids=[j for j in range(m.ngeom) if 'high_wall_' in m.geom(j).name and not m.geom(j).name.endswith('lesson_floor')]
      np.testing.assert_allclose(np.unique(m.geom_pos[ids,0]),[2.3,4.7])
      np.testing.assert_allclose(m.geom_pos[ids,2]+m.geom_size[ids,2],wall.heights[i])
    from tests.test_mgdp_wall_training import WallTrainingTests
    env,_,_,_=WallTrainingTests().fake()
    env.scene.terrain.cfg.terrain_generator=c.scene.terrain.terrain_generator
    term=WallGeometry(NS(),env)
    torch.testing.assert_close(term.centres,torch.tensor([1.3,3.7,6.1]))
    self.assertTrue((term.counts==2).all())

  def test_speed_caps_and_lateral_limit(self):
    from tests.test_mgdp_course_v3 import CourseV3Tests
    from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommand
    env,_,robot=CourseV3Tests().command_env();configure_skill_training(env.cfg,4)
    cmd=BodyAwarePassageCommand(env.cfg.commands['twist'],env)
    env.command_manager=NS(get_term=lambda _:cmd);cmd._resample_command(torch.arange(10))
    cmd.speed.fill_(1.);robot.data.heading_w[:]=0
    robot.data.root_link_pos_w[:,:2]=env.scene.env_origins[:,:2]
    cmd._update_command()
    self.assertAlmostEqual(cmd.command[4,:2].norm().item(),.6,places=5)
    self.assertAlmostEqual(cmd.command[9,:2].norm().item(),.6,places=5)
    robot.data.root_link_pos_w[4,0]+=1.3
    cmd._update_command();self.assertAlmostEqual(cmd.command[4,:2].norm().item(),.35,places=5)
    robot.data.root_link_pos_w[4,0]+=2.4
    cmd._update_command();self.assertAlmostEqual(cmd.command[4,:2].norm().item(),.35,places=5)

  def test_old_v3_restores_original_wall_geometry_and_speed(self):
    c=load_env_cfg(D435I20_TASK,play=True)
    restore_skill_for_play(c,{'infos':{'mgdp_skill_training':{'version':3}}})
    self.assertFalse(hasattr(c.scene.terrain.terrain_generator.sub_terrains['high_wall'],'wall_centres'))
    self.assertEqual(c.commands['twist'].obstacle_speed_limits,(.65,.25))

if __name__=='__main__':unittest.main()
