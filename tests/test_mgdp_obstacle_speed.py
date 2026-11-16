import unittest,torch
from types import SimpleNamespace as NS
from src.config.mgdp_course_contract import obstacle_speed_contract,validate_obstacle_speed_resume,restore_obstacle_speed
from src.tasks.velocity.mdp.wall_training import WallProgressReward

class ObstacleSpeedTests(unittest.TestCase):
  def test_near_walls_slow_far_walls_restore_and_other_columns_unaffected(self):
    from tests.test_mgdp_course_v3 import CourseV3Tests
    env,cmd,robot=CourseV3Tests().command_env();cmd.speed.fill_(.6)
    # Wall centre relative to spawn is 1.3. Initial stage has only one wall.
    for x,expected in ((0.,.6),(.8,.25),(1.7,.25),(2.,.6),(2.7,.6)):
      robot.data.root_link_pos_w[4,0]=x;cmd._update_command()
      self.assertAlmostEqual(cmd.command[4,:2].norm().item(),expected,places=5)
      self.assertAlmostEqual(cmd.command[0,:2].norm().item(),.6,places=5)
    # Later stages include the second wall at relative X=2.7.
    env.scene.terrain.terrain_levels[4]=7;cmd._update_command()
    self.assertAlmostEqual(cmd.command[4,:2].norm().item(),.25,places=5)
    robot.data.heading_w[9]=torch.atan2(torch.tensor(-.6),torch.tensor(.65));cmd._update_command()
    self.assertLessEqual(cmd.command[9,:2].norm().item(),.250001)
    cmd.speed.fill_(.15);cmd._update_command()
    self.assertLessEqual(cmd.command[:,:2].norm(dim=1).max().item(),.150001)

  def test_old_playback_and_mismatched_resume(self):
    from mjlab.tasks.registry import load_env_cfg
    cfg=load_env_cfg('LgDog1-Race-MGDP-WallTraining')
    saved={'infos':{'mgdp_obstacle_speed':obstacle_speed_contract(cfg)}}
    validate_obstacle_speed_resume(cfg,saved)
    with self.assertRaisesRegex(ValueError,'warm-start-checkpoint'):validate_obstacle_speed_resume(cfg,{'infos':{}})
    restore_obstacle_speed(cfg,{'infos':{}});self.assertIsNone(cfg.commands['twist'].obstacle_speed_limits)
    restore_obstacle_speed(cfg,saved);self.assertEqual(cfg.commands['twist'].obstacle_speed_limits,(.25,.25))
    with self.assertRaises(ValueError):restore_obstacle_speed(cfg,{'infos':{'mgdp_obstacle_speed':{'version':1,'limits':[float('nan'),.25]}}})

  def test_progress_gate_does_not_penalize_requested_slowdown(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    from mjlab.managers.reward_manager import RewardTermCfg
    env,xy,progress,_=WallTrainingTests().fake();cmd=env.command_manager.get_term('twist')
    cmd.cfg.obstacle_speed_limits=(.25,.25);cmd.command[:,:2]=torch.tensor([.25,0.])
    base=lambda env:torch.ones(env.num_envs)
    tc=RewardTermCfg(func=WallProgressReward,weight=1.,params={'base_func':base,'progress_reference_mps':.6})
    term=WallProgressReward(tc,env);term(env,base_func=base,progress_reference_mps=.6)
    progress[:]=.25*env.step_dt
    value=term(env,base_func=base,progress_reference_mps=.6)
    self.assertAlmostEqual(value[0].item(),1.,places=5)

if __name__=='__main__':unittest.main()
