import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from src.config.mgdp_camera import D435I20_TASK
from src.config.mgdp_skill_training import configure_skill_training,restore_skill_for_play,configure_skill_policy,validate_skill_resume
from src.tasks.velocity.mdp.obstacle_rules import BeamGroundContact,WallVelocityMask,WallDistanceProgress

class ObstacleRuleTests(unittest.TestCase):
  def test_wall_distance_credits_fast_steps_once_and_other_terrains_keep_cap(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    env,xy,p,failure=WallTrainingTests().fake()
    term=WallDistanceProgress(NS(),env);term.reset()
    p[:]=.2
    result=term(env)*env.step_dt*1.5
    self.assertAlmostEqual(result[0].item(),.3,places=6)
    self.assertAlmostEqual(result[4].item(),.03,places=6)
    p[:]=.1;self.assertEqual(term(env).sum(),0.)
    p[:]=.2;self.assertEqual(term(env).sum(),0.)
    p[:]=.3;failure[:]=True;self.assertEqual(term(env).sum(),0.)
  def test_single_loaded_floor_foot_invalidates_only_beam_field(self):
    from tests.test_mgdp_tire_guidance import fixture
    env,_,_,clear,_,_=fixture();env.scene.terrain.terrain_types[:]=torch.tensor([2,0])
    term=BeamGroundContact(NS(),env);d=env.scene['stone_support_contact'].data
    def contact(x,z=0.,force=2.,normal=1.):
      clear();d.found_history[:,0,0]=1;d.force_history[:,0,0,2]=force
      d.normal_history[:,0,0,2]=normal
      d.pos_history[:,0,0]=env.scene.terrain.env_origins+torch.tensor([x,1.,z])
      return term(env)
    self.assertEqual(contact(2.).tolist(),[True,False])
    for x in (1.,4.8):self.assertFalse(contact(x).any()) # approach and exit
    self.assertFalse(contact(2.,.1).any()) # bridge top
    self.assertFalse(contact(2.,force=.1).any())
    self.assertFalse(contact(2.,normal=0.).any())
    clear();self.assertFalse(term(env).any())

  def test_wall_tracking_zero_other_lessons_and_negative_penalties_preserved(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    env,_,_,_=WallTrainingTests().fake();value=torch.tensor([2.,2.,-1.,2.,3.,2.])
    cfg=NS(params={'velocity_base_func':lambda env:value},weight=1.)
    term=WallVelocityMask(cfg,env);actual=term(env,**cfg.params)
    torch.testing.assert_close(actual,torch.tensor([0.,0.,-1.,0.,3.,0.]))
    cfg.weight=-1.
    negative=WallVelocityMask(cfg,env)
    torch.testing.assert_close(negative(env,**cfg.params),value)

  def test_v3_resume_and_v2_play_isolation(self):
    cfg=load_env_cfg(D435I20_TASK);configure_skill_training(cfg,3)
    self.assertIn('beam_ground_contact',cfg.terminations)
    self.assertEqual(cfg.rewards['passage_progress'].weight,1.5)
    self.assertEqual(cfg.rewards['track_linear_velocity'].func,WallVelocityMask)
    with self.assertRaises(ValueError):validate_skill_resume(cfg,{'infos':{'mgdp_skill_training':{'version':2}}})
    old=load_env_cfg(D435I20_TASK,play=True);restore_skill_for_play(old,{'infos':{'mgdp_skill_training':{'version':2}}})
    self.assertNotIn('beam_ground_contact',old.terminations)
    new=load_env_cfg(D435I20_TASK,play=True);restore_skill_for_play(new,{'infos':{'mgdp_skill_training':{'version':3}}})
    self.assertIn('beam_ground_contact',new.terminations)
    agent=load_rl_cfg(D435I20_TASK);configure_skill_policy(agent,new)
    self.assertIn('BoundedMeanGaussian',agent.actor.distribution_cfg['class_name'])

  def test_workbench_v3_flag(self):
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      argv=build_command(Path(tmp),'python',{'mode':'train','task':'race_mgdp_wall_d435i20',
        'mgdp_vision_mode':'height_oracle','mgdp_skill_training':True,'mgdp_skill_version':3})
      self.assertIn('--mgdp-skill-version=3',argv)

if __name__=='__main__':unittest.main()
