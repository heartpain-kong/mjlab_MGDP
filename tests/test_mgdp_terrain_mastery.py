"""Terrain transitions, successful obstacle braking, and resume continuity."""
import unittest
from types import SimpleNamespace as NS
import torch
from src.tasks.velocity.mdp.passage import PassageCurriculum
from src.config.passage_speed import validate_passage_speed_resume

class TerrainMasteryTests(unittest.TestCase):
  def setup_case(self):
    from tests.test_mgdp_passage import MGDPPassageTests
    env,xy,failed=MGDPPassageTests().fake_env()
    env.scene.terrain.terrain_origins=torch.zeros(3,2,3)
    env.scene.terrain.env_origins=torch.zeros(4,3)
    params=dict(min_episodes=2,terrain_max_error=None,level_up_streak=2,level_down_streak=2)
    env.cfg.curriculum={'passage':NS(params=params)}
    return env,xy,failed,params,PassageCurriculum(NS(params=params),env)

  def test_successful_braking_advances_after_two_but_does_not_expand_speed(self):
    env,xy,failed,p,c=self.setup_case();xy[:,0]=5.4
    env.metrics_manager._episode_sums['velocity_error_xy_mps'].fill_(50)
    for expected in (0,1,1,2,2,2):
      logs=c(env,torch.arange(4),**p)
      self.assertTrue((c.terrain.terrain_levels==expected).all())
    self.assertEqual(c.command.cfg.ranges.lin_vel_x[1],.35)
    self.assertAlmostEqual(logs['tracking_error/flat'].item(),.5)
    self.assertEqual(logs['level_mean/flat'].item(),2)

  def test_isolated_failure_retains_stage_and_consecutive_failures_regress(self):
    env,xy,failed,p,c=self.setup_case();c.terrain.terrain_levels.fill_(1)
    failed[:]=True;c(env,torch.arange(4),**p)
    self.assertTrue((c.terrain.terrain_levels==1).all())
    xy[:,0]=5.4;failed[:]=False;c(env,torch.arange(4),**p)
    failed[:]=True;c(env,torch.arange(4),**p)
    self.assertTrue((c.terrain.terrain_levels==1).all())
    c(env,torch.arange(4),**p)
    self.assertTrue((c.terrain.terrain_levels==0).all())

  def test_partial_reset_and_resume_keep_per_environment_mastery(self):
    env,xy,failed,p,c=self.setup_case();xy[:,0]=5.4
    c(env,torch.tensor([0,2]),**p)
    state=c.state_dict();other=PassageCurriculum(NS(params=p),env);other.load_state_dict(state)
    other(env,torch.tensor([0]),**p)
    self.assertEqual(other.terrain.terrain_levels.tolist(),[1,0,0,0])
    self.assertEqual(other.level_success_streak.tolist(),[0,0,1,0])
    env.episode_length_buf.zero_();other(env,torch.arange(4),**p)
    self.assertEqual(other.level_success_streak.tolist(),[0,0,1,0])
    bad={**state,'level_failure_streak':torch.tensor([0.,float('nan'),0.,0.])}
    with self.assertRaisesRegex(ValueError,'mastery'):other.load_state_dict(bad)

  def test_changed_rules_reject_full_resume_before_physics(self):
    env,xy,failed,p,c=self.setup_case()
    with self.assertRaisesRegex(ValueError,'warm-start-checkpoint'):
      validate_passage_speed_resume(env.cfg,{'infos':{}})
    validate_passage_speed_resume(env.cfg,{'infos':{'mgdp_passage_curriculum':c.state_dict()}})

if __name__=='__main__':unittest.main()
