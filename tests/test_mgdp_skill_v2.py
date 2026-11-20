"""Skill-v2 objective, geometry, likelihood and compatibility contracts."""
import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import numpy as np
import torch
import mujoco
import src.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from src.config.mgdp_camera import D435I20_TASK
from src.config.mgdp_skill_training import configure_skill_training,configure_skill_policy,restore_skill_for_play,validate_skill_resume
from src.config.mgdp_vision import configure_vision,restore_vision_for_play,validate_vision_resume
from src.tasks.velocity.mdp.skill_training import SkillPhaseReward,FrontFootWallApproach
from src.tasks.velocity.rl.skill_distribution import BoundedMeanGaussian

class SkillV2Tests(unittest.TestCase):
  def test_mean_export_likelihood_gradients_and_exploration_floor(self):
    d=BoundedMeanGaussian(3)
    x=torch.tensor([[0.,5.,-10.]],requires_grad=True);d.update(x)
    torch.testing.assert_close(d.mean,x/(1+x.abs()))
    self.assertTrue((d.mean.abs()<1).all())
    sample=d.sample();torch.testing.assert_close(d.log_prob(sample),torch.distributions.Normal(d.mean,d.std).log_prob(sample).sum(-1))
    d.log_prob(torch.zeros_like(x)).sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad[0,1:]!=0).all())
    torch.testing.assert_close(torch.jit.script(d.as_deterministic_output_module())(x),d.deterministic_output(x))
    with torch.no_grad():d.std_param[:]=torch.tensor([-.1,.2,2.])
    d.update(x);torch.testing.assert_close(d.std,torch.tensor([[.05,.2,.35]]))
    self.assertEqual(d.kl_divergence(d.params,d.params).item(),0.)

  def test_entrance_gap_and_competition_geometry(self):
    from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import StrictStonesTerrainCfg
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c)
    skill=c.scene.terrain.terrain_generator.sub_terrains['stepping_stones']
    gaps=[]
    for stage in range(8):
      built=[]
      for terrain in (StrictStonesTerrainCfg(size=(12.,8.)),skill):
        s=mujoco.MjSpec();s.worldbody.add_body(name='terrain')
        terrain.function((stage+.5)/8,s,np.random.default_rng(42));m=s.compile()
        built.append({m.geom(i).name:np.r_[m.geom_pos[i],m.geom_size[i]] for i in range(m.ngeom)})
      if stage>=5:
        self.assertEqual(set(built[0]),set(built[1]))
        for name in built[0]:np.testing.assert_array_equal(built[0][name],built[1][name])
      else:
        tongues=[v for k,v in built[1].items() if k.endswith('_right_tongue')]
        self.assertTrue(tongues)
        end=tongues[0][0]+tongues[0][3]
        gaps.append(2.7-skill.widths[stage]/2-end)
    np.testing.assert_allclose(gaps,[.08,.12,.18,.24,.30],atol=1e-6)

  def test_phase_progress_preserves_negatives_and_does_not_farm(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    env,xy,p,_=WallTrainingTests().fake();cmd=env.command_manager.get_term('twist')
    env.scene.terrain.terrain_types.fill_(1)
    cmd.heading_error=torch.ones(6);cmd.waypoint_index=torch.zeros(6,dtype=torch.long)
    value=torch.ones(6);value[1]=-2.
    cfg=NS(params={'wrapped_func':lambda env:value},weight=1.)
    term=SkillPhaseReward(cfg,env)
    self.assertEqual(term(env,**cfg.params)[0],0.)
    p[:]=.006;self.assertEqual(term(env,**cfg.params)[0],1.)
    p[:]=0.;self.assertEqual(term(env,**cfg.params)[0],0.)
    p[:]=.006;result=term(env,**cfg.params)
    self.assertEqual(result[0],0.);self.assertEqual(result[1],-2.)
    cmd.command[:]=torch.tensor([0.,0.,.5]);cmd.heading_error[:]=.99
    self.assertAlmostEqual(term(env,**cfg.params)[0].item(),1.,places=5)
    cmd.heading_error[:]=1.;term(env,**cfg.params)
    cmd.heading_error[:]=.99;self.assertEqual(term(env,**cfg.params)[0],0.)
    cmd.waypoint_index+=1;cmd.heading_error[:]=.5
    self.assertEqual(term(env,**cfg.params)[0],0.)
    term.reset(torch.tensor([0]));self.assertFalse(term.ready[0])

  def test_front_foot_hint_finite_and_reset_seeds_without_credit(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    env,xy,_,failure=WallTrainingTests().fake();asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
    term=FrontFootWallApproach(NS(),env);term(env,asset)
    feet=env.scene['robot'].data.site_pos_w;feet[:,:2,0]=1.3;feet[:,:2,2]=.14
    first=term(env,asset)*env.step_dt
    self.assertGreater(first[0],0);self.assertLessEqual(first[0],1.)
    self.assertEqual(term(env,asset)[0],0.)
    feet[:,:2,0]=0.;term(env,asset);feet[:,:2,0]=1.3
    self.assertEqual(term(env,asset)[0],0.)
    term.reset(torch.tensor([0]));self.assertEqual(term(env,asset)[0],0.)

  def test_v1_restore_and_cross_version_resume_guard(self):
    c=load_env_cfg(D435I20_TASK,play=True)
    restore_skill_for_play(c,{'infos':{'mgdp_skill_training':{'version':1}}})
    self.assertEqual(c.scene.terrain.terrain_generator.sub_terrains['stepping_stones'].entry_extensions,(0.,)*8)
    self.assertNotIn('front_wall_approach',c.rewards)
    a=load_rl_cfg(D435I20_TASK);configure_skill_policy(a,c)
    self.assertEqual(a.actor.distribution_cfg['class_name'],'GaussianDistribution')
    with self.assertRaises(ValueError):validate_skill_resume(c,{'infos':{'mgdp_skill_training':{'version':2}}})

  def test_oracle_actor_contract_and_dashboard(self):
    c=load_env_cfg(D435I20_TASK);a=load_rl_cfg(D435I20_TASK)
    configure_vision(c,a,'height_oracle')
    self.assertNotIn('perception',c.observations)
    self.assertEqual(a.obs_groups['actor'],['actor','terrain_oracle'])
    self.assertFalse(c.observations['terrain_oracle'].enable_corruption)
    with self.assertRaises(ValueError):validate_vision_resume(c,{})
    validate_vision_resume(c,{'infos':{'mgdp_vision_mode':'height_oracle'}})
    p=load_env_cfg(D435I20_TASK,play=True);restore_vision_for_play(p,a,{'infos':{'mgdp_vision_mode':'height_oracle'}})
    self.assertIn('terrain_oracle',p.observations)
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      args={'mode':'train','task':'race_mgdp_wall_d435i20','mgdp_vision_mode':'height_oracle','mgdp_skill_training':True}
      self.assertIn('--mgdp-vision-mode=height_oracle',build_command(Path(tmp),'python',args))
      args['warm_start']=True
      with self.assertRaises(ValueError):build_command(Path(tmp),'python',args)

  def test_real_play_entry_restores_oracle_before_encoder_or_physics(self):
    from unittest.mock import patch
    from scripts import play
    from src.config.mgdp_course_contract import current_contract,obstacle_speed_contract
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c)
    infos={'mgdp_skill_training':{'version':2},'mgdp_vision_mode':'height_oracle',
      'mgdp_course_contract':current_contract(c),'mgdp_obstacle_speed':obstacle_speed_contract(c)}
    class StopBeforePhysics(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
      p=Path(tmp)/'model_0.pt';torch.save({'infos':infos},p)
      with patch.object(play,'configure_torch_backends'),patch.object(play,'ManagerBasedRlEnv',side_effect=StopBeforePhysics) as create:
        with self.assertRaises(StopBeforePhysics):play.run_play(D435I20_TASK,play.PlayConfig(checkpoint_file=str(p),device='cpu'))
      configured=create.call_args.kwargs['cfg']
      self.assertIn('terrain_oracle',configured.observations)
      self.assertNotIn('perception',configured.observations)

if __name__=='__main__':unittest.main()
