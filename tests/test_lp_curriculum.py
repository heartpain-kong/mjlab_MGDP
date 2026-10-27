"""CPU tests for the sampler and actual mjlab reset-term integration."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.curriculum.lp import LearningProgressSampler, build_terrain_tasks
from src.tasks.velocity.config.lgdog1.him_env_cfg import lgdog1_him_standard_env_cfg, lgdog1_him_flat_env_cfg
from src.tasks.velocity.config.lgdog1.terrain_curriculum import configure_terrain_curriculum
from src.tasks.velocity.mdp.lp_terrain import LPTerrainCurriculum


class SamplerTests(unittest.TestCase):
  def sampler(self, **kwargs):
    return LearningProgressSampler(3, **{"min_episodes":2, **kwargs})

  def stage(self, sampler, rewards):
    sampler.observe(np.repeat(np.arange(3),2), np.repeat(rewards,2))
    return sampler.update()

  def test_first_rewards_establish_baselines_not_progress_from_zero(self):
    s=self.sampler()
    self.stage(s,[100.,0.,-100.])
    np.testing.assert_allclose(s.probabilities,[1/3]*3)
    np.testing.assert_allclose(s.learning_progress,0.)

  def test_signed_improvement_is_prioritized_and_regression_is_not_absolute(self):
    s=self.sampler(temperature=1.,uniform_mix=.1)
    self.stage(s,[10.,10.,10.]); self.stage(s,[14.,10.,6.])
    np.testing.assert_allclose(s.learning_progress,[4.,0.,-4.])
    self.assertGreater(s.probabilities[0],s.probabilities[1])
    self.assertGreater(s.probabilities[1],s.probabilities[2])
    self.assertTrue((s.probabilities>=.1/3).all())

  def test_missing_samples_are_neither_zero_returns_nor_a_new_baseline(self):
    s=self.sampler()
    self.stage(s,[10.,20.,30.])
    s.observe(np.array([0,0,1]),np.array([12.,12.,99.]))
    s.update()
    self.assertEqual(s.mean_returns[1],20.)
    self.assertEqual(s.mean_returns[2],30.)
    self.assertEqual(s.window_counts[1],1)
    s.observe(np.array([1]),np.array([1.]))
    s.update()
    self.assertEqual(s.mean_returns[1],50.)

  def test_no_samples_does_not_update_or_create_progress(self):
    s=self.sampler()
    self.assertFalse(s.update())
    self.assertEqual(s.updates,0)

  def test_uniform_mode_remains_uniform_under_large_progress_differences(self):
    s=self.sampler(mode='uniform')
    self.stage(s,[0.,0.,0.]);self.stage(s,[100.,0.,-100.])
    np.testing.assert_allclose(s.probabilities,[1/3]*3)

  def test_softmax_is_stable_with_small_temperature_and_large_returns(self):
    s=self.sampler(temperature=1e-20)
    self.stage(s,[0.,0.,0.]); self.stage(s,[10000.,0.,-10000.])
    self.assertTrue(np.isfinite(s.probabilities).all())
    self.assertAlmostEqual(s.probabilities.sum(),1.)

  def test_checkpoint_roundtrip_preserves_partial_windows_and_random_draws(self):
    s=self.sampler(seed=7)
    self.stage(s,[3.,4.,5.]);s.observe(np.array([0,1]),np.array([5.,6.]))
    saved=s.state_dict();restored=self.sampler(seed=99);restored.load_state_dict(saved)
    np.testing.assert_array_equal(s.sample(100),restored.sample(100))
    s.observe(np.array([0,1]),np.array([7.,8.]))
    restored.observe(np.array([0,1]),np.array([7.,8.]))
    s.update();restored.update()
    np.testing.assert_allclose(s.probabilities,restored.probabilities)

  def test_bad_inputs_and_checkpoints_are_rejected(self):
    for kwargs in ({'temperature':0.},{'uniform_mix':2.},{'min_episodes':0}):
      with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.sampler(**kwargs)
    s=self.sampler()
    for ids,values in [([3],[1.]),([.5],[1.]),([0],[float('nan')])]:
      with self.assertRaises(ValueError):s.observe(ids,values)
    state=s.state_dict()
    for key,value in [('probabilities',[0.,0.,0.]),('probabilities',[.5,.5,.000001]),
                      ('counts',[1e30,0,0]),('known',[2,0,0])]:
      bad=deepcopy(state);bad[key]=value
      with self.subTest(key=key),self.assertRaises(ValueError):s.load_state_dict(bad)


class TerrainIntegrationTests(unittest.TestCase):
  def make_term(self, *, status_path=None, mode='lp_acrl', cfg=None):
    cfg=cfg or lgdog1_him_standard_env_cfg()
    cfg.seed=42
    configure_terrain_curriculum(cfg,mode,min_episodes=2,update_interval_s=100.,status_path=status_path)
    terrain_cfg=cfg.scene.terrain
    g=terrain_cfg.terrain_generator
    origins=torch.arange(g.num_rows*g.num_cols*3,dtype=torch.float32).reshape(g.num_rows,g.num_cols,3)
    terrain=NS(cfg=terrain_cfg,terrain_origins=origins,terrain_levels=torch.zeros(4,dtype=torch.long),
               terrain_types=torch.zeros(4,dtype=torch.long),env_origins=torch.zeros(4,3))
    env=NS(cfg=cfg,scene=NS(terrain=terrain),num_envs=4,device='cpu',common_step_counter=0,
           step_dt=.02,max_episode_length_s=20.,episode_length_buf=torch.zeros(4,dtype=torch.long),
           reward_manager=NS(_episode_sums={'tracking':torch.zeros(4),'posture':torch.zeros(4)}),
           metrics_manager=NS(_episode_sums={'tracking_progress':torch.zeros(4),'commanded_distance':torch.zeros(4)}),
           termination_manager=NS(terminated=torch.zeros(4,dtype=torch.bool)))
    return env,LPTerrainCurriculum(cfg.curriculum['terrain_sampling'],env)

  def test_real_config_has_no_competing_curricula_and_keeps_blind_observations(self):
    cfg=lgdog1_him_standard_env_cfg()
    obs=deepcopy(cfg.observations)
    ranges=deepcopy(cfg.commands['twist'].ranges)
    configure_terrain_curriculum(cfg,'lp_acrl')
    self.assertNotIn('command_vel',cfg.curriculum)
    self.assertNotIn('terrain_levels',cfg.curriculum)
    self.assertIs(cfg.curriculum['terrain_sampling'].func,LPTerrainCurriculum)
    self.assertEqual(cfg.commands['twist'].ranges,ranges)
    self.assertEqual(cfg.observations,obs)
    with self.assertRaisesRegex(ValueError,'平地'):
      configure_terrain_curriculum(lgdog1_him_flat_env_cfg(),'lp_acrl')

  def test_manual_configuration_is_unchanged(self):
    cfg=lgdog1_him_standard_env_cfg()
    original=deepcopy(cfg.curriculum)
    configure_terrain_curriculum(cfg,'manual')
    self.assertEqual(cfg.curriculum,original)

  def test_plain_ppo_rough_configuration_also_accepts_terrain_sampler(self):
    from src.tasks.velocity.config.lgdog1.env_cfgs import lgdog1_rough_env_cfg
    cfg=lgdog1_rough_env_cfg(use_height_scan=False)
    observations=deepcopy(cfg.observations)
    env,term=self.make_term(cfg=cfg)
    term(env,torch.arange(4))
    self.assertEqual(cfg.observations,observations)
    self.assertEqual(term.sampler.counts.sum(),0)

  def test_rows_without_difficulty_variation_are_grouped(self):
    cfg=lgdog1_him_standard_env_cfg()
    tasks=build_terrain_tasks(cfg.scene.terrain.terrain_generator)
    for name in ('flat','random_rough'):
      matches=[t for t in tasks if t['terrain']==name]
      self.assertEqual(len(matches),1)
      self.assertEqual(len(matches[0]['rows']),10)
    self.assertEqual(len([t for t in tasks if t['terrain']=='pyramid_stairs']),10)

  def test_task_columns_match_actual_installed_mjlab_generator(self):
    from mjlab.terrains.terrain_generator import TerrainGenerator
    cfg=lgdog1_him_standard_env_cfg().scene.terrain.terrain_generator
    cfg.num_rows=2;cfg.num_cols=15
    native=TerrainGenerator.__new__(TerrainGenerator)
    native.cfg=cfg;native.np_rng=np.random.default_rng(42)
    native.terrain_origins=np.zeros((2,15,3))
    observed={}
    def capture(spec,position,difficulty,sub,row,col):
      observed[col]=next(name for name,value in cfg.sub_terrains.items() if value is sub)
      return np.zeros(3)
    native._create_terrain_geom=capture
    native._generate_curriculum_terrains(object())
    for task in build_terrain_tasks(cfg):
      for col in task['columns']:
        self.assertEqual(observed[col],task['terrain'])

  def test_initial_reset_assigns_tiles_without_fake_episode_statistics(self):
    env,term=self.make_term()
    term(env,torch.arange(4))
    self.assertEqual(term.sampler.counts.sum(),0)
    self.assertTrue((term.env_task_ids>=0).all())
    for i in range(4):
      task=term.tasks[int(term.env_task_ids[i])]
      self.assertIn(int(env.scene.terrain.terrain_levels[i]),task['rows'])
      self.assertIn(int(env.scene.terrain.terrain_types[i]),task['columns'])
    expected=env.scene.terrain.terrain_origins[env.scene.terrain.terrain_levels,env.scene.terrain.terrain_types]
    torch.testing.assert_close(env.scene.terrain.env_origins,expected)

  def test_completed_returns_belong_to_old_task_before_partial_reset(self):
    env,term=self.make_term()
    term(env,torch.arange(4))
    term.env_task_ids[:]=0
    old_levels=env.scene.terrain.terrain_levels.clone()
    term.sampler.probabilities[:]=0;term.sampler.probabilities[1]=1
    env.common_step_counter=1000;env.episode_length_buf[:]=1000
    env.reward_manager._episode_sums['tracking'][:]=5
    env.reward_manager._episode_sums['posture'][:]=2
    env.metrics_manager._episode_sums['commanded_distance'][:]=4
    env.metrics_manager._episode_sums['tracking_progress'][:]=2
    env.termination_manager.terminated[1]=True
    term(env,torch.tensor([1,3]))
    self.assertEqual(term.sampler.counts[0],2)
    self.assertEqual(term.sampler.window_returns[0],14.)
    self.assertEqual(term.failed_counts[0],1)
    self.assertEqual(term.sampler.counts[1],0)
    self.assertEqual(term.env_task_ids.tolist(),[0,1,0,1])
    torch.testing.assert_close(env.scene.terrain.terrain_levels[[0,2]],old_levels[[0,2]])
    self.assertEqual(env.reward_manager._episode_sums['tracking'][1],5.)

  def test_constructor_does_not_require_metrics_manager_before_it_is_built(self):
    env,term=self.make_term()
    del env.metrics_manager
    new=LPTerrainCurriculum(env.cfg.curriculum['terrain_sampling'],env)
    new(env,torch.arange(4))
    self.assertEqual(new.sampler.counts.sum(),0)

  def test_randomized_initial_horizons_do_not_create_fake_learning_progress(self):
    env,term=self.make_term();term(env,torch.arange(4))
    env.common_step_counter=1000;env.episode_length_buf[:]=1000
    env.metrics_manager._step_count=torch.tensor([500,1000,10,1000])
    env.reward_manager._episode_sums['tracking'][:]=2.
    term(env,torch.arange(4))
    self.assertEqual(term.sampler.counts.sum(),2)
    self.assertEqual(term.skipped_partial_episodes,2)

  def test_manual_mid_episode_reset_is_not_counted_as_completed_return(self):
    env,term=self.make_term();term(env,torch.arange(4))
    env.common_step_counter=500;env.episode_length_buf[:]=500
    env.termination_manager.time_outs=torch.zeros(4,dtype=torch.bool)
    term(env,torch.arange(4))
    self.assertEqual(term.sampler.counts.sum(),0)

  def test_checkpoint_restores_sampler_and_next_tile_choices(self):
    env,term=self.make_term();term(env,torch.arange(4))
    term.sampler.observe(np.array([0,0,1,1]),np.array([3.,4.,5.,6.]));term.sampler.update()
    state=term.state_dict()
    other,restored=self.make_term();self.assertTrue(restored.load_state_dict(state))
    term(env,torch.arange(4));restored(other,torch.arange(4))
    torch.testing.assert_close(term.env_task_ids,restored.env_task_ids)
    torch.testing.assert_close(env.scene.terrain.env_origins,other.scene.terrain.env_origins)

  def test_reward_change_discards_incomparable_statistics(self):
    env,term=self.make_term()
    term.sampler.observe(np.array([0,0]),np.array([3.,4.]))
    cfg=lgdog1_him_standard_env_cfg();cfg.rewards['pose'].weight+=.1
    _,restored=self.make_term(cfg=cfg)
    with self.assertWarnsRegex(UserWarning,'configuration changed'):
      self.assertFalse(restored.load_state_dict(term.state_dict()))
    self.assertEqual(restored.sampler.counts.sum(),0)

  def test_status_file_contains_safe_task_distribution(self):
    with tempfile.TemporaryDirectory() as td:
      p=Path(td)/'curriculum/terrain_sampling.json'
      _,term=self.make_term(status_path=p)
      data=json.loads(p.read_text(encoding='utf-8'))
      self.assertEqual(data['mode'],'lp_acrl')
      self.assertAlmostEqual(sum(t['probability'] for t in data['tasks']),1.)
      self.assertIsNone(data['tasks'][0]['mean_return'])
      self.assertEqual(data['signature'],term.signature)

  def test_shared_runner_saves_and_loads_sampler_state(self):
    from src.tasks.velocity.rl.runner import PPOHIMRunner, MjlabOnPolicyRunner
    env,term=self.make_term()
    manager=NS(active_terms=['terrain_sampling'],get_term_cfg=lambda _:NS(func=term))
    env.curriculum_manager=manager
    runner=PPOHIMRunner.__new__(PPOHIMRunner)
    runner.env=NS(unwrapped=env,reset=Mock())
    runner.logger=NS(logger_type='tensorboard')
    runner.export_policy_to_onnx=Mock();runner.export_policy_to_jit=Mock()
    with patch.object(MjlabOnPolicyRunner,'save') as parent_save:
      runner.save('model_100.pt')
    saved=parent_save.call_args.args[1]
    self.assertIn('terrain_task_curriculum',saved)
    with patch.object(MjlabOnPolicyRunner,'load',return_value=saved):
      runner.load('model_100.pt')
    runner.env.reset.assert_called_once()

  def test_train_entry_uses_fixed_cli_ranges_in_automatic_mode(self):
    from scripts import train
    class StopBeforePhysics(Exception):pass
    for mode,expected in [('lp_acrl',(-.3,.5)),('manual',(-.15,.25))]:
      args=replace(train.TrainConfig.from_task('LgDog1-Rough-Blind-HIM-Standard'),
                   terrain_curriculum=mode,velocity_min_abs=.3,velocity_max_abs=.5)
      with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}), \
           patch.object(train,'configure_torch_backends'), \
           patch.object(train,'ManagerBasedRlEnv',side_effect=StopBeforePhysics):
        with self.assertRaises(StopBeforePhysics):
          train.run_train('LgDog1-Rough-Blind-HIM-Standard',args,Path(td))
      self.assertEqual(args.env.commands['twist'].ranges.lin_vel_x,expected)


if __name__=='__main__':unittest.main()
