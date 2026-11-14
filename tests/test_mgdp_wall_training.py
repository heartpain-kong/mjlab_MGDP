"""Behavioral gates for stall resets, auxiliary geometry and reward farming."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from src.tasks.velocity.mdp.wall_training import (
  NoPassageProgress, WallCrossingBonus, WallSwingClearance, WallProgressReward, WallWaitingPenalty,
)
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import WallSkillTerrainCfg, PassageTerrainCfg, GapSkillTerrainCfg
from src.tasks.velocity.mdp import passage


TASK = "LgDog1-Race-MGDP-WallTraining"


class Scene(dict):
  @property
  def terrain(self):
    return self["terrain"]


class WallTrainingTests(unittest.TestCase):
  def fake(self):
    n=6;xy=torch.zeros(n,2);progress=torch.zeros(n);failure=torch.zeros(n,dtype=torch.bool)
    command=NS(command=torch.tensor([[.3,0.,0.]]).repeat(n,1),local_xy=lambda:xy,
      route_state=lambda:(None,None,None,progress),route_success=lambda:progress>=5.3,
      cfg=NS(goal_x=5.3,lane_half_width=.28,ranges=NS(lin_vel_x=(.25,.35))))
    # Keep a real generator to exercise column selection and row stage selection.
    generator=load_env_cfg(TASK).scene.terrain.terrain_generator;generator.num_cols=10
    wall_index=list(generator.sub_terrains).index('high_wall')
    terrain=NS(cfg=NS(terrain_generator=generator),terrain_types=torch.tensor([wall_index]*4+[0,wall_index]),
      terrain_levels=torch.tensor([0,0,7,2,0,0]),env_origins=torch.zeros(n,3),max_terrain_level=8)
    robot=NS(data=NS(site_pos_w=torch.zeros(n,4,3),site_lin_vel_w=torch.zeros(n,4,3)))
    env=NS(num_envs=n,device="cpu",step_dt=.02,scene=Scene(terrain=terrain,robot=robot,
      feet_ground_contact=NS(data=NS(found=torch.zeros(n,4)))),
      command_manager=NS(get_term=lambda name:command),cfg=NS(terminations={"impact":NS(time_out=False)}),
      termination_manager=NS(get_term=lambda name:failure))
    # left_lane uses route_state()[2], not XY, on this directed command.
    command.route_state=lambda:(None,None,xy[:,1].abs(),progress)
    return env,xy,progress,failure

  def test_stall_does_not_refresh_on_oscillation_and_partial_reset_is_clean(self):
    env,xy,progress,_=self.fake();term=NoPassageProgress(NS(),env)
    env.command_manager.get_term('twist').command[2]=0.
    for step in range(201):
      progress[0]=.04 if step%2 else 0.  # Repeated motion without 5cm NEW progress.
      progress[1]=step*.002             # Genuine steady progress must survive.
      result=term(env)
      if step<200:self.assertFalse(result.any())
    self.assertTrue(result[0]);self.assertFalse(result[1]);self.assertFalse(result[2])
    term.reset(torch.tensor([0]));progress[0]=0.
    self.assertFalse(term(env)[0])
    self.assertEqual(term.highwater[0].item(),0.)
    self.assertTrue(term(env)[3])  # Another stalled environment was not reset.

  def test_stall_turning_route_progress_and_goal_do_not_fail(self):
    env,_,progress,_=self.fake();term=NoPassageProgress(NS(),env)
    for step in range(500):
      progress[:] = step*.004
      self.assertFalse(term(env).any())
    progress.fill_(5.3)
    for _ in range(250):self.assertFalse(term(env).any())

  def test_crossing_needs_hind_feet_and_pays_only_once_even_after_backtracking(self):
    env,xy,_,failure=self.fake();term=WallCrossingBonus(NS(),env)
    asset=SceneEntityCfg('robot',site_ids=[0,1,2,3]);feet=env.scene['robot'].data.site_pos_w
    self.assertEqual(term(env,asset).sum().item(),0.)
    xy[:,0]=1.8;feet[:,:,0]=1.8;feet[:,3,0]=1.2
    self.assertEqual(term(env,asset).sum().item(),0.)
    feet[:,3,0]=1.8;failure[1]=True
    result=term(env,asset)*env.step_dt
    torch.testing.assert_close(result,torch.tensor([1.,0.,1.,1.,0.,1.]))
    failure[1]=False
    self.assertEqual((term(env,asset)*env.step_dt)[1].item(),1.)
    xy[:,0]=0;feet[:,:,0]=0;term(env,asset)
    xy[:,0]=1.8;feet[:,:,0]=1.8
    self.assertEqual(term(env,asset).sum().item(),0.)
    # Partial reset may not award old terminal poses at the next episode start.
    term.reset(torch.tensor([0]));self.assertEqual(term(env,asset)[0].item(),0.)
    # Level 0 has only ONE wall; level 7 can earn the two remaining milestones.
    xy[:,0]=4.8;feet[:,:,0]=4.8
    result=term(env,asset)*env.step_dt
    self.assertEqual(result[0].item(),0.);self.assertEqual(result[2].item(),2.)

  def test_crossing_disallows_lane_cut_and_failed_episode(self):
    env,xy,_,failure=self.fake();term=WallCrossingBonus(NS(),env)
    asset=SceneEntityCfg('robot',site_ids=[0,1,2,3]);term(env,asset)
    xy[:,0]=1.8;env.scene['robot'].data.site_pos_w[:,:,0]=1.8
    xy[:,1]=.5
    self.assertEqual(term(env,asset).sum().item(),0.)
    xy[:,1]=0;failure[:]=True
    self.assertEqual(term(env,asset).sum().item(),0.)

  def test_clearance_cost_is_local_to_forward_swing_feet_and_wall_stage(self):
    env,_,_,_=self.fake();term=WallSwingClearance(NS(),env)
    asset=SceneEntityCfg('robot',site_ids=[0,1,2,3]);robot=env.scene['robot'].data
    robot.site_pos_w[:,:,0]=1.2;robot.site_pos_w[:,:,2]=.02;robot.site_lin_vel_w[:,:,0]=.2
    cost=term(env,asset)
    self.assertGreater(cost[0].item(),0.);self.assertEqual(cost[4].item(),0.)
    self.assertGreater(cost[2].item(),cost[0].item())
    env.scene['feet_ground_contact'].data.found[:]=1
    self.assertEqual(term(env,asset).sum().item(),0.)
    env.scene['feet_ground_contact'].data.found[:]=0;robot.site_lin_vel_w[:,:,0]=-.2
    self.assertEqual(term(env,asset).sum().item(),0.)
    robot.site_lin_vel_w[:,:,0]=.2;robot.site_pos_w[:,:,2]=.5
    self.assertEqual(term(env,asset).sum().item(),0.)
    robot.site_pos_w[:,:,2]=0;robot.site_pos_w[:,:,0]=0
    self.assertEqual(term(env,asset).sum().item(),0.)

  def test_auxiliary_wall_stages_and_original_course_geometry(self):
    wall=WallSkillTerrainCfg(size=(8.,8.))
    for stage,(height,count) in enumerate(zip(wall.heights,wall.counts)):
      for jitter in (.01,.99):
        spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
        wall.function((stage+jitter)/8,spec,np.random.default_rng(42));m=spec.compile()
        ids=[i for i in range(m.ngeom) if 'high_wall_' in m.geom(i).name and 'lesson_floor' not in m.geom(i).name]
        self.assertEqual(len(ids),9*count)
        np.testing.assert_allclose(m.geom_size[ids,2]*2,height)
    for d in (0.,1.):
      spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
      PassageTerrainCfg(kind='high_wall',size=(8.,8.)).function(d,spec,np.random.default_rng(42))
      m=spec.compile();ids=[i for i in range(m.ngeom) if 'high_wall_' in m.geom(i).name and 'lesson_floor' not in m.geom(i).name]
      self.assertEqual(len(ids),27);np.testing.assert_allclose(m.geom_size[ids,2]*2,.04+.26*d)

  def test_stationary_wall_has_zero_tracking_income_but_other_terrain_is_unchanged(self):
    from src.tasks.velocity.mdp.rewards import track_linear_velocity
    env,_,progress,_=self.fake();env.scene['robot'].data.root_link_lin_vel_b=torch.zeros(6,3)
    env.command_manager.get_command=lambda _:env.command_manager.get_term('twist').command
    params={'base_func':track_linear_velocity,'std':.5,'command_name':'twist',
            'relative_weight':.5,'relative_std':.5,'min_command_speed':.2}
    cfg=RewardTermCfg(func=WallProgressReward,weight=2.,params=params);term=WallProgressReward(cfg,env)
    old=track_linear_velocity(env,**{k:v for k,v in params.items() if k!='base_func'})
    self.assertGreater(old.min().item(),0.)
    for _ in range(250):
      value=term(env,**params);self.assertEqual(value[:4].sum().item(),0.)
      self.assertEqual(value[4].item(),old[4].item())
    progress[:]=.002;value=term(env,**params)
    torch.testing.assert_close(value,old)

  def test_oscillation_cannot_keep_farming_shaping_and_partial_reset_does_not_pay_spawn(self):
    env,_,progress,_=self.fake();base=lambda _:torch.ones(6)*4
    cfg=RewardTermCfg(func=WallProgressReward,weight=1.,params={'base_func':base})
    term=WallProgressReward(cfg,env);total=torch.zeros(6)
    for step in range(250):
      progress[0]=.04 if step%2 else 0.
      total+=term(env,base_func=base)*env.step_dt
    self.assertLessEqual(total[0].item(),4*.04/.1+1e-6)
    self.assertAlmostEqual(total[4].item(),20.,places=3)
    term.reset(torch.tensor([0]));progress[0]=2.;progress[1]=.002
    value=term(env,base_func=base);self.assertEqual(value[0].item(),0.);self.assertEqual(value[1].item(),4.)

  def test_guard_keeps_negative_values_and_handles_class_based_rewards(self):
    env,_,_,_=self.fake()
    class Base:
      def __init__(self,cfg,env):self.bias=cfg.params['bias'];self.reset_ids=None
      def __call__(self,env,bias):return torch.ones(6)*bias
      def reset(self,env_ids=None):self.reset_ids=env_ids
    cfg=RewardTermCfg(func=WallProgressReward,weight=1.,params={'base_func':Base,'bias':-2.})
    term=WallProgressReward(cfg,env)
    torch.testing.assert_close(term(env,**cfg.params),torch.full((6,),-2.))
    ids=torch.tensor([0]);term.reset(ids);self.assertIs(term.base.reset_ids,ids)
    cfg.weight=-1.;term=WallProgressReward(cfg,env);term.base.bias=2.
    torch.testing.assert_close(term(env,base_func=Base,bias=2.),torch.full((6,),2.))

  def test_waiting_penalty_has_grace_ramp_cap_and_is_wall_specific(self):
    env,xy,_,_=self.fake();xy[:,0]=1.15;env.scene['robot'].data.site_pos_w[:,:,0]=.8
    with self.assertRaisesRegex(ValueError,'nonpositive weight'):
      WallWaitingPenalty(RewardTermCfg(func=WallWaitingPenalty,weight=1.),env)
    env.command_manager.get_term('twist').command[1]=0.;xy[2,0]=0.
    term=WallWaitingPenalty(NS(),env);asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
    values=[]
    for step in range(201):
      value=term(env,asset);values.append(value[0].item())
      self.assertEqual(value[1].item(),0.);self.assertEqual(value[2].item(),0.);self.assertEqual(value[4].item(),0.)
      self.assertLessEqual(value.max().item(),2.)
    self.assertEqual(max(values[:38]),0.);self.assertGreater(values[60],0.);self.assertEqual(values[120],2.)
    term.reset(torch.tensor([0]));value=term(env,asset)
    self.assertEqual(value[0].item(),0.);self.assertEqual(value[3].item(),2.)

  def test_waiting_timer_needs_new_progress_and_stops_after_all_feet_clear_wall(self):
    env,xy,_,_=self.fake();xy[:,0]=1.15;feet=env.scene['robot'].data.site_pos_w;feet[:,:,0]=.8
    term=WallWaitingPenalty(NS(),env);asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
    for step in range(100):
      xy[0,0]=1.15+(.01 if step%2 else 0.);feet[0,:,0]=.8+(.01 if step%2 else 0.)
      value=term(env,asset)
    self.assertGreater(value[0].item(),0.)
    xy[0,0]=1.2;feet[0,:,0]=.85;self.assertEqual(term(env,asset)[0].item(),0.)
    xy[0,0]=1.8;feet[0,:,0]=1.8
    for _ in range(200):self.assertEqual(term(env,asset)[0].item(),0.)

  def test_steady_wall_approach_is_not_treated_as_waiting(self):
    env,xy,_,_=self.fake();term=WallWaitingPenalty(NS(),env)
    asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
    for step in range(200):
      xy[:,0]=.6+step*.002;env.scene['robot'].data.site_pos_w[:,:,0]=xy[:,0,None]-.35
      self.assertEqual(term(env,asset).sum().item(),0.)

  def test_waiting_cost_is_a_rate_scaled_once_by_control_dt(self):
    totals=[]
    for dt in (.02,.01):
      env,xy,_,_=self.fake();env.step_dt=dt;xy[:,0]=1.15
      env.scene['robot'].data.site_pos_w[:,:,0]=.8
      term=WallWaitingPenalty(NS(),env);asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
      term(env,asset);total=0.
      for _ in range(round(4/dt)):total+=term(env,asset)[0].item()*dt
      totals.append(total)
    self.assertAlmostEqual(totals[0],5.,delta=.05)
    self.assertAlmostEqual(totals[0],totals[1],delta=.03)

  def test_training_and_original_evaluation_have_separate_contracts(self):
    train=load_env_cfg(TASK);play=load_env_cfg(TASK,play=True)
    old=load_env_cfg('LgDog1-Race-MGDP',play=True)
    generator=train.scene.terrain.terrain_generator
    self.assertEqual(len(generator.sub_terrains),10)
    self.assertEqual((generator.num_rows,generator.num_cols),(8,20))
    self.assertEqual(generator.size,(12.,8.))
    self.assertNotIn('gap_stairs',play.scene.terrain.terrain_generator.sub_terrains)
    self.assertNotIn('gap_stairs',old.scene.terrain.terrain_generator.sub_terrains)
    self.assertEqual(play.scene.terrain.terrain_generator.sub_terrains['high_wall'].counts,(3,)*8)
    self.assertEqual(generator.sub_terrains['high_wall'].heights[0],.1)
    self.assertEqual(generator.sub_terrains['high_wall'].heights[5],.30)
    self.assertEqual(generator.sub_terrains['high_wall'].heights[-1],.40)
    self.assertNotEqual(play.commands['twist'].waypoints['slalom_brick'],old.commands['twist'].waypoints['slalom_brick'])
    self.assertIn('stone_ground_contact',play.terminations)
    self.assertIn('slalom_body_projection',play.terminations)
    self.assertEqual({k:v.weight for k,v in play.rewards.items() if k!='wall_waiting'},
                     {k:v.weight for k,v in old.rewards.items()})
    self.assertEqual(play.rewards['wall_waiting'].weight,-1.)
    for name in ('track_linear_velocity','track_angular_velocity','pose','foot_gait','feet_air_time'):
      self.assertIs(train.rewards[name].func,WallProgressReward)
      self.assertIsNot(old.rewards[name].func,WallProgressReward)
    self.assertIsInstance(train.scene.terrain.terrain_generator.sub_terrains['high_wall'],WallSkillTerrainCfg)
    self.assertNotIn('no_passage_progress',play.terminations)
    self.assertFalse(train.terminations['no_passage_progress'].time_out)
    self.assertEqual(train.rewards['wall_crossing'].weight,2.)
    self.assertEqual(load_rl_cfg(TASK).algorithm.entropy_coef,0.)

  def test_curriculum_advances_a_wall_environment_only_on_success(self):
    env,_,progress,failure=self.fake();n=env.num_envs
    env.episode_length_buf=torch.full((n,),100);env.reset_buf=torch.ones(n,dtype=torch.bool)
    env.metrics_manager=NS(_step_count=env.episode_length_buf.clone(),
      _episode_sums={'velocity_error_xy_mps':torch.full((n,),10.)})
    terrain=env.scene.terrain
    terrain.update_env_origins=lambda ids,up,down:terrain.terrain_levels.__setitem__(ids,
      (terrain.terrain_levels[ids]+up.long()-down.long()).clamp(0,7))
    course=passage.PassageCurriculum(NS(params={}),env)
    progress[0]=5.3;progress[1]=5.3;failure[1]=True
    course(env,torch.tensor([0,1]))
    self.assertEqual(terrain.terrain_levels[0].item(),1)
    self.assertEqual(terrain.terrain_levels[1].item(),0)

  def test_warm_start_keeps_new_critic_iteration_and_rejects_wrong_encoder(self):
    from src.mgdp_warmstart import warm_start_mgdp_policy
    actor=torch.nn.Linear(2,1);source=torch.nn.Linear(2,1);critic=torch.nn.Linear(2,1)
    old_critic={k:v.clone() for k,v in critic.state_dict().items()}
    env=NS(cfg=NS(observations={'perception':{}}),reset=lambda:None)
    runner=NS(env=NS(unwrapped=env),device='cpu',alg=NS(actor=actor,critic=critic),current_learning_iteration=0)
    with tempfile.TemporaryDirectory() as tmp:
      p=Path(tmp)/'old.pt';torch.save({'actor_state_dict':source.state_dict(),'iter':10300,
        'infos':{'mgdp_perception':{'fingerprint':'original'}}},p)
      with patch('src.mgdp_warmstart.validate_runtime_encoder') as validate:
        result=warm_start_mgdp_policy(runner,p);validate.assert_called_once()
      self.assertEqual(result['source_iteration'],10300)
      self.assertEqual(runner.current_learning_iteration,0)
      for k,v in source.state_dict().items():torch.testing.assert_close(actor.state_dict()[k],v)
      for k,v in old_critic.items():torch.testing.assert_close(critic.state_dict()[k],v)
      with patch('src.mgdp_warmstart.validate_runtime_encoder',side_effect=ValueError('wrong encoder')):
        with self.assertRaisesRegex(ValueError,'wrong encoder'):warm_start_mgdp_policy(runner,p)

  def test_workbench_exposes_new_task_and_checks_stage_layout(self):
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);encoder=root/'logs/mgdp/perception/mgdp_wall_v5.pt'
      encoder.parent.mkdir(parents=True);encoder.write_bytes(b'fixture')
      options={'mode':'train','task':'race_mgdp_wall'}
      command=build_command(root,'python',options)
      self.assertIn(TASK,command);self.assertIn('--terrain-num-rows=8',command)
      self.assertIn('--terrain-num-cols=20',command)
      for bad in ({'terrain_num_rows':9},{'terrain_difficulty_max':.5}):
        with self.assertRaisesRegex(ValueError,'八阶段'):build_command(root,'python',{**options,**bad})
      with self.assertRaisesRegex(ValueError,'地形训练列数'):
        build_command(root,'python',{**options,'terrain_num_cols':9})

  def test_paired_gap_variants_have_equal_row_heights_widths_and_safe_steps(self):
    def build(style,d):
      spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
      GapSkillTerrainCfg(size=(8.,8.),entry_style=style).function(d,spec,np.random.default_rng(42))
      m=spec.compile();data=mujoco.MjData(m);mujoco.mj_forward(m,data)
      return m,data
    for row in range(8):
      # Deliberately different jitter in the two columns of the SAME row.
      ramp,rd=build('ramp',(row+.01)/8);stairs,sd=build('stairs',(row+.99)/8)
      for prefix in ('gap_platform','gap_tread'):
        a=[g for g in range(ramp.ngeom) if prefix in ramp.geom(g).name]
        b=[g for g in range(stairs.ngeom) if prefix in stairs.geom(g).name]
        np.testing.assert_allclose(ramp.geom_pos[a],stairs.geom_pos[b])
        np.testing.assert_allclose(ramp.geom_size[a],stairs.geom_size[b])
      self.assertEqual(stairs.nmesh,0)
      rises=[g for g in range(stairs.ngeom) if 'gap_rise_0_' in stairs.geom(g).name]
      self.assertEqual(len(rises),3)
      height=.06+.30*row/7
      tops=stairs.geom_pos[rises,2]+stairs.geom_size[rises,2]
      np.testing.assert_allclose(tops,[height/3,height*2/3,height])
      for g,top in zip(rises,tops):
        ray=mujoco.mj_ray(stairs,sd,np.array([stairs.geom_pos[g,0],4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
        self.assertAlmostEqual(1-ray,top,places=6)
      for model,physics in ((ramp,rd),(stairs,sd)):
        platforms=[g for g in range(model.ngeom) if 'gap_platform' in model.geom(g).name and model.geom_pos[g,1]==4.]
        a,b=platforms;left=model.geom_pos[a,0]+model.geom_size[a,0];right=model.geom_pos[b,0]-model.geom_size[b,0]
        self.assertAlmostEqual(right-left,.04+.32*row/7,places=7)
        ray=mujoco.mj_ray(model,physics,np.array([(left+right)/2,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
        self.assertAlmostEqual(ray,1.,places=6)
        for x in (1.,6.3):
          ray=mujoco.mj_ray(model,physics,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
          self.assertAlmostEqual(ray,1.,places=6)


if __name__=='__main__':unittest.main()
