"""Experimental skill rewards must be finite, spatially valid and replayable."""
import unittest,copy,math
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np,torch,mujoco
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.config.mgdp_camera import D435I20_TASK
from src.config.mgdp_skill_training import configure_skill_training,configure_action_scale,restore_skill_for_play,skill_contract,SkillStonesTerrainCfg
from src.tasks.velocity.mdp.skill_training import WallSkillHints,StoneTouchdown,HeadingProgress

class SkillTests(unittest.TestCase):
  def test_heading_hint_only_pays_new_improvement(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    env,xy,_,fail=WallTrainingTests().fake()
    cmd=env.command_manager.get_term('twist');cmd.heading_error=torch.ones(6)
    cmd.waypoint_index=torch.ones(6,dtype=torch.long)
    env.scene.terrain.terrain_types.fill_(9)
    term=HeadingProgress(NS(),env);self.assertEqual(term(env).sum(),0)
    cmd.heading_error.fill_(.7);self.assertAlmostEqual((term(env)*env.step_dt)[0].item(),.3,places=5)
    cmd.heading_error.fill_(1.);self.assertEqual(term(env).sum(),0)
    cmd.heading_error.fill_(.7);self.assertEqual(term(env).sum(),0)
    cmd.heading_error.fill_(.5);fail[:]=True;self.assertEqual(term(env).sum(),0)
    term.reset(torch.tensor([0]));self.assertEqual(term.waypoint[0],-1)
  def test_workbench_launches_explicit_experiment(self):
    import tempfile
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);model=root/'model_20000.pt';model.write_bytes(b'fixture')
      argv=build_command(root,'python',{'mode':'train','task':'race_mgdp_wall_d435i20',
        'warm_start':True,'checkpoint':str(model),'mgdp_skill_training':True,'mgdp_action_scale':.5})
      self.assertIn('--mgdp-skill-training=True',argv)
      self.assertIn('--mgdp-action-scale=0.5',argv)
      self.assertNotIn('--agent.resume=True',argv)
  def test_opt_in_defaults_and_replay_contract(self):
    c=load_env_cfg(D435I20_TASK);self.assertIsNone(skill_contract(c));configure_skill_training(c);configure_action_scale(c,.5)
    self.assertEqual(c.actions['joint_pos'].scale,.5)
    self.assertEqual(c.rewards['wall_waiting'].weight,-.5)
    self.assertEqual(c.terminations['no_passage_progress'].params['timeout_s'],6.)
    p=load_env_cfg(D435I20_TASK,play=True);restore_skill_for_play(p,{'infos':{'mgdp_skill_training':skill_contract(c)}})
    self.assertFalse(p.commands['twist'].slalom_use_lateral)  # 2026-11-19 起改为转弯模式
    self.assertEqual(p.commands['twist'].terrain_level_up_streak,{'slalom_brick':1})
    self.assertEqual(p.scene.terrain.terrain_generator.sub_terrains['stepping_stones'].counts,(2,3,4,5,6,6,6,6))
    for scale in (.51,float('nan'),True):
      with self.assertRaises(ValueError):configure_action_scale(c,scale)

  def test_short_stones_compile_and_competition_rows_remain_exact(self):
    from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import StrictStonesTerrainCfg
    for stage in range(8):
      built=[]
      for terrain in (StrictStonesTerrainCfg(size=(12.,8.)),SkillStonesTerrainCfg(size=(12.,8.))):
        s=mujoco.MjSpec();s.worldbody.add_body(name='terrain');terrain.function((stage+.5)/8,s,np.random.default_rng(42));m=s.compile()
        built.append({m.geom(i).name:np.r_[m.geom_pos[i],m.geom_size[i]] for i in range(m.ngeom)})
      if stage>=4:
        self.assertEqual(set(built[0]),set(built[1]))
        for name in built[0]:np.testing.assert_allclose(built[0][name],built[1][name])
      else:self.assertLess(len(built[1]),len(built[0]))

  def test_wall_hints_are_bounded_no_oscillation_farming(self):
    from tests.test_mgdp_wall_training import WallTrainingTests
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    env,xy,progress,fail=WallTrainingTests().fake();term=WallSkillHints(NS(),env);asset=SceneEntityCfg('robot',site_ids=[0,1,2,3])
    term(env,asset);feet=env.scene['robot'].data.site_pos_w
    xy[:,0]=1.2;feet[:,:,0]=1.1;feet[:,:,2]=.07
    first=term(env,asset)*env.step_dt;self.assertGreater(first[0],0)
    self.assertEqual((term(env,asset)*env.step_dt)[0],0)
    feet[:,:,2]=0;term(env,asset);feet[:,:,2]=.07
    self.assertEqual((term(env,asset)*env.step_dt)[0],0)
    xy[:,0]=1.5;feet[:,:,0]=1.5;feet[:,:,2]=0
    self.assertEqual((term(env,asset)*env.step_dt)[0],1.)
    self.assertEqual((term(env,asset)*env.step_dt)[0],0)
    term.reset(torch.tensor([0]));self.assertFalse(term.ready[0])

  def test_stone_touchdown_requires_top_correct_row_and_new_contact(self):
    from tests.test_mgdp_tire_guidance import fixture
    env,_,land,clear,_,fail=fixture();configure_skill_training(env.cfg)
    env.scene.terrain.cfg.terrain_generator=env.cfg.scene.terrain.terrain_generator
    env.scene.terrain.cfg.terrain_generator.num_cols=10
    env.scene.terrain.terrain_types=torch.tensor([1,0]);env.scene.terrain.terrain_levels=torch.zeros(2,dtype=torch.long)
    tc=env.cfg.rewards['stone_touchdown'];term=StoneTouchdown(tc,env)
    def call():return term(env,**tc.params)*env.step_dt*tc.weight
    call();data=env.scene['stone_support_contact'].data
    slot=2 # fixture order RL,FR,RR,FL; two slots each
    def touchdown(col=0,z=.02,dy=0.):
      data.found_history[:,slot,0]=1;data.force_history[:,slot,0,2]=2
      data.normal_history[:,slot,0,2]=1
      data.pos_history[:,slot,0,:2]=env.scene.terrain.env_origins[:,:2]+term.centres[0,col]+torch.tensor([0.,dy])
      data.pos_history[:,slot,0,2]=z
    touchdown();self.assertEqual(call()[0],.25);self.assertEqual(call()[0],0.)
    clear();call();touchdown();self.assertEqual(call()[0],0.)
    for change in ({'z':0.},{'dy':.4},{'col':3}):
      clear();call();touchdown(**change);self.assertEqual(call().sum(),0.)

  def test_lateral_command_route_and_body_clearance(self):
    from tests.test_mgdp_course_v3 import CourseV3Tests
    from src.tasks.velocity.mdp.course_v3 import box_corners,box_projection_distance
    from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommand
    env,cmd,robot=CourseV3Tests().command_env();configure_skill_training(env.cfg)
    env.cfg.commands['twist'].slalom_use_lateral=True  # 本测试专门覆盖横移模式
    cmd=BodyAwarePassageCommand(env.cfg.commands['twist'],env)
    env.command_manager=NS(get_term=lambda _:cmd)
    cmd._resample_command(torch.arange(10));cmd.speed.fill_(.6)
    poses=[];lateral_peak=0.
    for _ in range(6000):
      cmd._update_command();heading=robot.data.heading_w.clone();vx,vy=cmd.command[:,0],cmd.command[:,1]
      lateral_peak=max(lateral_peak,abs(vy[9].item()))
      world=torch.stack((vx*heading.cos()-vy*heading.sin(),vx*heading.sin()+vy*heading.cos()),-1)
      robot.data.root_link_pos_w[:,:2]+=world*env.step_dt
      robot.data.heading_w+=cmd.command[:,2]*env.step_dt
      poses.append(torch.cat((robot.data.root_link_pos_w[9,:2],robot.data.heading_w[9,None])))
      if cmd.route_success()[9]:break
    self.assertTrue(cmd.route_success()[9]);states=torch.stack(poses);yaw=states[:,2];center=torch.zeros(len(states),3)
    self.assertGreater(lateral_peak,.2)
    center[:,:2]=states[:,:2]-.202*torch.stack((yaw.cos(),yaw.sin()),-1)
    quat=torch.stack(((yaw/2).cos(),torch.zeros_like(yaw),torch.zeros_like(yaw),(yaw/2).sin()),-1)
    poles=torch.tensor([(1.4+i*.9,-.6) for i in range(4)]+[(4.1,.3),(4.1,1.2)])
    distance=box_projection_distance(box_corners(center,quat,torch.tensor([.202,.133,.05705])),poles.expand(len(states),-1,-1))-.05
    self.assertGreater(distance.min().item(),.03,msg=str((states[distance.min(dim=1).values.argmin()].tolist(),distance.min(dim=0).values.tolist())))

if __name__=='__main__':unittest.main()
