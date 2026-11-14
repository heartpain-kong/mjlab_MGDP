from types import SimpleNamespace as NS
from pathlib import Path
import xml.etree.ElementTree as ET
import math,unittest
import numpy as np
import torch,mujoco
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import ContactMatch
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.mdp.course_v3 import (
 SAFE_SLALOM_ROUTE,BodyAwarePassageCommand,StoneGroundContact,
 PositionHistoryContactSensorCfg,box_corners,box_projection_distance,
)
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import (
 CombinedGapTerrainCfg,StrictStonesTerrainCfg,WallSkillTerrainCfg,
)
TASK='LgDog1-Race-MGDP-WallTraining'


def compile(cfg,d):
 spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
 cfg.function(d,spec,np.random.default_rng(42));m=spec.compile();p=mujoco.MjData(m);mujoco.mj_forward(m,p)
 return m,p


class CourseV3Tests(unittest.TestCase):
  def test_combined_gap_has_both_sequences_in_one_cell_at_equal_difficulty(self):
    for row in range(8):
      m,p=compile(CombinedGapTerrainCfg(size=(12.,8.)),(row+.37)/8)
      names=[m.geom(i).name for i in range(m.ngeom)]
      rises=[i for i,n in enumerate(names) if 'first_rise_0_' in n]
      down=[i for i,n in enumerate(names) if 'second_tread_0_' in n]
      self.assertEqual(len(rises),3);self.assertEqual(len(down),2)
      self.assertEqual(sum('downslope' in n for n in names),9)
      self.assertEqual(sum('upslope' in n for n in names),9)
      h=.06+.30*row/7;gap=.04+.32*row/7;length=.8+.189092*row/7
      np.testing.assert_allclose(m.geom_pos[rises,2]+m.geom_size[rises,2],h*np.arange(1,4)/3)
      np.testing.assert_allclose(m.geom_pos[down,2]+m.geom_size[down,2],[2*h/3,h/3])
      platforms=[]
      for part in ('first_platform0_0','first_platform1_0','second_platform0_0','second_platform1_0'):
        ids=[i for i,n in enumerate(names) if n.endswith(part)];self.assertEqual(len(ids),1);platforms+=ids
      np.testing.assert_allclose(m.geom_pos[platforms,2]+m.geom_size[platforms,2],h)
      for a,b in (platforms[:2],platforms[2:]):
        left=m.geom_pos[a,0]+.4;right=m.geom_pos[b,0]-.4
        self.assertAlmostEqual(right-left,gap,places=7)
        ray=mujoco.mj_ray(m,p,np.array([(left+right)/2,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
        self.assertAlmostEqual(ray,1.,places=6)
      first_end=2+length+1.6+gap+length;second=first_end+.8
      for x in (2+length+1.6+gap+length/2,second+length/2):
        ray=mujoco.mj_ray(m,p,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
        self.assertAlmostEqual(1-ray,h/2,places=5)
      for x in (1.,first_end+.4,11.4):
        ray=mujoco.mj_ray(m,p,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
        self.assertAlmostEqual(ray,1.,places=6)

  def test_combined_layout_rejects_old_eight_meter_cell(self):
    with self.assertRaisesRegex(ValueError,'12x6'):compile(CombinedGapTerrainCfg(size=(8.,8.)),1.)

  def test_wall_height_bank_matches_user_correction(self):
    self.assertEqual(WallSkillTerrainCfg().heights,(.1,.12,.15,.18,.21,.25,.30,.35))
    m,_=compile(WallSkillTerrainCfg(size=(12.,8.)),1.)
    ids=[i for i in range(m.ngeom) if 'high_wall_' in m.geom(i).name and 'floor' not in m.geom(i).name]
    np.testing.assert_allclose(m.geom_pos[ids,2]+m.geom_size[ids,2],.35)

  def test_stones_are_staggered_from_the_very_first_stage(self):
    for d in (0.,1.):
      m,_=compile(StrictStonesTerrainCfg(size=(12.,8.)),d)
      left=[m.geom(i) for i in range(m.ngeom) if 'stone_0_0_' in m.geom(i).name]
      right=[m.geom(i) for i in range(m.ngeom) if 'stone_0_1_' in m.geom(i).name]
      self.assertEqual(len(left),6);self.assertEqual(len(right),6)
      np.testing.assert_allclose([b.pos[0]-a.pos[0] for a,b in zip(left,right)],.3)
      self.assertAlmostEqual(left[0].pos[0],2.4);self.assertAlmostEqual(right[0].pos[0],2.7)

  def test_sixth_stone_stage_matches_bundled_competition_dimensions_without_jitter(self):
    root=Path(__file__).resolve().parents[1]
    reference=ET.parse(root/'src/assets/terrains/race_obstacle_2027.xml').find(".//body[@name='stepping_stones']")
    def values(name,attribute):
      return np.fromstring(reference.find(f"geom[@name='{name}']").get(attribute),sep=' ')
    size=values('stone_0_0','size');first=values('stone_0_0','pos')
    next_stone=values('stone_0_1','pos');other_row=values('stone_1_0','pos')
    for d in (5/8,(5+.01)/8,(5+.50)/8,(5+.999999)/8):
      m,_=compile(StrictStonesTerrainCfg(size=(12.,8.)),d)
      stones=[m.geom(i) for i in range(m.ngeom) if '_stone_' in m.geom(i).name]
      self.assertGreater(len(stones),0)
      np.testing.assert_allclose([g.size for g in stones],np.tile(size,(len(stones),1)),atol=1e-12)
      np.testing.assert_allclose([g.pos[2]+g.size[2] for g in stones],first[2]+size[2],atol=1e-12)
      a=next(g for g in stones if g.name.endswith('stone_0_0_0'))
      b=next(g for g in stones if g.name.endswith('stone_0_0_1'))
      c=next(g for g in stones if g.name.endswith('stone_0_1_0'))
      np.testing.assert_allclose(b.pos-a.pos,next_stone-first,atol=1e-12)
      np.testing.assert_allclose(c.pos-a.pos,other_row-first,atol=1e-12)
      pad=next(m.geom(i) for i in range(m.ngeom) if m.geom(i).name.endswith('stones_entry_0'))
      np.testing.assert_allclose(pad.size,values('stones_entry_platform','size'),atol=1e-12)
    # Earlier rows remain easier than the competition reference.
    m,_=compile(StrictStonesTerrainCfg(size=(12.,8.)),.5)
    g=next(m.geom(i) for i in range(m.ngeom) if m.geom(i).name.endswith('stone_0_0_0'))
    self.assertAlmostEqual(2*g.size[0],.22);self.assertAlmostEqual(g.pos[2]+g.size[2],.08)

  def test_stone_stages_progress_through_competition_then_raise_the_ceiling(self):
    widths=(.36,.32,.28,.25,.22,.20,.18,.16);heights=(.02,.03,.04,.06,.08,.10,.12,.15)
    for stage,(width,height) in enumerate(zip(widths,heights)):
      for jitter in (.001,.999):
        m,_=compile(StrictStonesTerrainCfg(size=(12.,8.)),(stage+jitter)/8)
        stones=[m.geom(i) for i in range(m.ngeom) if '_stone_' in m.geom(i).name]
        np.testing.assert_allclose([2*g.size[0] for g in stones],width,atol=1e-12)
        np.testing.assert_allclose([g.pos[2]+g.size[2] for g in stones],height,atol=1e-12)
    self.assertTrue(all(a>b for a,b in zip(widths,widths[1:])))
    self.assertTrue(all(a<b for a,b in zip(heights,heights[1:])))
    cfg=StrictStonesTerrainCfg();self.assertEqual(cfg.stone_parameters(1.),(.15,.16))

  def test_ground_rule_uses_actual_loaded_contacts_and_keeps_pads_safe(self):
    n=6;generator=load_env_cfg(TASK).scene.terrain.terrain_generator;generator.num_cols=10
    class Scene(dict):pass
    data=NS(pos_history=torch.zeros(n,1,4,3),normal_history=torch.zeros(n,1,4,3),
      force_history=torch.zeros(n,1,4,3),found_history=torch.zeros(n,1,4))
    origin=torch.tensor([[100.,200.,0.]]).repeat(n,1)
    data.pos_history[:,0,3,0]=origin[:,0]-1+torch.tensor([2.25,3.,3.,1.9,6.,3.])
    data.pos_history[:,:,:,1]=200.
    data.pos_history[1,0,3,2]=.02 # Lowest stone top, not floor.
    data.normal_history[:,0,3,2]=1.;data.force_history[:,0,3,2]=2.;data.found_history[:,0,3]=1.
    data.found_history[2]=0 # Swinging through the same XY is allowed.
    scene=Scene(stone_support_contact=NS(data=data))
    scene.terrain=NS(cfg=NS(terrain_generator=generator),env_origins=origin,terrain_types=torch.tensor([1,1,1,1,1,0]))
    env=NS(scene=scene,device='cpu',num_envs=n);term=StoneGroundContact(NS(),env)
    torch.testing.assert_close(term(env),torch.tensor([True,False,False,False,False,False]))
    # A first-physics-substep contact survives until the control-step check.
    data.normal_history[0,0,3,2]=0.;self.assertFalse(term(env)[0])
    data.normal_history[0,0,3,2]=1.;data.force_history[0,0,3,2]=.1;self.assertFalse(term(env)[0])

  def test_contact_position_history_retains_early_hit_and_resets_partially(self):
    sensor=PositionHistoryContactSensorCfg(name='test',primary=ContactMatch(mode='geom',pattern='foot'),
      fields=('found','force','pos','normal'),history_length=4).build()
    sensor._history_state={k:torch.zeros(2,1,4,3) for k in ('force','pos','normal')}
    sensor._history_state['found']=torch.zeros(2,1,4)
    value=NS(found=torch.ones(2,1),force=torch.ones(2,1,3),pos=torch.ones(2,1,3)*3,
             normal=torch.tensor([[[0.,0.,1.]]]*2))
    sensor._extract_sensor_data=lambda:value
    sensor._update_history();value.found.zero_();value.force.zero_();value.pos.zero_()
    for _ in range(3):sensor._update_history()
    self.assertEqual(sensor.data.found_history[0,0,3].item(),1.)
    self.assertEqual(sensor.data.pos_history[0,0,3,0].item(),3.)
    sensor.reset(torch.tensor([0]))
    self.assertEqual(sensor.data.found_history[0].sum().item(),0.)
    self.assertEqual(sensor.data.found_history[1].sum().item(),1.)

  def test_box_projection_interior_edges_corners_and_pitch(self):
    position=torch.zeros(1,3);quat=torch.tensor([[1.,0.,0.,0.]]);size=torch.tensor([.202,.133,.05705])
    points=torch.tensor([[[0.,0.],[.3,0.],[.302,.233]]])
    result=box_projection_distance(box_corners(position,quat,size),points)
    torch.testing.assert_close(result,torch.tensor([[0.,.098,math.sqrt(.02)]]),atol=1e-6,rtol=0)
    quat=torch.tensor([[math.cos(math.pi/8),0.,math.sin(math.pi/8),0.]])
    result=box_projection_distance(box_corners(position,quat,size),torch.tensor([[[.3,0.]]]))
    self.assertAlmostEqual(result.item(),.3-(.202+.05705)/math.sqrt(2),places=6)

  def command_env(self):
    cfg=load_env_cfg(TASK);g=cfg.scene.terrain.terrain_generator;g.num_cols=10
    n=10;robot=NS(data=NS(root_link_pos_w=torch.zeros(n,3),heading_w=torch.zeros(n)))
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.terrain=NS(cfg=NS(terrain_generator=g),terrain_types=torch.arange(n));scene.env_origins=torch.zeros(n,3)
    env=NS(scene=scene,num_envs=n,device='cpu',step_dt=.02,cfg=cfg)
    command=BodyAwarePassageCommand(cfg.commands['twist'],env);env.command_manager=NS(get_term=lambda _:command)
    command._resample_command(torch.arange(n));command.speed.fill_(.25)
    return env,command,robot

  def test_command_waits_for_heading_and_uses_six_centimeter_waypoint_gate(self):
    env,command,robot=self.command_env();i=9
    command._update_command();self.assertEqual(command.command[i,:2].norm().item(),0.)
    robot.data.heading_w[i]=math.atan2(-.6,.65);command._update_command()
    self.assertGreater(command.command[i,0].item(),0.)
    target=torch.tensor(SAFE_SLALOM_ROUTE[1]);robot.data.root_link_pos_w[i,:2]=target-torch.tensor([.1,0.])
    command._update_command();self.assertEqual(command.waypoint_index[i].item(),1)
    robot.data.root_link_pos_w[i,:2]=target;command._update_command()
    self.assertEqual(command.waypoint_index[i].item(),2)

  def test_ideal_command_follows_all_routes_within_time_and_body_clearance(self):
    env,command,robot=self.command_env();complete=torch.zeros(10,dtype=torch.bool);poses=[]
    for _ in range(int(env.cfg.episode_length_s/env.step_dt)):
      command._update_command();heading=robot.data.heading_w.clone()
      vx,vy=command.command[:,0],command.command[:,1]
      world=torch.stack((vx*heading.cos()-vy*heading.sin(),vx*heading.sin()+vy*heading.cos()),-1)
      robot.data.root_link_pos_w[~complete,:2]+=world[~complete]*env.step_dt
      robot.data.heading_w[~complete]+=command.command[~complete,2]*env.step_dt
      if not complete[9]:poses.append(torch.cat((robot.data.root_link_pos_w[9,:2],robot.data.heading_w[9,None])))
      complete|=passage.passage_success(env)
      self.assertFalse(passage.left_lane(env)[~complete].any())
      if complete.all():break
    self.assertTrue(complete.all())
    states=torch.stack(poses);yaw=states[:,2];center=torch.zeros(len(states),3)
    center[:,:2]=states[:,:2]-.202*torch.stack((yaw.cos(),yaw.sin()),-1)
    quat=torch.stack(((yaw/2).cos(),torch.zeros_like(yaw),torch.zeros_like(yaw),(yaw/2).sin()),-1)
    poles=torch.tensor([(1.4+i*.9,-.6) for i in range(4)]+[(4.1,.3),(4.1,1.2)])
    distance=box_projection_distance(box_corners(center,quat,torch.tensor([.202,.133,.05705])),poles.expand(len(states),-1,-1))-.05
    self.assertGreater(distance.min().item(),.03)

  def test_new_course_contract_and_legacy_isolation(self):
    train=load_env_cfg(TASK);play=load_env_cfg(TASK,play=True);old=load_env_cfg('LgDog1-Race-MGDP')
    self.assertEqual(len(train.scene.terrain.terrain_generator.sub_terrains),10)
    self.assertEqual(train.scene.terrain.terrain_generator.size,(12.,8.))
    self.assertEqual(train.episode_length_s,120.)
    self.assertGreater(train.commands['twist'].resampling_time_range[0],120.)
    for cfg in (train,play):
      self.assertIn('stone_ground_contact',cfg.terminations);self.assertIn('slalom_body_projection',cfg.terminations)
    self.assertNotIn('stone_ground_contact',old.terminations)
    self.assertEqual(old.scene.terrain.terrain_generator.size,(8.,8.))


if __name__=='__main__':unittest.main()
