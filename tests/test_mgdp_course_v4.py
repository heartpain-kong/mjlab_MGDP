"""Compiled geometry checks against the bundled competition model."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch,mujoco
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import (
  UnifiedObstacleTerrainCfg, UnifiedGapTerrainCfg, UnifiedWallTerrainCfg,
  race_reference, vector,
)
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg
from src.tasks.velocity.mdp.course_v3 import SlalomBodyProjection

TASK='LgDog1-Race-MGDP-WallTraining'


def build(cfg,d):
  spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
  result=cfg.function(d,spec,np.random.default_rng(42))
  model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  return spec,result,model,data


def ray_height(model,data,x,y):
  return 1.-mujoco.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,1,-1,None)


class CourseV4Tests(unittest.TestCase):
  def test_sixth_stage_bridge_and_wall_match_competition_and_hard_stages_exceed_it(self):
    ref=race_reference();bridge=ref.find(".//geom[@name='balance_beam_surface']")
    wall=ref.find(".//geom[@name='wall_steel_cap']")
    wall_top=vector(wall,'pos','')[2]+vector(wall,'size','')[2]
    for d in ((5+.001)/8,(5+.999)/8):
      _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='balance_beam',size=(12.,8.)),d)
      beams=[m.geom(i) for i in range(m.ngeom) if 'beam_lane_' in m.geom(i).name]
      np.testing.assert_allclose([g.size for g in beams],np.tile(vector(bridge,'size',''),(len(beams),1)))
      _,_,m,_=build(UnifiedWallTerrainCfg(size=(12.,8.)),d)
      walls=[m.geom(i) for i in range(m.ngeom) if 'high_wall_' in m.geom(i).name and 'floor' not in m.geom(i).name]
      self.assertEqual(len(walls),27)
      np.testing.assert_allclose([g.size for g in walls],np.tile([.04,.4,wall_top/2],(27,1)))
    widths=[];heights=[]
    for row in (5,6,7):
      _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='balance_beam',size=(12.,8.)),(row+.5)/8)
      g=next(m.geom(i) for i in range(m.ngeom) if 'beam_lane_' in m.geom(i).name)
      widths.append(2*g.size[1]);heights.append(2*g.size[2])
    self.assertTrue(widths[0]>widths[1]>widths[2]);self.assertTrue(heights[0]<heights[1]<heights[2])
    self.assertEqual(UnifiedWallTerrainCfg().heights[5:],(.30,.35,.40))

  def test_both_gap_modules_have_reference_steps_platform_gap_and_ramp_at_stage_six(self):
    ref=race_reference();platform=ref.find(".//geom[@name='ramp_platform']")
    other=ref.find(".//geom[@name='stairs_platform']")
    height=2*vector(platform,'size','')[2]
    gap=vector(other,'pos','')[0]-2*vector(platform,'size','')[0]
    tread=2*vector(ref.find(".//geom[@name='stair_low']"),'size','')[0]
    for d in ((5+.001)/8,(5+.999)/8):
      spec,_,m,data=build(UnifiedGapTerrainCfg(size=(12.,8.)),d)
      for part in ('first_platform0_0','first_platform1_0','second_platform0_0','second_platform1_0'):
        g=next(m.geom(i) for i in range(m.ngeom) if m.geom(i).name.endswith(part))
        np.testing.assert_allclose(g.size,[.4,.4,height/2])
      for prefix in ('first_rise_0_','second_tread_0_'):
        gs=[m.geom(i) for i in range(m.ngeom) if prefix in m.geom(i).name]
        np.testing.assert_allclose([2*g.size[0] for g in gs],tread)
      gs=[m.geom(i) for i in range(m.ngeom) if 'first_rise_0_' in m.geom(i).name]
      np.testing.assert_allclose([g.pos[2]+g.size[2] for g in gs],np.arange(1,4)*.12)
      for prefix in ('first','second'):
        a=next(m.geom(i) for i in range(m.ngeom) if m.geom(i).name.endswith(prefix+'_platform0_0'))
        b=next(m.geom(i) for i in range(m.ngeom) if m.geom(i).name.endswith(prefix+'_platform1_0'))
        self.assertAlmostEqual(b.pos[0]-a.pos[0]-.8,gap)
        self.assertAlmostEqual(ray_height(m,data,(a.pos[0]+b.pos[0])/2,4.),0.)
      mesh=next(mesh for mesh in spec.meshes if 'second_upslope' in mesh.name)
      v=np.array(mesh.uservert).reshape(-1,3)
      rv=vector(ref.find(".//mesh[@name='ramp_wedge']"),'vertex','').reshape(-1,3)
      np.testing.assert_allclose(np.ptp(v,axis=0),np.ptp(rv,axis=0)[[1,0,2]],atol=2e-6)
      self.assertAlmostEqual(ray_height(m,data,11.4,4.),0.)
    for row in (6,7):
      h,_,g,up,down=UnifiedGapTerrainCfg().gap_parameters((row+.5)/8)
      self.assertGreater(h,height);self.assertGreater(g,gap);self.assertLess(up,tread);self.assertLess(down,tread)

  def test_clearance_frame_uses_reference_underside_and_then_lowers_it(self):
    ref=race_reference().find(".//geom[@name='frame_crossbar_0']")
    underside=vector(ref,'pos','')[2]-vector(ref,'size','')[2];levels=[]
    for row in (5,6,7):
      _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='clearance_frame',size=(12.,8.)),(row+.5)/8)
      bars=[m.geom(i) for i in range(m.ngeom) if m.geom(i).type==mujoco.mjtGeom.mjGEOM_BOX and abs(m.geom(i).size[2]-.01)<1e-8]
      self.assertEqual(len(bars),7*4*5)
      bottoms=[g.pos[2]-g.size[2] for g in bars];np.testing.assert_allclose(bottoms,bottoms[0]);levels.append(bottoms[0])
    self.assertAlmostEqual(levels[0],underside);self.assertTrue(levels[0]>levels[1]>levels[2])

  def test_tire_reference_mesh_and_real_hole_are_preserved(self):
    ref=race_reference();ref_vertices=vector(ref.find(".//mesh[@name='tire_sector']"),'vertex','').reshape(-1,3)
    for row,height in ((5,.165),(6,.19),(7,.22)):
      spec,_,m,data=build(UnifiedObstacleTerrainCfg(kind='tire_array',size=(12.,8.)),(row+.5)/8)
      vertices=np.array(spec.meshes[0].uservert).reshape(-1,3)
      np.testing.assert_allclose(vertices[:,:2],ref_vertices[:,:2])
      np.testing.assert_allclose(vertices[:,2],ref_vertices[:,2]*height/.165)
      self.assertEqual(m.ngeom-1,12*4*48)
      # Bottom and hole are genuine geometry, rather than a filled cylinder.
      self.assertAlmostEqual(ray_height(m,data,2.925,3.725),0.,places=6)
      self.assertAlmostEqual(ray_height(m,data,2.925+.22,3.725),height,places=6)

  def test_competition_tires_are_official_2x4_and_training_field_cannot_be_bypassed(self):
    """比赛验收=官方 2×4；训练仍铺满整格，机器人无法从阵侧绕过去。"""
    _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='tire_array',size=(8.,8.),competition_tires=True),.6875)
    tires=[i for i in range(m.ngeom) if 'v4_tires' in m.geom(i).name]
    self.assertEqual(len(tires),8*48)
    centers={}
    for i in tires:
      parts=m.geom(i).name.split('_');key=(parts[-3],parts[-2])
      centers.setdefault(key,[]).append(m.geom_pos[i][:2])
    self.assertEqual(len(centers),8)
    xy=np.array([np.mean(v,axis=0) for v in centers.values()])
    self.assertAlmostEqual(float(xy[:,1].max()-xy[:,1].min()),.55,places=6)  # 短边 1100 mm
    self.assertAlmostEqual(float(xy[:,0].max()-xy[:,0].min()),1.65,places=6) # 长边 2200 mm
    for i in tires:
      self.assertAlmostEqual(float(m.geom_pos[i][2]),.0825,places=6)  # 高 165 mm
    _,_,training,_=build(UnifiedObstacleTerrainCfg(kind='tire_array',size=(8.,8.)),.6875)
    ty=[float(training.geom_pos[i][1]) for i in range(training.ngeom) if 'v4_tires' in training.geom(i).name]
    self.assertGreater(max(ty)-min(ty),6.)  # 铺满整格，无侧向绕行空间

  def test_full_course_eval_switches_tires_to_competition_layout(self):
    from src.config.mgdp_skill_training import configure_skill_training
    from src.config.mgdp_stages import configure_full_course_eval
    import src.tasks as tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    cfg=load_env_cfg('LgDog1-Race-MGDP-WallTraining-D435i20',play=True)
    configure_skill_training(cfg)
    cfg.commands['twist'].skill_training_profile={'version':5}
    configure_full_course_eval(cfg,6)
    self.assertTrue(cfg.scene.terrain.terrain_generator.sub_terrains['tire_array'].competition_tires)

  def test_pit_stage_six_copies_every_reference_box_and_hardens_vertical_relief(self):
    ref=race_reference().find(".//body[@name='brick_bark_pit']")
    boxes=[g for g in ref.findall('geom') if g.get('contype','1')!='0']
    for row,scale in ((5,1.),(6,1.2),(7,1.4)):
      _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='brick_pit',size=(12.,8.)),(row+.5)/8)
      self.assertEqual(m.ngeom,len(boxes)+1)
      for i,g in enumerate(boxes,1):
        expected_pos=vector(g,'pos','0 0 0')+np.array([2.5,3.,0.]);expected_pos[2]*=scale
        expected_size=vector(g,'size','');expected_size[2]*=scale
        np.testing.assert_allclose(m.geom_pos[i],expected_pos,atol=1e-9)
        np.testing.assert_allclose(m.geom_size[i],expected_size,atol=1e-9)

  def test_posts_keep_reference_centers_and_increase_radius_and_height_after_six(self):
    reference=race_reference().find(".//body[@name='slalom']").findall('geom')[:6]
    for row,radius,height in ((5,.05,.10),(6,.0575,.13),(7,.065,.16)):
      _,_,m,_=build(UnifiedObstacleTerrainCfg(kind='slalom_brick',size=(12.,8.)),(row+.5)/8)
      posts=[m.geom(i) for i in range(m.ngeom) if 'slalom_pole_' in m.geom(i).name]
      self.assertEqual(len(posts),6)
      np.testing.assert_allclose([g.pos[:2] for g in posts],
        [vector(g,'pos','')[:2]+np.array([2.4,3.4]) for g in reference])
      np.testing.assert_allclose([g.size[0] for g in posts],radius)
      np.testing.assert_allclose([2*g.size[1] for g in posts],height)
      self.assertEqual(sum('lesson_floor' in m.geom(i).name for i in range(m.ngeom)),1)

  def test_projection_termination_uses_current_stage_radius(self):
    cfg=load_env_cfg(TASK);g=cfg.scene.terrain.terrain_generator;g.num_cols=10
    class Scene(dict):pass
    position=torch.tensor([[[1.4,-.6+.133+.09,0.]]]*2)
    robot=NS(data=NS(geom_pos_w=position,geom_quat_w=torch.tensor([[[1.,0.,0.,0.]]]*2)))
    scene=Scene(robot=robot);scene.terrain=NS(cfg=NS(terrain_generator=g),terrain_types=torch.tensor([9,9]),
      terrain_levels=torch.tensor([5,7]),env_origins=torch.zeros(2,3))
    model=NS(geom=lambda _:NS(size=[.202,.133,.05705]))
    env=NS(scene=scene,device='cpu',num_envs=2,sim=NS(mj_model=model))
    term=SlalomBodyProjection(NS(),env)
    torch.testing.assert_close(term(env,NS(name='robot',geom_ids=[0])),torch.tensor([False,True]))

  def test_synthetic_reference_and_flat_have_explicit_separate_meaning(self):
    for kind in ('random_hills','mjlab_rough'):
      _,_,baseline,_=build(PassageTerrainCfg(kind=kind,size=(12.,8.)),1.)
      scales=[]
      for row in (5,6,7):
        _,_,m,_=build(UnifiedObstacleTerrainCfg(kind=kind,size=(12.,8.)),(row+.5)/8)
        np.testing.assert_allclose(m.hfield_data,baseline.hfield_data)
        scales.append(m.hfield_size[0,2])
      np.testing.assert_allclose(scales,baseline.hfield_size[0,2]*np.array([1.,1.25,1.5]))
    for row in (0,5,7):
      _,_,m,data=build(UnifiedObstacleTerrainCfg(kind='start_flat',size=(12.,8.)),(row+.5)/8)
      self.assertAlmostEqual(ray_height(m,data,4.,4.),0.)


if __name__=='__main__':unittest.main()
