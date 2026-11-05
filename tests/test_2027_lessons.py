"""2027 lesson geometry and route contracts, including shortcut rejection."""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import mujoco
import numpy as np
import torch

from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import (
  PassageTerrainCfg, KINDS, lesson_route, lgdog1_passage_terrain_cfg)
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
from src.tasks.velocity.mdp import passage


class Lessons2027Tests(unittest.TestCase):
  def test_race_pretraining_collects_race_obstacles_and_has_separate_output(self):
    from scripts import pretrain_mgdp
    import sys
    with patch.object(sys,"argv",["pretrain_mgdp.py","--task","LgDog1-Race-MGDP"]):
      args=pretrain_mgdp.parse_args()
    self.assertTrue(args.out.endswith("mgdp_race_v3.pt"))
    args.num_envs=2;args.terrain_rows=2
    with patch.object(pretrain_mgdp,"ManagerBasedRlEnv") as create:
      pretrain_mgdp.build_env(args)
    cfg=create.call_args.kwargs["cfg"]
    self.assertEqual(set(cfg.scene.terrain.terrain_generator.sub_terrains),set(KINDS))
    self.assertEqual(cfg.scene.terrain.terrain_generator.difficulty_range,(0.,1.))
    self.assertFalse(cfg.curriculum)
    self.assertEqual(cfg.events["reset_base"].params["pose_range"]["x"],(0.,.2))

  def compile(self, kind, difficulty):
    spec=mujoco.MjSpec();spec.worldbody.add_body(name="terrain")
    PassageTerrainCfg(kind=kind,size=(8.,8.)).function(difficulty,spec,np.random.default_rng(2027))
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data

  def matching(self, model, suffix):
    return [model.geom(i) for i in range(model.ngeom) if suffix in model.geom(i).name]

  def test_extended_stones_repeat_reference_units_without_changing_gaps(self):
    model,data=self.compile("stepping_stones",0.)
    for y in (-.162,.162):
      hit=mujoco.mj_ray(model,data,np.array([3.,4.+y,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
      self.assertAlmostEqual(1-hit,.02,places=6)
    model,_=self.compile("stepping_stones",1.)
    stones=self.matching(model,"_stone_")
    self.assertEqual(len(stones),12)
    source=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/
      "src/assets/terrains/race_obstacle_2027.xml"))
    for geom in stones:
      row,col=map(int,geom.name.split("_stone_")[-1].split("_"))
      ref=source.geom(f"stone_{row}_{col%3}")
      np.testing.assert_allclose(geom.size,ref.size,atol=1e-8)
      np.testing.assert_allclose(geom.pos[:2]-[1.4+1.8*(col//3),4.],ref.pos[:2],atol=1e-8)
    self.assertAlmostEqual(self.matching(model,"stones_exit")[0].pos[0],6.3)
    self.assertEqual(lesson_route("stepping_stones")[-1],(6.5,0.))

  def test_gap_has_solid_ramp_and_reference_platforms_gap_and_treads(self):
    model,data=self.compile("gap_steps",1.)
    self.assertEqual(len(self.matching(model,"gap_ramp")),1)
    platforms=self.matching(model,"gap_platform")
    for geom in platforms:
      np.testing.assert_allclose(geom.size,[.4,.4,.18],atol=1e-8)
    self.assertAlmostEqual(platforms[1].pos[0]-platforms[0].pos[0]-.8,.36)
    treads=self.matching(model,"gap_tread")
    self.assertEqual(len(treads),2)
    np.testing.assert_allclose([g.pos[2]+g.size[2] for g in treads],[.24,.12])
    # Mid-slope support is continuous, not a disconnected thin floating slab.
    x=2.+.989092/2
    hit=mujoco.mj_ray(model,data,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
    self.assertAlmostEqual(1-hit,.18,places=5)

  def test_slalom_is_six_short_cylinders_in_an_l_and_pit_has_two_arms(self):
    model,_=self.compile("slalom_posts",1.)
    poles=self.matching(model,"slalom_pole")
    self.assertEqual(len(poles),6)
    for pole in poles:
      self.assertEqual(pole.type,mujoco.mjtGeom.mjGEOM_CYLINDER)
      np.testing.assert_allclose(pole.size[:2],[.05,.05])
    np.testing.assert_allclose(poles[-1].pos[:2]-poles[-3].pos[:2],[0,1.8])
    model,_=self.compile("brick_pit",1.)
    self.assertEqual(len(self.matching(model,"pit_base_")),2)
    self.assertEqual(len(self.matching(model,"pit_rail_")),6)
    self.assertEqual(len(self.matching(model,"pit_diagonal")),1)

  def test_joined_course_preserves_units_rotates_pit_and_shares_one_floor(self):
    model,data=self.compile("slalom_brick",1.)
    source,source_data=self.compile("brick_pit",1.)
    self.assertEqual(len(self.matching(model,"slalom_pole")),6)
    self.assertEqual(len(self.matching(model,"lesson_floor")),1)
    self.assertEqual(len(self.matching(model,"pit_base_")),2)
    self.assertEqual(len(self.matching(model,"pit_rail_")),6)
    rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    for geom in self.matching(model,"pit_"):
      suffix=geom.name.split("_pit_")[-1]
      ref=self.matching(source,"_pit_"+suffix)[0]
      np.testing.assert_allclose(geom.size,ref.size)
      np.testing.assert_allclose(geom.pos,rotation@(ref.pos-[2.5,3.,0.])+[5.5,6.9,0.])
      np.testing.assert_allclose(data.geom_xmat[geom.id].reshape(3,3),
        rotation@source_data.geom_xmat[ref.id].reshape(3,3),atol=1e-8)
      bounds=np.abs(data.geom_xmat[geom.id].reshape(3,3))@geom.size
      self.assertTrue(((geom.pos-bounds)[:2]>=0).all())
      self.assertTrue(((geom.pos+bounds)[:2]<=8).all())
    # Flat connecting aisle and flat hind-foot landing at the earliest success.
    for x,y in ((5.1,6.4),(5.3,6.4),(7.,4.15+.15+.52)):
      ray=mujoco.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
      self.assertAlmostEqual(ray,1.,places=6)
    # Turn on the floor/fill (3-5cm), away from the 15cm diagonal rail.
    for x,y in ((6.4,6.4),(7.,5.4)):
      ray=mujoco.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
      self.assertGreaterEqual(ray,.95-1e-6)
      self.assertLessEqual(ray,.97+1e-6)

  def test_joined_course_requires_pit_after_slalom_without_resetting_progress(self):
    env,command,robot=self.make_command()
    i=KINDS.index("slalom_brick")
    route=lesson_route("slalom_brick")
    robot.data.root_link_pos_w[i,:2]=torch.tensor(route[-1])
    self.assertFalse(passage.passage_success(env)[i])
    robot.data.root_link_pos_w.zero_()
    for wp in lesson_route("slalom_posts")[1:]:
      robot.data.root_link_pos_w[i,:2]=torch.tensor(wp)
      command._update_command()
    self.assertFalse(passage.passage_success(env)[i])
    slalom_progress=command.route_state()[3][i].item()
    for wp in route[len(lesson_route("slalom_posts")):]:
      robot.data.root_link_pos_w[i,:2]=torch.tensor(wp)
      command._update_command()
    self.assertTrue(passage.passage_success(env)[i])
    self.assertGreater(command.route_state()[3][i].item(),slalom_progress+4.1)

  def test_no_artificial_pits_and_safe_route_endpoints_all_levels(self):
    for kind in KINDS:
      for difficulty in (0.,.5,1.):
        with self.subTest(kind=kind,difficulty=difficulty):
          model,data=self.compile(kind,difficulty)
          for x,y in (lesson_route(kind)[0],lesson_route(kind)[-1]):
            distance=mujoco.mj_ray(model,data,np.array([x+1.,y+4.,1.]),
              np.array([0.,0.,-1.]),None,1,-1,None)
            self.assertAlmostEqual(distance,1.,places=5)
          # The reference uses the arena floor as the lowest support surface.
          for x in np.linspace(2.,5.6,25):
            distance=mujoco.mj_ray(model,data,np.array([x,4.,2.]),np.array([0.,0.,-1.]),None,1,-1,None)
            self.assertLessEqual(distance,2.+1e-6)

  def make_command(self):
    cfg=lgdog1_race_mgdp_env_cfg(with_perception=False)
    n=len(KINDS)
    generator=lgdog1_passage_terrain_cfg();generator.num_cols=n
    robot=NS(data=NS(root_link_pos_w=torch.zeros(n,3),heading_w=torch.zeros(n)))
    class Scene(dict): pass
    scene=Scene(robot=robot)
    scene.terrain=NS(cfg=NS(terrain_generator=generator),terrain_types=torch.arange(n))
    scene.env_origins=torch.zeros(n,3)
    env=NS(num_envs=n,device="cpu",scene=scene,step_dt=.02,cfg=cfg)
    command=passage.PassageCommand(cfg.commands["twist"],env)
    env.command_manager=NS(get_term=lambda _:command)
    command._resample_command(torch.arange(n))
    return env,command,robot

  def test_routes_reject_goal_teleport_and_reward_y_motion_without_farming(self):
    env,command,robot=self.make_command()
    index=KINDS.index("slalom_posts")
    final=torch.tensor(lesson_route("slalom_posts")[-1])
    robot.data.root_link_pos_w[index,:2]=final
    self.assertFalse(passage.passage_success(env)[index])
    self.assertTrue(passage.left_lane(env)[index])
    # Walk every segment geometrically; this tests the command contract,
    # not whether a learned policy can physically execute it.
    robot.data.root_link_pos_w.zero_()
    for wp in lesson_route("slalom_posts")[1:]:
      robot.data.root_link_pos_w[index,:2]=torch.tensor(wp)
      command._update_command()
    self.assertTrue(passage.passage_success(env)[index])
    before=command.waypoint_index.clone()
    command._resample_command(torch.tensor([index]))
    self.assertEqual(command.waypoint_index[index],1)
    torch.testing.assert_close(command.waypoint_index[:index],before[:index])

  def test_vertical_segment_has_progress_and_heading_and_frame_clearance_matches(self):
    env,command,robot=self.make_command()
    i=KINDS.index("brick_pit")
    command.waypoint_index[i]=3
    start=torch.tensor(lesson_route("brick_pit")[2])
    robot.data.root_link_pos_w[i,:2]=start
    env.cfg=NS(terminations={})
    progress=passage.ForwardProgress(NS(),env);progress.reset()
    p0=command.route_state()[3][i].item()
    robot.data.root_link_pos_w[i,1]+=.2
    command._update_command()
    self.assertGreater(command.route_state()[3][i].item(),p0+.19)
    self.assertGreater(progress(env)[i].item(),0)
    robot.data.root_link_pos_w[i,1]-=.2
    self.assertEqual(progress(env)[i].item(),0)
    robot.data.root_link_pos_w[i,1]+=.2
    self.assertEqual(progress(env)[i].item(),0)
    self.assertGreater(command.command[i,1],0)
    self.assertAlmostEqual(command.heading_target[i].item(),np.pi/2,places=5)
    model,_=self.compile("clearance_frame",1.)
    # Four cages along X, seven lanes across Y, and one shared floor.
    self.assertEqual(model.ngeom,1+4*7*11)
    roofs=[model.geom(i) for i in range(model.ngeom) if abs(model.geom(i).size[2]-.01)<1e-8]
    self.assertEqual(len(roofs),4*7*5)
    for geom in roofs:
      self.assertAlmostEqual(geom.pos[2]-geom.size[2],.29)

  def test_dense_wall_crawl_and_tyre_fields_keep_reference_units_and_clear_pads(self):
    for kind in ("high_wall","clearance_frame","tire_array"):
      with self.subTest(kind=kind):
        model,data=self.compile(kind,1.)
        floors=[i for i in range(model.ngeom) if min(model.geom_size[i,:2])>=4.]
        self.assertEqual(len(floors),1)
        ids=[i for i in range(model.ngeom) if i not in floors]
        bounds=np.array([np.abs(data.geom_xmat[i].reshape(3,3))@model.geom_size[i] for i in ids])
        low=(data.geom_xpos[ids]-bounds).min(0)
        high=(data.geom_xpos[ids]+bounds).max(0)
        self.assertGreaterEqual(low[0],1.65)
        self.assertLessEqual(high[0],5.55)
        self.assertGreaterEqual(low[1],.35)
        self.assertLessEqual(high[1],7.65)
        self.assertGreater(high[1]-low[1],6.5)
        # Includes the earliest successful root position minus the hind-foot
        # offset: the last obstacle must actually be behind the robot.
        for x in (1.,1.5,6.3-.15-.52,6.3):
          ray=mujoco.mj_ray(model,data,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
          self.assertAlmostEqual(ray,1.,places=6)
        if kind=="high_wall":
          self.assertEqual(len(ids),27)
          np.testing.assert_allclose(np.unique(data.geom_xpos[ids,0]),[2.3,3.7,5.1])
          for i in ids:
            np.testing.assert_allclose(model.geom_size[i],[.04,.4,.15])
        elif kind=="clearance_frame":
          for y in range(1,8):
            ray=mujoco.mj_ray(model,data,np.array([1.,float(y),.30]),np.array([1.,0.,0.]),None,1,-1,None)
            self.assertAlmostEqual(ray,.69,places=5)
        else:
          self.assertEqual(len(ids),12*6*16)
          centres=data.geom_xpos[ids].reshape(12,6,16,3).mean(2)
          np.testing.assert_allclose(np.diff(centres[0,:,0]),.55)
          np.testing.assert_allclose(np.diff(centres[:,0,1]),.55)
          np.testing.assert_allclose(centres[5:7,0,1],[4.-.275,4.+.275])

  def test_command_can_follow_every_route_inside_lane_within_timeout(self):
    env,command,robot=self.make_command()
    command.speed.fill_(.25)
    n=len(KINDS)
    complete=torch.zeros(n,dtype=torch.bool)
    for _ in range(int(env.cfg.episode_length_s/env.step_dt)):
      command._update_command()
      heading=robot.data.heading_w.clone()
      vx,vy=command.command[:,0],command.command[:,1]
      world=torch.stack((vx*heading.cos()-vy*heading.sin(),
                         vx*heading.sin()+vy*heading.cos()),dim=-1)
      robot.data.root_link_pos_w[~complete,:2]+=world[~complete]*env.step_dt
      robot.data.heading_w[~complete]+=command.command[~complete,2]*env.step_dt
      complete |= passage.passage_success(env)
      self.assertFalse(passage.left_lane(env)[~complete].any())
      if complete.all(): break
    self.assertTrue(complete.all(),[KINDS[i] for i in torch.where(~complete)[0].tolist()])
    terrain=env.scene.terrain
    terrain.terrain_levels=torch.zeros(n,dtype=torch.long)
    terrain.max_terrain_level=10
    terrain.terrain_origins=torch.zeros(10,n,3)
    terrain.env_origins=env.scene.env_origins
    env.cfg=NS(episode_length_s=75.)
    curriculum=passage.PassageCurriculum(NS(params={}),env)
    saved=curriculum.state_dict()
    curriculum.load_state_dict(saved)
    # Same grid and statistics, different route: full-resume must be rejected.
    command.cfg.waypoints={**command.cfg.waypoints,"slalom_posts":((0.,0.),(5.,1.))}
    changed=passage.PassageCurriculum(NS(params={}),env)
    with self.assertRaisesRegex(ValueError,"Incompatible"):
      changed.load_state_dict(saved)


if __name__=="__main__":
  unittest.main()
