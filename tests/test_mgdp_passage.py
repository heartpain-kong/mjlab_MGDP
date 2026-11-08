"""Passage goal, anti-farming, contacts and outcome-gated speed regression."""

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import numpy as np
import torch
import mujoco
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains import TerrainGeneratorCfg
import src.tasks
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.config.lgdog1.blind_mdp import sustained_leg_contact
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg, KINDS, lesson_route


class MGDPPassageTests(unittest.TestCase):
  def test_workbench_cannot_replace_passage_gate_with_lp_or_signed_speed(self):
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      checkpoint = root/"logs/mgdp/perception/mgdp_race_v3.pt"
      checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"placeholder")
      for option in ({"terrain_curriculum":"lp_acrl"}, {"velocity_max_abs":.8}):
        with self.assertRaisesRegex(ValueError,"通过率门控"):
          build_command(root,"python",{"mode":"train","task":"race_mgdp",**option})

  def test_config_objective_reward_and_task_isolation(self):
    cfg = load_env_cfg("LgDog1-Race-MGDP")
    self.assertEqual(set(cfg.curriculum), {"passage"})
    self.assertIsInstance(cfg.commands["twist"], passage.PassageCommandCfg)
    self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (.25,.35))
    self.assertEqual(cfg.curriculum["passage"].params["speed_limit"], 1.0)
    for task in ("LgDog1-Race-MGDP", "LgDog1-Rough-MGDP"):
      c = load_env_cfg(task)
      self.assertEqual(c.rewards["foot_clearance"].weight, 0)
      self.assertEqual(c.rewards["feet_air_time"].weight, .25)
      self.assertEqual(c.terminations["illegal_contact"].params["sensor_name"], "trunk_ground_contact")
      self.assertIn("sustained_leg_contact", c.terminations)
      self.assertTrue(load_rl_cfg(task).experiment_name.endswith("_v3" if task=="LgDog1-Race-MGDP" else "_v2"))
    rough = load_env_cfg("LgDog1-Rough-MGDP")
    self.assertNotIn("velocity_stages", rough.curriculum["command_vel"].params)
    self.assertEqual(load_env_cfg("LgDog1-Rough-Blind-HIM-Standard").commands["twist"].ranges.lin_vel_x,(-.2,.4))

  def test_all_lessons_have_clear_entry_and_exit(self):
    for kind in KINDS:
      for difficulty in (0., 1.):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        terrain = PassageTerrainCfg(kind=kind, size=(8.,8.))
        result = terrain.function(difficulty, spec, np.random.default_rng(42))
        model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model,data)
        np.testing.assert_array_equal(result.origin, [1.,4.,0.])
        for x,y in ([1.,4.],np.array(lesson_route(kind)[-1])+[1.,4.]):
          hit = mujoco.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,1,-1,None)
          self.assertAlmostEqual(hit,1.,places=5,msg=kind)

  def fake_env(self):
    xy = torch.zeros(4,2)
    command = NS(local_xy=lambda:xy, cfg=NS(goal_x=5.3,lane_half_width=.28,
      ranges=NS(lin_vel_x=(.25,.35))))
    failure = torch.zeros(4,dtype=torch.bool)
    terrain = NS(terrain_types=torch.tensor([0,0,1,1]),terrain_levels=torch.zeros(4,dtype=torch.long),
      max_terrain_level=3, cfg=NS(terrain_generator=TerrainGeneratorCfg(size=(8.,8.),num_cols=2,num_rows=3,
        sub_terrains={"flat":PassageTerrainCfg(kind="start_flat",proportion=.5),
                      "wall":PassageTerrainCfg(kind="high_wall",proportion=.5)})))
    terrain.update_env_origins = lambda ids,up,down:terrain.terrain_levels.__setitem__(ids,
      (terrain.terrain_levels[ids]+up.long()-down.long()).clamp(0,2))
    env = NS(num_envs=4,device="cpu",step_dt=.02, scene=NS(terrain=terrain),
      command_manager=NS(get_term=lambda name:command),episode_length_buf=torch.ones(4,dtype=torch.long)*100,
      reset_buf=torch.ones(4,dtype=torch.bool),
      cfg=NS(terminations={"impact":NS(time_out=False)}),
      termination_manager=NS(get_term=lambda name:failure),
      metrics_manager=NS(_step_count=torch.ones(4,dtype=torch.long)*100,
        _episode_sums={"velocity_error_xy_mps":torch.ones(4)*10}))
    return env,xy,failure

  def test_goal_lane_collision_and_progress_farming(self):
    env,xy,failure = self.fake_env()
    progress = passage.ForwardProgress(NS(),env); progress.reset()
    xy[:,0] = .1
    self.assertTrue((progress(env)>0).all())
    xy[:,0] = -.1
    self.assertTrue((progress(env)==0).all())
    xy[:,0] = .1
    self.assertTrue((progress(env)==0).all())
    xy[:,0] = 5.4; xy[1,1] = .4; failure[2] = True
    self.assertTrue(passage.passage_success(env)[0])
    self.assertFalse(passage.passage_success(env)[1])
    self.assertEqual(passage.success_bonus(env)[2].item(),0)
    self.assertEqual(passage.failure_cost(env)[0].item(),0)
    self.assertEqual(passage.success_bonus(env)[0].item()*env.step_dt,1.)

  def test_speed_waits_for_all_columns_and_three_success_windows(self):
    env,xy,_ = self.fake_env()
    curriculum = passage.PassageCurriculum(NS(params={}),env)
    xy[:,0] = 5.4
    for _ in range(3):
      curriculum(env,torch.tensor([0,1]),min_episodes=2)
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x[1],.35)
    curriculum.count.zero_(); curriculum.success.zero_(); curriculum.error.zero_()
    for i in range(3):
      curriculum(env,torch.arange(4),min_episodes=2)
      self.assertAlmostEqual(curriculum.command.cfg.ranges.lin_vel_x[1],.375 if i==2 else .35)
    # Initial reset cannot advance counters or change the selected terrain.
    env.episode_length_buf.zero_(); before = env.scene.terrain.terrain_levels.clone()
    curriculum(env,torch.arange(4),min_episodes=2)
    self.assertEqual(curriculum.count.sum().item(),0)
    torch.testing.assert_close(before,env.scene.terrain.terrain_levels)
    env.episode_length_buf.fill_(200)  # randomized counter, only 100 real steps
    curriculum(env,torch.arange(4),min_episodes=2)
    self.assertEqual(curriculum.count.sum().item(),0)

  def test_resume_legacy_speed_ceiling_and_expand_to_one_meter_per_second(self):
    old_env, _, _ = self.fake_env()
    old = passage.PassageCurriculum(NS(params={"min_episodes": 2, "speed_limit": .6}), old_env)
    old.command.cfg.ranges.lin_vel_x = (.25, .6)
    old.terrain.terrain_levels.fill_(1)
    old.streak = 2
    state = old.state_dict()

    env, xy, _ = self.fake_env()
    env.scene.terrain.terrain_origins = torch.zeros(3, 2, 3)
    env.scene.terrain.env_origins = torch.zeros(4, 3)
    curriculum = passage.PassageCurriculum(NS(params={"min_episodes": 2, "speed_limit": 1.}), env)
    curriculum.load_state_dict(state)
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x, (.25, .6))
    self.assertEqual(curriculum.streak, 2)
    torch.testing.assert_close(curriculum.terrain.terrain_levels, state["levels"])
    xy[:, 0] = 5.4
    curriculum(env, torch.arange(4), **curriculum.params)
    self.assertAlmostEqual(curriculum.command.cfg.ranges.lin_vel_x[1], .625)
    for _ in range(50):
      curriculum(env, torch.arange(4), **curriculum.params)
    self.assertEqual(curriculum.command.cfg.ranges.lin_vel_x, (.25, 1.))

    with self.assertRaisesRegex(ValueError, "Incompatible passage curriculum"):
      curriculum.load_state_dict({**state, "params": {"min_episodes": 3, "speed_limit": .6}})
    with self.assertRaisesRegex(ValueError, "Invalid passage speed range"):
      curriculum.load_state_dict({**state, "ranges": (.25, .625)})

  def test_leg_scrape_grace_and_severe_contact(self):
    data = NS(force_history=torch.zeros(2,3,4,3))
    env = NS(step_dt=.02,scene={"legs":NS(data=data)})
    term = sustained_leg_contact(NS(params={"duration_s":.1}),env)
    data.force_history[0,0,:,0] = 15
    for _ in range(4):
      self.assertFalse(term(env,"legs").any())
    self.assertTrue(term(env,"legs")[0])
    term.reset(torch.tensor([0])); data.force_history[0].zero_()
    data.force_history[1,1,0,0] = 100
    self.assertTrue(term(env,"legs")[1])

  def test_curriculum_fingerprint_includes_himloco_physics_and_delay(self):
    env, _, _ = self.fake_env()
    original = passage.PassageCurriculum(NS(params={}), env).fingerprint
    cfg = load_env_cfg("LgDog1-Race-MGDP")
    env.cfg.events = cfg.events
    env.cfg.actions = cfg.actions
    current = passage.PassageCurriculum(NS(params={}), env).fingerprint
    self.assertNotEqual(original, current)
    self.assertEqual(current, passage.PassageCurriculum(NS(params={}), env).fingerprint)
    env.cfg.events["himloco_pd_motor"].params["kp_range"] = (.8, 1.2)
    modified = passage.PassageCurriculum(NS(params={}), env).fingerprint
    self.assertNotEqual(current, modified)
    env.cfg.actions["joint_pos"].max_delay_steps = 1
    self.assertNotEqual(modified, passage.PassageCurriculum(NS(params={}), env).fingerprint)


if __name__ == "__main__":
  unittest.main()
