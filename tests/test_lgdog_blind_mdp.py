"""CPU regression tests. Run: python -m unittest discover -s tests -v"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


# Load tensor terms directly, without importing the GPU task registry.
PATH = Path(__file__).resolve().parents[1] / "src/tasks/velocity/config/lgdog1/blind_mdp.py"
SPEC = importlib.util.spec_from_file_location("lgdog_blind_mdp_test", PATH)
mdp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mdp)


def ray_env(foot_z, ground_z, distance):
  return NS(scene={"foot": NS(data=NS(
    pos_w=torch.tensor([[0., 0., foot_z]]),
    hit_pos_w=torch.tensor([[[0., 0., ground_z]]]),
    distances=torch.tensor([[distance]]),
  ))})


class ClearanceTests(unittest.TestCase):
  def test_world_translation_invariance(self):
    flat, _ = mdp.foot_clearance_relative(ray_env(.15, 0., .65), ["foot"])
    raised, _ = mdp.foot_clearance_relative(ray_env(1.15, 1., .65), ["foot"])
    torch.testing.assert_close(flat, raised)

  def test_stair_height_is_subtracted(self):
    h, valid = mdp.foot_clearance_relative(ray_env(.25, .20, .55), ["foot"])
    self.assertAlmostEqual(h.item(), .05)
    self.assertTrue(valid.item())

  def test_ray_miss_and_nan_are_masked(self):
    h, valid = mdp.foot_clearance_relative(ray_env(.25, float("nan"), -1.), ["foot"])
    result = mdp.clearance_cost(h, torch.ones_like(h), torch.zeros_like(valid),
                                valid, torch.tensor([True]), .1)
    self.assertEqual(result.item(), 0.)

  def test_stance_and_standing_are_not_penalized(self):
    h = torch.zeros(2, 2)
    cost = mdp.clearance_cost(h, torch.ones_like(h),
      torch.tensor([[True, False], [False, False]]), torch.ones_like(h, dtype=torch.bool),
      torch.tensor([True, False]), .1)
    torch.testing.assert_close(cost, torch.tensor([.1, 0.]))


class SwingClearanceTests(unittest.TestCase):
  def make_env(self,dt=.02):
    scene={'feet':NS(data=NS(found=torch.ones(2,4,dtype=torch.long)))}
    for i in range(4):
      scene[str(i)]=NS(data=NS(pos_w=torch.zeros(2,3),hit_pos_w=torch.zeros(2,1,3),distances=torch.ones(2,1)))
    command=torch.tensor([[.4,0.,0.],[.4,0.,0.]])
    env=NS(scene=scene,num_envs=2,device='cpu',step_dt=dt,episode_length_buf=torch.full((2,),23,dtype=torch.long),
           extras={'log':{}},command_manager=NS(get_command=lambda _:command))
    params={'sensor_names':tuple(str(i) for i in range(4)),'asset_cfg':None,'sensor_name':'feet'}
    return env,mdp.feet_clearance_swing(NS(params=params),env),params,command

  def test_required_height_ramps_up_and_down_without_penalizing_stance(self):
    phase=torch.tensor([[0.,.56,.78,.99999]])
    cost,desired,usable=mdp.swing_clearance_cost(torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool),
                        torch.tensor([True]),phase,.06,.01,.56)
    self.assertFalse(usable[0,0].item())
    self.assertAlmostEqual(desired[0,2].item(),.06,places=5)
    self.assertLess(desired[0,1].item(),1e-6)
    self.assertLess(desired[0,3].item(),1e-6)
    self.assertGreater(cost.item(),0.)

  def test_staying_on_ground_and_short_contact_chatter_cannot_evade_cost(self):
    env,term,params,_=self.make_env()
    grounded=term(env,**params)
    self.assertTrue((grounded>0).all())
    env.scene['feet'].data.found.zero_()
    torch.testing.assert_close(grounded,term(env,**params))
    env.scene['feet'].data.found[:]=1
    torch.testing.assert_close(grounded,term(env,**params))

  def test_sufficient_clearance_has_no_extra_incentive_to_lift_higher(self):
    env,term,params,_=self.make_env()
    for height in [.06,.12,.30]:
      for i in range(4):env.scene[str(i)].data.pos_w[:,2]=height
      torch.testing.assert_close(term(env,**params),torch.zeros(2))

  def test_standing_invalid_rays_and_partial_reset_have_no_stale_swing_state(self):
    env,term,params,cmd=self.make_env()
    cmd[0].zero_()
    self.assertEqual(term(env,**params)[0].item(),0.)
    for i in range(4):
      env.scene[str(i)].data.distances[1]=-1
      env.scene[str(i)].data.hit_pos_w[1]=float('nan')
    torch.testing.assert_close(term(env,**params),torch.zeros(2))
    env,term,params,_=self.make_env()
    env.episode_length_buf[0]=0
    result=term(env,**params)
    self.assertEqual(result[0].item(),0.)
    self.assertGreater(result[1].item(),0.)

  def test_vertical_world_translation_does_not_change_cost(self):
    env,term,params,_=self.make_env();original=term(env,**params)
    for i in range(4):
      env.scene[str(i)].data.pos_w[:,2]+=2.
      env.scene[str(i)].data.hit_pos_w[:,:,2]+=2.
    torch.testing.assert_close(original,term(env,**params))

  def test_dense_cost_has_no_landing_impulse_and_integrates_consistently(self):
    integrals=[]
    for dt in [.01,.02]:
      env,term,params,_=self.make_env(dt);total=torch.zeros(2)
      for step in range(round(.6/dt)):
        env.episode_length_buf[:]=step
        cost=term(env,**params)
        self.assertTrue(((cost>=0)&(cost<=1)).all())
        total+=cost*dt
      integrals.append(total)
    torch.testing.assert_close(integrals[0],integrals[1],atol=1e-3,rtol=1e-3)

  def test_real_reward_manager_scales_dense_cost_once(self):
    from mjlab.managers.reward_manager import RewardManager,RewardTermCfg
    env,term,params,_=self.make_env();env.max_episode_length_s=20.
    expected=term(env,**params)*(-.2)*env.step_dt
    manager=RewardManager({'foot_clearance':RewardTermCfg(func=mdp.feet_clearance_swing,weight=-.2,params=params)},env)
    torch.testing.assert_close(manager.compute(env.step_dt),expected)
    manager.reset(torch.tensor([0]));env.episode_length_buf[0]=0
    self.assertEqual(manager.compute(env.step_dt)[0].item(),0.)

  def test_invalid_phase_or_height_parameters_fail_at_construction(self):
    env,_,params,_=self.make_env()
    for extra in ({'period':0.},{'stance_fraction':1.},{'target_height':0.},
                  {'tolerance':.06},{'offset':[0.,.5]}):
      with self.subTest(extra=extra),self.assertRaises(ValueError):
        mdp.feet_clearance_swing(NS(params={**params,**extra}),env)


class CurriculumTests(unittest.TestCase):
  def test_success_failure_hysteresis_and_zero_command(self):
    expected = torch.tensor([5., 5., 5., 0., 5., 5.])
    progress = torch.tensor([4.5, 1., 2.5, 8., 5., 5.])
    exposure = torch.tensor([2., 2., 2., 2., 0., 2.])
    elapsed = torch.full((6,), 20.)
    failed = torch.tensor([False, False, False, False, False, True])
    up, down, _ = mdp.terrain_decisions(expected, progress, exposure, elapsed, failed, 20.)
    self.assertEqual(up.tolist(), [True, False, False, False, False, False])
    self.assertEqual(down.tolist(), [False, True, False, False, False, True])

  def test_short_episode_cannot_promote(self):
    up, _, _ = mdp.terrain_decisions(torch.tensor([3.]), torch.tensor([3.]),
      torch.tensor([2.]), torch.tensor([5.]), torch.tensor([False]), 20.)
    self.assertFalse(up.item())

  def test_velocity_quality_gate_blocks_promotion(self):
    args = (
      torch.tensor([5.]), torch.tensor([4.5]), torch.tensor([2.]),
      torch.tensor([20.]), torch.tensor([False]), 20.,
    )
    up_bad, _, _ = mdp.terrain_decisions(*args, velocity_error=torch.tensor([.9]))
    up_good, _, _ = mdp.terrain_decisions(*args, velocity_error=torch.tensor([.7]))
    self.assertFalse(up_bad.item())
    self.assertTrue(up_good.item())

  def test_progress_ignores_drift_backwards_and_overspeed(self):
    command = torch.tensor([[1., 0., 0.]] * 4)
    env = NS(step_dt=.02, scene={"robot": NS(data=NS(
      root_link_lin_vel_b=torch.tensor([[0., 1., 0.], [-1., 0., 0.], [3., 0., 0.], [.5, 0., 0.]])))},
      command_manager=NS(get_command=lambda _: command))
    torch.testing.assert_close(mdp.tracking_progress(env), torch.tensor([0., 0., .02, .01]))

  def test_velocity_tracking_score_excludes_standing_commands(self):
    command = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = NS(
      command_manager=NS(get_command=lambda _: command),
      scene={"robot": NS(data=NS(
        root_link_lin_vel_b=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
      ))},
    )
    active = mdp.velocity_tracking_active(env)
    score = mdp.velocity_tracking_score(env)
    torch.testing.assert_close(active, torch.tensor([1.0, 0.0]))
    self.assertGreater(score[0].item(), 0.0)
    self.assertEqual(score[1].item(), 0.0)

  def test_command_resampling_integrates_each_segment(self):
    command = torch.tensor([[1., 0., 0.]])
    env = NS(step_dt=1., command_manager=NS(get_command=lambda _: command),
      scene={"robot": NS(data=NS(root_link_lin_vel_b=torch.tensor([[1., 0., 0.]])))})
    expected = mdp.commanded_distance(env)
    progress = mdp.tracking_progress(env)
    command[:] = torch.tensor([[0., 1., 0.]])
    env.scene["robot"].data.root_link_lin_vel_b[:] = torch.tensor([[0., 1., 0.]])
    expected += mdp.commanded_distance(env)
    progress += mdp.tracking_progress(env)
    command.zero_()  # final standing command must not erase previous demand
    expected += mdp.commanded_distance(env)
    self.assertEqual(expected.item(), 2.)
    self.assertEqual(progress.item(), 2.)

  def test_reset_uses_completed_sums_and_only_selected_envs(self):
    selected = []
    terrain = NS(terrain_levels=torch.tensor([1, 2]),
      update_env_origins=lambda ids, up, down: selected.append((ids.tolist(), up.tolist(), down.tolist())))
    env = NS(metrics_manager=NS(_episode_sums={
      "commanded_distance": torch.tensor([5., 0.]), "tracking_progress": torch.tensor([5., 0.]),
      "outside_spawn_platform": torch.tensor([2., 0.])}),
      termination_manager=NS(terminated=torch.tensor([False, False])),
      episode_length_buf=torch.tensor([1000, 0]), step_dt=.02, max_episode_length_s=20.,
      scene=NS(terrain=terrain))
    result = mdp.terrain_levels_tracking(env, torch.tensor([0]))
    self.assertEqual(selected, [([0], [True], [False])])
    self.assertEqual(result["episode_progress_m"].item(), 5.)
    self.assertEqual(env.metrics_manager._episode_sums["tracking_progress"][0].item(), 5.)

  def test_separate_terrain_gate_metrics_do_not_change_upgrade_decisions(self):
    selected=[]
    terrain=NS(terrain_levels=torch.zeros(4), update_env_origins=lambda ids, up, down:
               selected.append((up.tolist(), down.tolist())))
    env=NS(metrics_manager=NS(_episode_sums={
      "commanded_distance":torch.tensor([4.,2.,4.,4.]),
      "tracking_progress":torch.tensor([2.,1.8,4.,4.]),
      "outside_spawn_platform":torch.tensor([2.,2.,0.,2.])}),
      termination_manager=NS(terminated=torch.zeros(4,dtype=torch.bool)),
      episode_length_buf=torch.full((4,),1000),step_dt=.02,max_episode_length_s=20.,
      scene=NS(terrain=terrain))
    result=mdp.terrain_levels_tracking(env,torch.arange(4))
    self.assertEqual(selected[0][0],[False,False,False,True])
    self.assertEqual(float(result["ratio_gate_pass_fraction"]),.75)
    self.assertEqual(float(result["distance_gate_pass_fraction"]),.75)
    self.assertEqual(float(result["exposure_gate_pass_fraction"]),.75)
    self.assertEqual(float(result["progress_gate_pass_fraction"]),.5)

  def test_velocity_gate_uses_control_step_mean(self):
    selected = []
    terrain = NS(terrain_levels=torch.tensor([1.]),
      update_env_origins=lambda ids, up, down: selected.append((ids.tolist(), up.tolist(), down.tolist())))
    # 0.2 m/s accumulated over 20 seconds at a 50 Hz control rate.  The raw
    # sum is 200, while dividing by elapsed seconds would incorrectly yield 10.
    env = NS(metrics_manager=NS(_episode_sums={
      "commanded_distance": torch.tensor([5.]), "tracking_progress": torch.tensor([4.5]),
      "outside_spawn_platform": torch.tensor([2.]),
      "velocity_error_xy_mps": torch.tensor([200.])}),
      termination_manager=NS(terminated=torch.tensor([False])),
      episode_length_buf=torch.tensor([1000]), step_dt=.02, max_episode_length_s=20.,
      scene=NS(terrain=terrain))
    result = mdp.terrain_levels_tracking(env, torch.tensor([0]), max_velocity_error=.8)
    self.assertEqual(selected, [([0], [True], [False])])
    self.assertAlmostEqual(result["velocity_error_xy_mps"].item(), .2, places=6)

  def test_legacy_curriculum_can_disable_velocity_gate(self):
    expected = torch.tensor([5.])
    progress = torch.tensor([4.5])
    exposure = torch.tensor([2.])
    elapsed = torch.tensor([20.])
    failed = torch.tensor([False])
    up, _, _ = mdp.terrain_decisions(
      expected, progress, exposure, elapsed, failed, 20.,
      velocity_error=None,
    )
    self.assertTrue(up.item())


class ContactTests(unittest.TestCase):
  def setUp(self):
    self.force = torch.zeros(2, 2, 4, 3)
    self.env = NS(step_dt=.02, scene={"legs": NS(data=NS(force_history=self.force))})
    self.term = mdp.sustained_leg_contact(NS(params={"duration_s": .1}), self.env)

  def test_brief_scrape_recovers(self):
    self.force[0, 0, :, 2] = 15.
    self.assertFalse(self.term(self.env, "legs")[0].item())
    self.force.zero_()
    self.assertFalse(self.term(self.env, "legs")[0].item())
    self.assertEqual(self.term.count[0, 0].item(), 0)

  def test_persistent_contact_and_partial_reset(self):
    self.force[:, 0, :, 2] = 15.
    for _ in range(4):
      self.assertFalse(self.term(self.env, "legs").any().item())
    self.assertTrue(self.term(self.env, "legs").all().item())
    self.term.reset(torch.tensor([0]))
    self.assertEqual(self.term.count[0].sum().item(), 0)
    self.assertEqual(self.term.count[1, 0].item(), 5)

  def test_severe_substep_hit_terminates_immediately(self):
    self.force[0, 1, 2, 0] = 81.
    self.assertEqual(self.term(self.env, "legs").tolist(), [True, False])


if __name__ == "__main__":
  unittest.main()
