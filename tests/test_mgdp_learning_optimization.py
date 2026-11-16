"""Start at 0.6 and detect stalled turns without punishing real progress."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,tempfile,unittest
import torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from src.config.mgdp_camera import D435I20_TASK,D435I20_ALIAS
from src.config.mgdp_course_contract import rotation_stall_contract,validate_rotation_stall_resume
from src.tasks.velocity.mdp.wall_training import NoPassageProgress
from src.dashboard.commands import build_command


class LearningOptimizationTests(unittest.TestCase):
  def test_zero_width_start_still_enables_adaptive_curriculum(self):
    cfg=load_env_cfg(D435I20_TASK);agent=load_rl_cfg(D435I20_TASK)
    self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x,(.6,.6))
    self.assertEqual(cfg.curriculum['passage'].params['speed_floor'],.6)
    self.assertEqual(cfg.curriculum['passage'].params['speed_limit'],2.)
    self.assertNotIn('fixed_speed',cfg.curriculum['passage'].params)
    self.assertEqual(agent.actor.distribution_cfg['init_std'],.25)
    self.assertEqual(agent.algorithm.entropy_coef,0.)
    self.assertEqual(load_rl_cfg('LgDog1-Race-MGDP').actor.distribution_cfg['init_std'],.5)

  def test_turn_stall_vs_genuine_turns_stationary_and_waypoint_change(self):
    n=5;progress=torch.zeros(n);xy=torch.zeros(n,2)
    cmd=NS(command=torch.tensor([[0.,0.,.4]]).repeat(n,1),heading_error=torch.full((n,),.8),
      waypoint_index=torch.ones(n,dtype=torch.long),local_xy=lambda:xy,
      route_state=lambda:(None,None,None,progress),route_success=lambda:progress>=5.3)
    cmd.command[2]=0. # Intentional standing is never treated as a stuck turn.
    env=NS(num_envs=n,device='cpu',step_dt=.02,command_manager=NS(get_term=lambda _:cmd))
    term=NoPassageProgress(NS(),env);params={'turn_timeout_s':4.,'turn_progress_rad':.05,'turn_min_command_radps':.05}
    for step in range(201):
      cmd.heading_error[1]=.8-step*.00035 # >.05 rad progress within each window.
      cmd.heading_error[3]=.8+(.02 if step%2 else 0.) # Oscillation cannot refresh best progress.
      if step==100:cmd.waypoint_index[4]=2 # New target resets the clock.
      result=term(env,**params)
      if step<200:self.assertFalse(result.any())
    self.assertTrue(result[0]);self.assertFalse(result[1]);self.assertFalse(result[2])
    self.assertTrue(result[3]);self.assertFalse(result[4])
    term.reset(torch.tensor([0]));self.assertFalse(term(env,**params)[0])
    self.assertTrue(term(env,**params)[3])

  def test_rotation_rule_is_part_of_full_resume_contract(self):
    cfg=load_env_cfg(D435I20_TASK);contract=rotation_stall_contract(cfg)
    self.assertEqual(contract,{'turn_timeout_s':4.,'turn_progress_rad':.05,'turn_min_command_radps':.05})
    with self.assertRaisesRegex(ValueError,'warm-start-checkpoint'):validate_rotation_stall_resume(cfg,{'infos':{}})
    validate_rotation_stall_resume(cfg,{'infos':{'mgdp_rotation_stall':contract}})
    with self.assertRaises(ValueError):validate_rotation_stall_resume(cfg,{'infos':{'mgdp_rotation_stall':{**contract,'turn_timeout_s':8.}}})

  def test_workbench_warm_start_uses_model_without_requiring_default_encoder(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);checkpoint=root/'model_20000.pt';checkpoint.touch()
      options={'mode':'train','task':D435I20_ALIAS,'warm_start':True,'checkpoint':str(checkpoint)}
      argv=build_command(root,sys.executable,options)
      self.assertIn('--warm-start-checkpoint='+str(checkpoint.resolve()),argv)
      self.assertNotIn('--agent.resume=True',argv)
      with self.assertRaisesRegex(ValueError,'只能选择一种'):build_command(root,sys.executable,{**options,'resume':True})
      with self.assertRaises(ValueError):build_command(root,sys.executable,{**options,'task':'standard'})


if __name__=='__main__':unittest.main()
