"""V6 reward defaults, 0.3 action mapping, 0.6–2.0 curriculum and replay contracts."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
import torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.config.mgdp_camera import D435I20_TASK,D435I20_ALIAS
from src.config.mgdp_course_contract import current_contract,validate_resume,restore_for_play
from src.config.passage_speed import restore_fixed_speed_for_play
from src.dashboard.catalog import TASKS
from src.tasks.velocity.mdp.course_v3 import StoneGroundContact
from src.tasks.velocity.mdp.passage import PassageCurriculum
from src.wall_reward_settings import wall_reward_contract,load_wall_reward_settings
from tests.test_mgdp_fixed_speed import fake_env


class RebaselineTests(unittest.TestCase):
  def test_actual_wall_task_defaults_match_v6_reward_archive(self):
    cfg=load_env_cfg(D435I20_TASK)
    expected={'track_linear_velocity':2.4,'track_angular_velocity':1.2,'body_orientation_l2':-1.,'pose':.5,
      'foot_gait':0.,'body_ang_vel':-.05,'angular_momentum':-.025,'joint_acc_l2':-2.5e-7,
      'joint_pos_limits':-10.,'action_rate_l2':-.05,'foot_clearance':0.,'foot_slip':-.25,
      'soft_landing':-.002,'feet_stumble':-.1,'feet_air_time':.5,'leg_contact_force':-.25,
      'stand_still':-1.,'is_terminated':-200.,'passage_progress':1.5,'passage_success':10.,
      'wall_waiting':-1.5,'wall_crossing':5.,'wall_swing_clearance':-.2}
    self.assertEqual({n:r.weight for n,r in cfg.rewards.items()},expected)
    self.assertEqual({k:cfg.rewards['track_linear_velocity'].params[k] for k in ('relative_weight','relative_std','min_command_speed','progress_reference_mps','min_command_mps')},
      {'relative_weight':.5,'relative_std':.5,'min_command_speed':.2,'progress_reference_mps':.6,'min_command_mps':.05})
    self.assertEqual({k:cfg.rewards['wall_waiting'].params[k] for k in ('approach_m','grace_s','ramp_s','max_rate','min_progress_m','min_command_mps','root_clearance_m','foot_clearance_m')},
      {'approach_m':.5,'grace_s':.75,'ramp_s':1.5,'max_rate':2.,'min_progress_m':.02,'min_command_mps':.05,'root_clearance_m':.1,'foot_clearance_m':.03})
    self.assertEqual(wall_reward_contract(load_wall_reward_settings()),'1:5687753ef1012ea739656366ad204e8435b0ecd39dfe66318d37825c2116ace3')
    self.assertEqual(cfg.actions['joint_pos'].scale,.3)
    self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x,(.6,.6))
    self.assertEqual(cfg.curriculum['passage'].params['speed_floor'],.6)
    self.assertEqual(cfg.curriculum['passage'].params['speed_limit'],2.)
    self.assertEqual(cfg.terminations['stone_ground_contact'].params['min_feet'],2)
    self.assertEqual((TASKS[D435I20_ALIAS].passage_start_low,TASKS[D435I20_ALIAS].passage_start_high,TASKS[D435I20_ALIAS].passage_speed_limit),(.6,.6,2.))
    # Ordinary race MGDP has its original speed and action mapping.
    old=load_env_cfg('LgDog1-Race-MGDP')
    self.assertEqual(old.actions['joint_pos'].scale,.25)
    self.assertEqual(old.commands['twist'].ranges.lin_vel_x,(.25,.35))

  def test_speed_floor_and_upper_limit_survive_repeated_windows(self):
    env,_,_,_=fake_env(False);env.command_manager.get_term('twist').cfg.ranges.lin_vel_x=(.6,.7)
    params={'min_episodes':1,'speed_floor':.6,'speed_limit':2.}
    term=PassageCurriculum(NS(params=params),env);empty=torch.empty(0,dtype=torch.long)
    for _ in range(20):
      term.count.fill_(1);term.success.zero_();term.error.zero_();term(env,empty,**params)
    self.assertEqual(term.command.cfg.ranges.lin_vel_x,(.6,.6))
    for _ in range(3):
      term.count.fill_(1);term.success.fill_(1);term.error.zero_();term(env,empty,**params)
    self.assertAlmostEqual(term.command.cfg.ranges.lin_vel_x[1],.625)
    term.command.cfg.ranges.lin_vel_x=(.6,2.)
    for _ in range(3):
      term.count.fill_(1);term.success.fill_(1);term.error.zero_();term(env,empty,**params)
    self.assertEqual(term.command.cfg.ranges.lin_vel_x,(.6,2.))
    state=term.state_dict();term.load_state_dict(state)
    with self.assertRaises(ValueError):term.load_state_dict({**state,'ranges':(.25,.7)})

  def test_play_restores_legacy_scale_stone_rule_and_speed(self):
    cfg=load_env_cfg(D435I20_TASK);legacy={'infos':{'mgdp_passage_curriculum':{'params':{},'ranges':(.25,.35)}}}
    with self.assertRaisesRegex(ValueError,'warm-start-checkpoint'):validate_resume(cfg,legacy)
    restore_for_play(cfg,legacy);restore_fixed_speed_for_play(cfg,legacy)
    self.assertEqual(cfg.actions['joint_pos'].scale,.25)
    self.assertEqual(cfg.terminations['stone_ground_contact'].params['min_feet'],1)
    self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x,(.25,.35))
    fresh=load_env_cfg(D435I20_TASK);saved={'infos':{'mgdp_course_contract':current_contract(fresh)}}
    validate_resume(fresh,saved)
    restore_for_play(cfg,saved);self.assertEqual(current_contract(cfg),current_contract(fresh))

  def test_stone_counts_distinct_feet_in_same_substep(self):
    cfg=load_env_cfg(D435I20_TASK);g=cfg.scene.terrain.terrain_generator;g.num_cols=10
    class Scene(dict):pass
    n=6;data=NS(pos_history=torch.zeros(n,16,4,3),normal_history=torch.zeros(n,16,4,3),
      force_history=torch.zeros(n,16,4,3),found_history=torch.zeros(n,16,4))
    origin=torch.tensor([[100.,200.,0.]]).repeat(n,1)
    data.pos_history[:,:,:,0]=102.;data.pos_history[:,:,:,1]=200.;data.normal_history[:,:,:,2]=1.;data.force_history[:,:,:,2]=2.
    # env0: one foot, multiple contact slots; env1: two feet at same substep;
    # env2: two feet at different substeps; env3: one unloaded; env4: pad; env5: wrong terrain.
    for i,slot,h in ((0,0,2),(0,1,2),(1,0,2),(1,4,2),(2,0,1),(2,4,2),
                     (3,0,2),(3,4,2),(4,0,2),(4,4,2),(5,0,2),(5,4,2)):data.found_history[i,slot,h]=1
    data.force_history[3,4,2,2]=.1;data.pos_history[4,:,:,0]=100.9
    scene=Scene(stone_support_contact=NS(data=data));scene.terrain=NS(cfg=NS(terrain_generator=g),env_origins=origin,terrain_types=torch.tensor([1,1,1,1,1,0]))
    env=NS(scene=scene,device='cpu',num_envs=n);term=StoneGroundContact(NS(),env)
    torch.testing.assert_close(term(env,min_feet=2),torch.tensor([False,True,False,False,False,False]))
    self.assertTrue(term(env,min_feet=1)[0])


if __name__=='__main__':unittest.main()
