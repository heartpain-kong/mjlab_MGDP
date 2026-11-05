"""Physics contracts for the released HIMLoco randomization migration."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.mdp.himloco_actions import HIMLocoJointPositionAction
from src.tasks.velocity.mdp.himloco_randomization import (
  initial_joint_positions, pd_motor_strength, push_xy_velocity,
)


class HIMLocoRandomizationTests(unittest.TestCase):
  def test_pd_and_motor_factors_share_joints_preserve_caps_and_do_not_compound(self):
    n, joints = 16, 12
    gain = torch.zeros(joints, 10)
    bias = torch.zeros_like(gain)
    gain[:, 0] = torch.tensor([20., 20., 40.] * 4)
    bias[:, 1] = -gain[:, 0]
    bias[:, 2] = -torch.tensor([1., 1., 2.] * 4)
    caps = torch.tensor([[-v, v] for v in [10., 10., 20.] * 4])
    defaults = {"actuator_gainprm": gain, "actuator_biasprm": bias}
    model = NS(actuator_gainprm=gain.repeat(n, 1, 1), actuator_biasprm=bias.repeat(n, 1, 1),
      actuator_forcerange=caps.repeat(n, 1, 1))
    env = NS(num_envs=n, device="cpu", sim=NS(model=model, get_default_field=defaults.__getitem__),
      scene={"robot": NS(indexing=NS(ctrl_ids=torch.arange(joints, dtype=torch.int)))})
    args = dict(kp_range=(.9, 1.1), kd_range=(.9, 1.1), motor_strength_range=(.9, 1.1),
      asset_cfg=NS(name="robot"))
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(42)
      pd_motor_strength(env, None, **args)
      before = model.actuator_gainprm.clone()
      pd_motor_strength(env, torch.tensor([1, 3]), **args)
    for value in env._himloco_samples.values():
      self.assertTrue(((value >= .9) & (value <= 1.1)).all())
    sample = env._himloco_samples
    kp = sample["kp_factor"] * sample["motor_strength"]
    kd = sample["kd_factor"] * sample["motor_strength"]
    torch.testing.assert_close(model.actuator_gainprm[:, :, 0], gain[None, :, 0] * kp)
    torch.testing.assert_close(model.actuator_biasprm[:, :, 1], bias[None, :, 1] * kp)
    torch.testing.assert_close(model.actuator_biasprm[:, :, 2], bias[None, :, 2] * kd)
    torch.testing.assert_close(model.actuator_forcerange, caps.repeat(n, 1, 1))
    mask = torch.ones(n, dtype=torch.bool); mask[[1, 3]] = False
    torch.testing.assert_close(model.actuator_gainprm[mask], before[mask])

  def action(self):
    """Exercise real inherited processing with a recording robot interface."""
    action = HIMLocoJointPositionAction.__new__(HIMLocoJointPositionAction)
    action._env = NS(num_envs=4, device="cpu", common_step_counter=0)
    action.cfg = NS(max_delay_steps=3, force_interval_steps=8, force_range=(-30., 30.))
    action._offset = torch.tensor([[0., 0., .2] * 4]).repeat(4, 1)
    action._scale = .25
    action._raw_actions = torch.zeros(4, 12)
    action._processed_actions = action._offset.clone()
    action.previous_targets = action._offset.clone()
    action.delay_steps = torch.zeros(4, dtype=torch.long)
    action.local_force = torch.zeros(4, 1, 3)
    action.last_force_world = torch.zeros_like(action.local_force)
    action.physics_phase = 0
    action.force_body_ids = [0]
    action._target_ids = torch.arange(12)
    action._entity = NS(data=NS(encoder_bias=torch.full((4, 12), .01),
      root_link_quat_w=torch.tensor([[2**-.5, 0., 0., 2**-.5]]).repeat(4, 1)))
    action._entity.set_joint_position_target = lambda targets, **kw: setattr(action, "written_target", targets.clone())
    action._entity.write_external_wrench_to_sim = lambda force, torque, **kw: setattr(action, "written_force", force.clone())
    return action

  def test_each_physics_substep_uses_old_then_new_target_with_one_delay_per_robot(self):
    action = self.action()
    with patch("torch.randint", return_value=torch.arange(4)):
      action.process_actions(torch.ones(4, 12))
    for phase in range(4):
      action.apply_actions()
      expected = action._offset + (phase >= torch.arange(4))[:, None] * .25 - .01
      torch.testing.assert_close(action.written_target, expected)
      self.assertEqual(action.written_force.abs().max().item(), 0.)
    # A second policy block holds the preceding target, not the nominal pose.
    with patch("torch.randint", return_value=torch.full((4,), 3)):
      action.process_actions(torch.zeros(4, 12))
    action.apply_actions()
    torch.testing.assert_close(action.written_target, action._offset + .25 - .01)
    for _ in range(3): action.apply_actions()
    torch.testing.assert_close(action.written_target, action._offset - .01)

  def test_force_is_local_xyz_every_eight_steps_and_cleared_after_one_physics_step(self):
    action = self.action()
    action._env.common_step_counter = 7
    action.process_actions(torch.zeros(4, 12))
    self.assertEqual(action.local_force.abs().max().item(), 0.)
    action._env.common_step_counter = 8
    action.process_actions(torch.zeros(4, 12))
    self.assertGreater(action.local_force.abs().max().item(), 0.)
    self.assertLessEqual(action.local_force.abs().max().item(), 30.)
    # A 90 degree yaw rotates body-local +X into world +Y.
    action.local_force[:] = torch.tensor([10., 20., 30.])
    action.apply_actions()
    torch.testing.assert_close(action.written_force, torch.tensor([[[-20., 10., 30.]]]).repeat(4, 1, 1))
    for _ in range(3):
      action.apply_actions()
      self.assertEqual(action.written_force.abs().max().item(), 0.)

  def test_subset_action_reset_clears_previous_target_and_force_without_touching_others(self):
    action = self.action()
    action.process_actions(torch.ones(4, 12))
    action.local_force[:] = 20.
    action.reset(torch.tensor([1, 3]))
    torch.testing.assert_close(action._processed_actions[[1, 3]], action._offset[[1, 3]])
    torch.testing.assert_close(action.previous_targets[[1, 3]], action._offset[[1, 3]])
    self.assertEqual(action.local_force[[1, 3]].abs().max().item(), 0.)
    self.assertEqual(action.local_force[[0, 2]].min().item(), 20.)
    torch.testing.assert_close(action._processed_actions[[0, 2]], action._offset[[0, 2]] + .25)

  def test_push_replaces_xy_globally_and_preserves_vertical_and_angular_velocity(self):
    velocity = torch.tensor([[4., -4., .3, .4, .5, .6]]).repeat(8, 1)
    writes = []
    asset = NS(data=NS(root_link_vel_w=velocity), write_root_link_velocity_to_sim=lambda value, **kw: writes.append(value))
    env = NS(num_envs=8, device="cpu", step_dt=.02, common_step_counter=799, scene={"robot": asset})
    args = dict(interval_s=16., max_velocity=1., asset_cfg=NS(name="robot"))
    push_xy_velocity(env, None, **args)
    self.assertEqual(writes, [])
    env.common_step_counter = 800
    push_xy_velocity(env, None, **args)
    self.assertEqual(len(writes), 1)
    self.assertLessEqual(writes[0][:, :2].abs().max().item(), 1.)
    torch.testing.assert_close(writes[0][:, 2:], velocity[:, 2:])

  def test_initial_joint_pose_multiplies_nominal_and_resets_velocity(self):
    nominal = torch.tensor([[0., 0., .2] * 4]).repeat(8, 1)
    limits = torch.tensor([-1., 1.]).repeat(8, 12, 1)
    writes = []
    robot = NS(data=NS(default_joint_pos=nominal, soft_joint_pos_limits=limits),
      write_joint_state_to_sim=lambda pos, vel, **kw: writes.append((pos, vel, kw)))
    env = NS(num_envs=8, device="cpu", scene={"robot": robot})
    ids = torch.tensor([1, 4, 7])
    initial_joint_positions(env, ids, (.5, 1.5), NS(name="robot", joint_ids=slice(None)))
    pos, vel, kw = writes[0]
    self.assertEqual(pos[:, :2].abs().max().item(), 0.)
    self.assertTrue(((pos[:, 2::3] >= .1) & (pos[:, 2::3] <= .3)).all())
    self.assertEqual(vel.abs().max().item(), 0.)
    torch.testing.assert_close(kw["env_ids"], ids)


if __name__ == "__main__":
  unittest.main()
