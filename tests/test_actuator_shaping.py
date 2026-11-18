"""Physics contracts for actuator reality shaping (paper #5, arXiv:2607.02205).

Covers the four pieces added on top of the released HIMLoco profile:
transport delay ring buffer, torque-speed envelope, servo dead band and the
multiplicative joint friction/damping randomization, plus the task-cfg wiring
that keeps the profile disabled by default.
"""

from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg

from src.tasks.velocity.config.lgdog1 import domain_rand_cfg
from src.tasks.velocity.config.lgdog1.domain_rand_cfg import (
  ActuatorRealityProfile, HIMLOCO_DR, configure_race_himloco_randomization,
  himloco_randomization_metadata,
)
from src.tasks.velocity.mdp.actuator_reality import (
  ActionDelayBuffer, ActuatorRealityJointPositionActionCfg,
  resolve_actuator_ids, resolve_no_load_speeds, torque_speed_scale,
)
from src.tasks.velocity.mdp.himloco_actions import HIMLocoJointPositionActionCfg
from src.tasks.velocity.mdp.himloco_randomization import (
  joint_friction_damping_strength,
)

JOINTS = 4
JOINT_NAMES = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint"]


def make_action(kp=20., kd=0., **overrides):
  """Build the real action term against a recording robot interface."""
  num_envs = 4
  writes = {}
  entity = NS(
    find_joints_by_actuator_names=lambda names: (list(range(JOINTS)), list(JOINT_NAMES)),
    find_bodies=lambda name: ([0], ["trunk"]),
    actuator_names=tuple(JOINT_NAMES),
    indexing=NS(ctrl_ids=torch.arange(JOINTS, dtype=torch.int)),
    data=NS(
      default_joint_pos=torch.zeros(num_envs, JOINTS),
      encoder_bias=torch.zeros(num_envs, JOINTS),
      joint_pos=torch.zeros(num_envs, JOINTS),
      joint_vel=torch.zeros(num_envs, JOINTS),
      root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(num_envs, 1),
    ),
  )
  entity.set_joint_position_target = lambda targets, **kw: writes.__setitem__(
    "target", targets.clone())
  entity.write_external_wrench_to_sim = lambda force, torque, **kw: writes.__setitem__(
    "force", force.clone())
  gainprm = torch.zeros(JOINTS, 10)
  gainprm[:, 0] = kp
  biasprm = torch.zeros(JOINTS, 10)
  biasprm[:, 1] = -kp
  biasprm[:, 2] = -kd
  defaults = {"actuator_forcerange": torch.tensor([[-10., 10.]]).repeat(JOINTS, 1)}
  env = NS(
    num_envs=num_envs, device="cpu", common_step_counter=0,
    scene={"robot": entity},
    sim=NS(
      model=NS(actuator_gainprm=gainprm, actuator_biasprm=biasprm),
      get_default_field=lambda field: defaults[field],
    ),
  )
  cfg = ActuatorRealityJointPositionActionCfg(
    entity_name="robot", actuator_names=(".*",), scale=.25, offset=0.,
    max_delay_steps=0, force_range=(-30., 30.), force_interval_steps=8,
    **overrides,
  )
  action = cfg.build(env)
  action.delay_steps[:] = 0  # no substep hold unless a test asks for it.
  return action, env, entity, writes


def push_substep(action, processed):
  action._processed_actions = processed.clone()
  action.previous_targets = processed.clone()
  action.apply_actions()
  return action  # ``writes`` holds the target written this substep


class ActionDelayBufferTests(unittest.TestCase):
  def test_ring_buffer_returns_value_pushed_k_steps_ago(self):
    buffer = ActionDelayBuffer(num_envs=3, max_delay=2, dim=1, device="cpu")
    buffer.reset(torch.zeros(3, 1))
    buffer.delayed(torch.tensor([2, 1, 0]))  # advancing cursor keeps zeros
    for step in range(1, 5):
      buffer.push(torch.full((3, 1), float(step)))
      out = buffer.delayed(torch.tensor([2, 1, 0]))
      expected = torch.tensor([
        float(step - 2) if step >= 2 else 0., float(step - 1) if step >= 1 else 0.,
        float(step)])
      torch.testing.assert_close(out[:, 0], expected)

  def test_reset_refills_selected_environments_only(self):
    buffer = ActionDelayBuffer(num_envs=3, max_delay=2, dim=1, device="cpu")
    buffer.reset(torch.zeros(3, 1))
    buffer.push(torch.tensor([[1.], [2.], [3.]]))
    buffer.reset(7., torch.tensor([1]))
    torch.testing.assert_close(buffer.buffer[1], torch.full((3, 1), 7.))
    torch.testing.assert_close(buffer.buffer[0], torch.tensor([[1.], [0.], [0.]]))
    torch.testing.assert_close(buffer.buffer[2], torch.tensor([[3.], [0.], [0.]]))


class TorqueSpeedEnvelopeTests(unittest.TestCase):
  def test_scale_is_one_below_base_and_falls_linearly_to_no_load(self):
    velocity = torch.tensor([0., 5., 7.5, 10., 15.])
    scale = torque_speed_scale(velocity, base_speed=5., no_load_speed=10.)
    torch.testing.assert_close(
      scale, torch.tensor([1., 1., .5, 0., 0.]), rtol=0, atol=1e-6)

  def test_min_torque_factor_floors_the_envelope(self):
    scale = torque_speed_scale(
      torch.tensor([10., 20.]), 5., 10., min_scale=.2)
    torch.testing.assert_close(scale, torch.tensor([.2, .2]))

  def test_no_load_speeds_match_urdf(self):
    speeds = resolve_no_load_speeds(
      ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_calf_joint"],
      domain_rand_cfg.LGDOG1_NO_LOAD_SPEEDS, default_speed=18.85)
    torch.testing.assert_close(speeds, torch.tensor([18.85, 18.85, 9.425, 9.425]))

  def test_actuator_ids_follow_names_not_positions(self):
    entity = NS(
      actuator_names=("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
      indexing=NS(ctrl_ids=torch.tensor([7, 8, 9], dtype=torch.int)))
    ids = resolve_actuator_ids(entity, ["FL_calf_joint", "FL_hip_joint"])
    self.assertEqual(ids.tolist(), [9, 7])

  def test_real_lgdog1_joint_to_actuator_mapping(self):
    # Compiled LgDog1 order: joints are grouped per leg, actuators per joint type
    # (all hips, then thighs, then calves), so positional indexing would be wrong.
    joint_names = (
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")
    actuator_names = (
      "FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint",
      "FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint",
      "FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint")
    entity = NS(
      actuator_names=actuator_names,
      indexing=NS(ctrl_ids=torch.arange(12, dtype=torch.int)))
    self.assertEqual(
      resolve_actuator_ids(entity, joint_names).tolist(),
      [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

  def test_high_velocity_derates_the_written_pd_target(self):
    action, _, entity, writes = make_action(
      torque_speed_envelope=True, base_speed_fraction=.5,
      min_torque_factor=0., default_no_load_speed=10., no_load_speeds={})
    target = torch.full((4, JOINTS), .4)  # kp * .4 = 8 Nm, below the 10 Nm cap
    push_substep(action, target)
    torch.testing.assert_close(writes["target"], target)
    entity.data.joint_vel[:] = 7.5  # envelope scale = .5 -> only 5 Nm available
    push_substep(action, target)
    torch.testing.assert_close(writes["target"], torch.full((4, JOINTS), .25))

  def test_envelope_accounts_for_the_damping_term(self):
    action, _, entity, writes = make_action(
      kp=20., kd=2., torque_speed_envelope=True, base_speed_fraction=.5,
      min_torque_factor=0., default_no_load_speed=10., no_load_speeds={})
    # At 5 rad/s the envelope is still full (10 Nm); PD asks
    # kp*1.5 - kd*5 = 20 Nm, so ctrl = q + (10 + kd*qd)/kp = 1.0.
    entity.data.joint_vel[:] = 5.
    push_substep(action, torch.full((4, JOINTS), 1.5))
    torch.testing.assert_close(writes["target"], torch.full((4, JOINTS), 1.0))
    # Opposite direction: kp*1.5 + kd*5 = 40 Nm -> ctrl = (10 - 10)/20 = 0.
    entity.data.joint_vel[:] = -5.
    push_substep(action, torch.full((4, JOINTS), 1.5))
    torch.testing.assert_close(writes["target"], torch.zeros(4, JOINTS))


class DeadbandTests(unittest.TestCase):
  def test_small_position_errors_are_swallowed(self):
    action, _, _, writes = make_action(target_deadband=.01)
    for error in (.005, -.005, .0):
      push_substep(action, torch.full((4, JOINTS), error))
      torch.testing.assert_close(writes["target"], torch.zeros(4, JOINTS))
    for error in (.02, -.02):
      push_substep(action, torch.full((4, JOINTS), error))
      torch.testing.assert_close(writes["target"], torch.full((4, JOINTS), error))


class TransportDelayTests(unittest.TestCase):
  def test_delay_stream_is_shifted_per_environment(self):
    action, _, _, writes = make_action(action_delay_steps=(0, 2))
    action.transport_delay_steps = torch.tensor([2, 1, 0, 0])
    outputs = []
    for step in range(1, 5):
      push_substep(action, torch.full((4, JOINTS), float(step)))
      outputs.append(writes["target"][:, 0].clone())
    outputs = torch.stack(outputs)
    # delay 2: 0, 0, 1, 2 / delay 1: 0, 1, 2, 3 / delay 0: 1, 2, 3, 4
    torch.testing.assert_close(
      outputs,
      torch.tensor([[0., 0., 1., 1.], [0., 1., 2., 2.],
                    [1., 2., 3., 3.], [2., 3., 4., 4.]]))

  def test_reset_resamples_delay_and_refills_that_envs_buffer(self):
    action, _, _, _ = make_action(action_delay_steps=(1, 2))
    action.reset(torch.tensor([1]))
    self.assertTrue(1 <= action.transport_delay_steps[1].item() <= 2)
    torch.testing.assert_close(
      action._delay_buffer.buffer[1], torch.zeros_like(action._delay_buffer.buffer[1]))

  def test_zero_range_keeps_actions_immediate(self):
    action, _, _, writes = make_action(action_delay_steps=(0, 0))
    self.assertIsNone(action._delay_buffer)
    push_substep(action, torch.full((4, JOINTS), .3))
    torch.testing.assert_close(writes["target"], torch.full((4, JOINTS), .3))


def make_dr_env(num_envs=8, num_dofs=12):
  defaults = {
    "dof_frictionloss": torch.full((num_dofs,), .2),
    "dof_damping": torch.full((num_dofs,), .01),
  }
  fields = {name: value.unsqueeze(0).expand(num_envs, -1).clone()
            for name, value in defaults.items()}
  asset = NS(indexing=NS(joint_v_adr=torch.arange(num_dofs, dtype=torch.int)))
  env = NS(
    num_envs=num_envs, device="cpu", scene={"robot": asset},
    sim=NS(model=NS(**fields), get_default_field=lambda name: defaults[name]),
  )
  ranges = dict(frictionloss_scale_range=(.8, 1.2), damping_scale_range=(.8, 1.2),
                asset_cfg=NS(name="robot", joint_ids=slice(None)))
  return env, fields, defaults, ranges


class FrictionDampingRandomizationTests(unittest.TestCase):
  def test_factors_stay_in_range_scale_defaults_and_never_compound(self):
    env, fields, defaults, ranges = make_dr_env()
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(7)
      joint_friction_damping_strength(env, None, **ranges)
      first = {name: value.clone() for name, value in fields.items()}
      joint_friction_damping_strength(env, None, **ranges)
    for key, name in (("frictionloss_scale", "dof_frictionloss"),
                      ("damping_scale", "dof_damping")):
      factors = env._himloco_samples[key]
      self.assertTrue(((factors >= .8) & (factors <= 1.2)).all())
      self.assertGreater(factors.std().item(), 1e-3)
      torch.testing.assert_close(
        fields[name], defaults[name][None, :] * factors, rtol=1e-5, atol=1e-6)
    # Second call starts from the compile-time defaults, not from ``first``.
    self.assertFalse(torch.allclose(first["dof_frictionloss"], fields["dof_frictionloss"]))

  def test_subset_uses_only_the_selected_environments(self):
    env, fields, defaults, ranges = make_dr_env()
    ids = torch.tensor([1, 5])
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(3)
      joint_friction_damping_strength(env, ids, **ranges)
    for name, sample in (("dof_frictionloss", "frictionloss_scale"),
                         ("dof_damping", "damping_scale")):
      torch.testing.assert_close(
        fields[name][ids],
        defaults[name][None, :] * env._himloco_samples[sample][ids],
        rtol=1e-5, atol=1e-6)
    mask = torch.ones(8, dtype=torch.bool)
    mask[ids] = False
    for name in defaults:
      torch.testing.assert_close(fields[name][mask], defaults[name][None, :].expand(6, -1))

  def test_single_world_unexpanded_fields_are_supported(self):
    num_dofs = 6
    defaults = {"dof_frictionloss": torch.full((num_dofs,), .2),
                "dof_damping": torch.full((num_dofs,), .01)}
    fields = {name: value.clone() for name, value in defaults.items()}
    env = NS(
      num_envs=1, device="cpu", scene={"robot": NS(
        indexing=NS(joint_v_adr=torch.arange(num_dofs, dtype=torch.int)))},
      sim=NS(model=NS(**fields), get_default_field=lambda name: defaults[name]),
    )
    joint_friction_damping_strength(
      env, None, (.8, 1.2), (.8, 1.2), NS(name="robot", joint_ids=slice(None)))
    self.assertTrue((fields["dof_frictionloss"] != 0).all())
    self.assertTrue((fields["dof_damping"] != 0).all())


def make_task_cfg():
  def noop(*args, **kwargs):
    pass

  return NS(
    decimation=4,
    sim=NS(mujoco=NS(timestep=.005)),
    events={
      "push_robot": EventTermCfg(func=noop, mode="interval", interval_range_s=(10., 15.)),
      "base_com": EventTermCfg(func=noop, mode="startup", params={"ranges": {}}),
      "foot_friction": EventTermCfg(func=noop, mode="startup", params={"ranges": (.5, 1.)}),
      "reset_robot_joints": EventTermCfg(func=noop, mode="reset"),
    },
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=.25)},
  )


class DomainRandWiringTests(unittest.TestCase):
  def test_default_profile_keeps_the_released_action_and_events(self):
    cfg = make_task_cfg()
    configure_race_himloco_randomization(cfg)
    self.assertFalse(HIMLOCO_DR.actuator_reality.enabled)
    self.assertNotIn("actuator_reality", cfg.events)
    # The released term is installed unchanged, not the reality-shaped subclass.
    self.assertIs(type(cfg.actions["joint_pos"]), HIMLocoJointPositionActionCfg)
    self.assertEqual(cfg.actions["joint_pos"].max_delay_steps, 3)
    self.assertEqual(cfg.actions["joint_pos"].force_range, (-30., 30.))
    metadata = himloco_randomization_metadata(cfg)
    self.assertTrue(metadata["actuator_reality"]["enabled"] is False)

  def test_enabled_profile_installs_action_shaping_and_the_unified_event(self):
    profile = replace(
      HIMLOCO_DR, actuator_reality=ActuatorRealityProfile(enabled=True))
    with patch.object(domain_rand_cfg, "HIMLOCO_DR", profile):
      cfg = make_task_cfg()
      configure_race_himloco_randomization(cfg)
      metadata = himloco_randomization_metadata(cfg)
    action = cfg.actions["joint_pos"]
    self.assertIsInstance(action, ActuatorRealityJointPositionActionCfg)
    self.assertEqual(action.action_delay_steps, (0, 2))
    self.assertTrue(action.torque_speed_envelope)
    self.assertEqual(action.base_speed_fraction, .5)
    self.assertEqual(action.min_torque_factor, 0.)
    self.assertEqual(action.target_deadband, .01)
    self.assertEqual(action.default_no_load_speed, 18.85)
    self.assertEqual(action.no_load_speeds, domain_rand_cfg.LGDOG1_NO_LOAD_SPEEDS)
    # The released HIMLoco release terms are preserved on top of the shaping.
    self.assertEqual(action.max_delay_steps, 3)
    self.assertEqual(action.force_range, (-30., 30.))
    self.assertEqual(action.force_interval_steps, 8)
    event = cfg.events["actuator_reality"]
    self.assertEqual(event.mode, "reset")
    self.assertEqual(event.params["frictionloss_scale_range"], (.8, 1.2))
    self.assertEqual(event.params["damping_scale_range"], (.8, 1.2))
    self.assertEqual(event.params["asset_cfg"].joint_names, (".*",))
    # Declaring the fields is what tells the env to expand them per world.
    self.assertEqual(
      set(event.func.model_fields), {"dof_frictionloss", "dof_damping"})
    self.assertTrue(metadata["actuator_reality"]["enabled"])

  def test_configuring_twice_rebuilds_the_same_action_without_duplicate_kwargs(self):
    profile = replace(
      HIMLOCO_DR, actuator_reality=ActuatorRealityProfile(enabled=True))
    with patch.object(domain_rand_cfg, "HIMLOCO_DR", profile):
      cfg = make_task_cfg()
      configure_race_himloco_randomization(cfg)
      configure_race_himloco_randomization(cfg)
    self.assertIsInstance(
      cfg.actions["joint_pos"], ActuatorRealityJointPositionActionCfg)


if __name__ == "__main__":
  unittest.main()
