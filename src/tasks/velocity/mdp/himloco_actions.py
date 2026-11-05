"""HIMLoco substep action delay and one-physics-step local force pulses."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_apply


@dataclass(kw_only=True)
class HIMLocoJointPositionActionCfg(JointPositionActionCfg):
  max_delay_steps: int = 3
  force_range: tuple[float, float] = (-30., 30.)
  force_interval_steps: int = 8
  force_body: str = "trunk"

  def build(self, env):
    return HIMLocoJointPositionAction(self, env)


class HIMLocoJointPositionAction(JointPositionAction):
  """Hold the preceding target for a sampled number of physics substeps."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._processed_actions[:] = self._offset
    self.previous_targets = self._processed_actions.clone()
    self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.local_force = torch.zeros((self.num_envs, 1, 3), device=self.device)
    self.last_force_world = torch.zeros_like(self.local_force)
    bodies, _ = self._entity.find_bodies(cfg.force_body)
    if len(bodies) != 1:
      raise ValueError("HIMLoco force pulse requires one trunk body")
    self.force_body_ids = bodies
    self.physics_phase = 0

  def process_actions(self, actions):
    self.previous_targets.copy_(self._processed_actions)
    super().process_actions(actions)
    self.delay_steps = torch.randint(
      0, self.cfg.max_delay_steps + 1, (self.num_envs,), device=self.device)
    self.physics_phase = 0
    self.local_force.zero_()
    step = self._env.common_step_counter
    if step > 0 and step % self.cfg.force_interval_steps == 0:
      self.local_force.uniform_(*self.cfg.force_range)

  def apply_actions(self):
    targets = torch.where(
      (self.physics_phase < self.delay_steps)[:, None],
      self.previous_targets, self._processed_actions)
    targets = targets - self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(targets, joint_ids=self._target_ids)
    if self.physics_phase == 0:
      self.last_force_world[:] = quat_apply(
        self._entity.data.root_link_quat_w, self.local_force[:, 0])[:, None, :]
    else:
      self.last_force_world.zero_()
    # MuJoCo retains xfrc_applied. Clear it after one physics substep.
    self._entity.write_external_wrench_to_sim(
      self.last_force_world, torch.zeros_like(self.last_force_world),
      body_ids=self.force_body_ids)
    self.physics_phase += 1

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    super().reset(ids)
    default = self._offset[ids] if isinstance(self._offset, torch.Tensor) else self._offset
    self._processed_actions[ids] = default
    self.previous_targets[ids] = default
    self.delay_steps[ids] = 0
    self.local_force[ids] = 0.
    self.last_force_world[ids] = 0.
    self._entity.write_external_wrench_to_sim(
      self.last_force_world[ids], torch.zeros_like(self.last_force_world[ids]),
      body_ids=self.force_body_ids, env_ids=ids)
