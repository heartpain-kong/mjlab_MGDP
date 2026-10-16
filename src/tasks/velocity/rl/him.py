"""HIM-inspired actor, auxiliary learning, and export for mjlab/RSL-RL 5."""

import copy
from pathlib import Path

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from .him_estimator import HIMEstimator


class HIMActor(MLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim, history_length=6,
               frame_dim=47, latent_dim=16, **kwargs):
    self.history_length, self.frame_dim, self.latent_dim = history_length, frame_dim, latent_dim
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    # One shared normalizer for all timestamps, updated from current frames only.
    self.obs_normalizer = EmpiricalNormalization(frame_dim) if self.obs_normalization else nn.Identity()
    self.estimator = HIMEstimator(frame_dim, history_length, latent_dim)

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    groups = obs_groups[obs_set]
    if tuple(groups) != ("actor",):
      raise ValueError("HIM actor must receive only the proprioceptive actor history")
    if tuple(obs["actor"].shape[1:]) != (self.history_length, self.frame_dim):
      raise ValueError("HIM requires frame-major [N, history_length, frame_dim] observations")
    return list(groups), self.history_length * self.frame_dim

  def _get_latent_dim(self):
    return self.frame_dim + 3 + self.latent_dim

  def get_latent(self, obs, masks=None, hidden_state=None):
    history = self.obs_normalizer(obs["actor"])
    # PPO must not update identification through the action loss.
    with torch.no_grad():
      velocity, latent = self.estimator.encode(history)
    return torch.cat((history[:, -1], velocity, latent), dim=-1)

  def update_normalization(self, obs):
    if self.obs_normalization:
      self.obs_normalizer.update(obs["actor"][:, -1])

  def as_onnx(self, verbose=False):
    return HIMExport(self)

  def as_jit(self):
    return HIMExport(self)


class HIMExport(nn.Module):
  """Stateless export: deployment supplies six raw frames, oldest first."""
  is_recurrent = False

  def __init__(self, actor):
    super().__init__()
    self.frame_dim = actor.frame_dim
    self.history_length = actor.history_length
    self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
    self.encoder = copy.deepcopy(actor.estimator.encoder)
    self.mlp = copy.deepcopy(actor.mlp)
    self.output = actor.distribution.as_deterministic_output_module()

  def forward(self, obs):
    history = self.obs_normalizer(obs.reshape(-1, self.history_length, self.frame_dim))
    prediction = self.encoder(history.flatten(start_dim=1))
    latent = torch.nn.functional.normalize(prediction[:, 3:], dim=-1)
    features = torch.cat((history[:, -1], prediction[:, :3], latent), dim=-1)
    return self.output(self.mlp(features))

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.history_length * self.frame_dim),)

  @property
  def input_names(self):
    return ["obs_history"]

  @property
  def output_names(self):
    return ["actions"]


class HIMPPO(PPO):
  """Standard PPO followed by separately optimized HIM identification.

  Auto-reset transitions (including timeouts) are excluded from identification:
  mjlab returns reset observations, not terminal successor observations.
  """

  def __init__(self, *args, estimator_learning_rate=1e-3, estimator_epochs=1, **kwargs):
    super().__init__(*args, **kwargs)
    if self.is_multi_gpu or self.symmetry or self.rnd:
      raise ValueError("This HIM experiment supports single-device PPO without symmetry/RND")
    self.estimator_epochs = estimator_epochs
    estimator_ids = {id(p) for p in self.actor.estimator.parameters()}
    for group in self.optimizer.param_groups:
      group["params"] = [p for p in group["params"] if id(p) not in estimator_ids]
    self.estimator_optimizer = torch.optim.Adam(
      self.actor.estimator.parameters(), lr=estimator_learning_rate
    )
    shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
    self.next_frames = torch.zeros(*shape, self.actor.frame_dim, device=self.device)
    self.next_velocities = torch.zeros(*shape, 3, device=self.device)
    self.identification_valid = torch.zeros(*shape, dtype=torch.bool, device=self.device)

  def process_env_step(self, obs, rewards, dones, extras):
    step = self.storage.step
    self.next_frames[step].copy_(obs["him_target"])
    self.next_velocities[step].copy_(obs["him_velocity"])
    self.identification_valid[step].copy_(~dones.bool().reshape(-1))
    super().process_env_step(obs, rewards, dones, extras)

  def update(self):
    # PPO keeps the encoder fixed throughout its epochs. Storage.clear() only
    # resets the write index in RSL-RL 5.0.1; observations remain readable here.
    history = self.storage.observations["actor"].flatten(0, 1)
    result = super().update()
    indices = self.identification_valid.flatten().nonzero().flatten()
    result["him_valid_fraction"] = self.identification_valid.float().mean().item()
    result["him_velocity_mse"] = 0.0
    result["him_swap"] = 0.0
    updates = 0
    next_frames = self.next_frames.flatten(0, 1)
    velocities = self.next_velocities.flatten(0, 1)
    for _ in range(self.estimator_epochs):
      shuffled = indices[torch.randperm(indices.numel(), device=self.device)]
      for ids in torch.tensor_split(shuffled, self.num_mini_batches):
        if ids.numel() < 2:
          continue  # A single example cannot support batch-level assignments.
        with torch.no_grad():
          normalized_history = self.actor.obs_normalizer(history[ids])
          normalized_next = self.actor.obs_normalizer(next_frames[ids])
        velocity_loss, swap_loss = self.actor.estimator.losses(
          normalized_history, normalized_next, velocities[ids]
        )
        loss = velocity_loss + swap_loss
        if not torch.isfinite(loss):
          raise FloatingPointError("Non-finite HIM identification loss")
        self.estimator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.estimator.parameters(), 10.0)
        self.estimator_optimizer.step()
        result["him_velocity_mse"] += velocity_loss.item()
        result["him_swap"] += swap_loss.item()
        updates += 1
    # Prevent stale estimator gradients from entering PPO's global clipping.
    self.estimator_optimizer.zero_grad(set_to_none=True)
    if updates:
      result["him_velocity_mse"] /= updates
      result["him_swap"] /= updates
    result["him_updates"] = float(updates)
    return result

  def save(self):
    data = super().save()
    data["him_estimator_optimizer"] = self.estimator_optimizer.state_dict()
    data["him_contract"] = {"frame_dim": self.actor.frame_dim,
                            "history_length": self.actor.history_length,
                            "order": "oldest_to_newest"}
    return data

  def load(self, loaded_dict, load_cfg=None, strict=True):
    expected = {"frame_dim": self.actor.frame_dim,
                "history_length": self.actor.history_length, "order": "oldest_to_newest"}
    if loaded_dict.get("him_contract") != expected:
      raise ValueError("Not a compatible LgDog HIM checkpoint; use fresh training")
    iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer"):
      self.estimator_optimizer.load_state_dict(loaded_dict["him_estimator_optimizer"])
    return iteration


class HIMRunner(MjlabOnPolicyRunner):
  def save(self, path, infos=None):
    super().save(path, infos)
    directory = str(Path(path).parent)
    self.export_policy_to_onnx(directory)

  def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
    super().export_policy_to_onnx(path, filename, verbose)
    metadata = get_base_metadata(self.env.unwrapped, "lgdog_him")
    metadata.update({"policy_architecture": "lgdog_him_v1",
                     "history_length": str(self.alg.actor.history_length),
                     "frame_dim": str(self.alg.actor.frame_dim),
                     "history_order": "oldest_to_newest",
                     "history_reset": "repeat_first_frame",
                     "control_dt": str(self.env.unwrapped.step_dt)})
    attach_metadata_to_onnx(str(Path(path) / filename), metadata)
