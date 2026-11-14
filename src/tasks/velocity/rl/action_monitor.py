"""RSL-RL action wrapper with diagnostics for raw policy outputs.

The environment clips actions before sending them to the action manager.  If
the policy output grows large, the clipped action can still look reasonable in
the normal action-rate metric while the exported policy is already unusable.
This wrapper records both sides of that boundary in ``extras["log"]``.
"""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


class ActionMonitorVecEnvWrapper(RslRlVecEnvWrapper):
  """Record raw action magnitude and gently penalize unbounded outputs."""

  def __init__(self, env, clip_actions: float | None = None,
               raw_action_penalty_coef: float = 0.0):
    super().__init__(env, clip_actions=clip_actions)
    self.raw_action_penalty_coef = float(raw_action_penalty_coef)
    self._raw_action_abs_max_seen = None
    self._raw_action_abs_mean_last = None
    self._action_saturation_fraction_last = None
    self.action_monitor_steps = 0

  # Convert only when a caller explicitly requests the diagnostic scalar.
  # The rollout hot path keeps these reductions on device.
  @property
  def raw_action_abs_max_seen(self):
    return 0.0 if self._raw_action_abs_max_seen is None else float(self._raw_action_abs_max_seen.item())

  @property
  def raw_action_abs_mean_last(self):
    return 0.0 if self._raw_action_abs_mean_last is None else float(self._raw_action_abs_mean_last.item())

  @property
  def action_saturation_fraction_last(self):
    return 0.0 if self._action_saturation_fraction_last is None else float(self._action_saturation_fraction_last.item())

  def step(self, actions: torch.Tensor):
    raw_actions = actions.detach()
    raw_abs = raw_actions.abs()
    raw_max = raw_abs.amax()
    raw_mean = raw_abs.mean()

    if self.clip_actions is None:
      saturation = torch.zeros((), device=raw_actions.device)
      clipped_actions = actions
    else:
      limit = float(self.clip_actions)
      saturation = (raw_abs >= limit).float().mean()
      clipped_actions = torch.clamp(actions, -limit, limit)

    self._raw_action_abs_max_seen = (raw_max if self._raw_action_abs_max_seen is None
                                    else torch.maximum(self._raw_action_abs_max_seen, raw_max))
    self._raw_action_abs_mean_last = raw_mean
    self._action_saturation_fraction_last = saturation
    self.action_monitor_steps += 1

    # Let the base wrapper preserve TensorDict conversion, timeout handling,
    # and the VecEnv done convention.  It will clamp a second time when a
    # limit is configured; the already clipped tensor makes that idempotent.
    obs, rewards, dones, extras = super().step(clipped_actions)
    raw_penalty = torch.zeros_like(rewards)
    if self.raw_action_penalty_coef > 0.0 and self.clip_actions is not None:
      excess = (raw_abs - float(self.clip_actions)).clamp_min(0.0)
      raw_penalty = (
        excess.square().mean(dim=-1)
        * self.raw_action_penalty_coef
        * float(self.unwrapped.step_dt)
      )
      raw_penalty = raw_penalty.reshape_as(rewards)
      rewards = rewards - raw_penalty
    log = extras.setdefault("log", {})
    log["Metrics/raw_action_abs_max"] = raw_max
    log["Metrics/raw_action_abs_mean"] = raw_mean
    log["Metrics/action_saturation_fraction"] = saturation
    if self.raw_action_penalty_coef > 0.0:
      log["Metrics/raw_action_penalty"] = raw_penalty.mean()
    return obs, rewards, dones, extras
