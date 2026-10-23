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
  """Record raw action magnitude and clipping saturation before ``step``."""

  def __init__(self, env, clip_actions: float | None = None):
    super().__init__(env, clip_actions=clip_actions)
    self.raw_action_abs_max_seen = 0.0
    self.raw_action_abs_mean_last = 0.0
    self.action_saturation_fraction_last = 0.0
    self.action_monitor_steps = 0

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

    self.raw_action_abs_max_seen = max(self.raw_action_abs_max_seen, float(raw_max.item()))
    self.raw_action_abs_mean_last = float(raw_mean.item())
    self.action_saturation_fraction_last = float(saturation.item())
    self.action_monitor_steps += 1

    # Let the base wrapper preserve TensorDict conversion, timeout handling,
    # and the VecEnv done convention.  It will clamp a second time when a
    # limit is configured; the already clipped tensor makes that idempotent.
    obs, rewards, dones, extras = super().step(clipped_actions)
    log = extras.setdefault("log", {})
    log["Metrics/raw_action_abs_max"] = raw_max
    log["Metrics/raw_action_abs_mean"] = raw_mean
    log["Metrics/action_saturation_fraction"] = saturation
    return obs, rewards, dones, extras
