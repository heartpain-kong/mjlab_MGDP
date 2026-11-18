"""Lightweight, opt-in automatic-curriculum helpers for the passage task.

The passage curriculum used to advance on raw success counts only.  The
following helpers add the four ideas from the local curriculum reading list
without importing mjlab, so they can be unit-tested with a fake environment:

* ``trajectory_quality_score`` folds one completed passage episode into a
  single score: route progress, speed-tracking error and a termination-reason
  penalty.  ``TrajectoryDifficultyScheduler`` promotes or demotes terrain
  difficulty from a streak of those window scores instead of bare successes
  (Trajectory-Level Automatic Curriculum Learning, arXiv:2608.16164).
* ``interpolate_bandwidth`` expands a non-forward command axis (yaw rate,
  lateral speed) proportionally with a speed group's forward progress, the
  multi-dimensional command curriculum of TransCurriculum (arXiv:2603.14156).
* ``survival_safety_margin`` is a sensor-free safety proxy (success plus the
  number of controlled steps survived) used to veto promotions when the
  predicted margin is too small (Predictive Safety Curricula, arXiv:2609.37070).

Every helper is disabled unless the task opts in; the defaults reproduce the
legacy outcome-gated curriculum exactly.
"""

from __future__ import annotations

import math

import torch

__all__ = [
  "interpolate_bandwidth",
  "survival_safety_margin",
  "trajectory_quality_score",
  "TrajectoryDifficultyScheduler",
]


def _tensor(value, name):
  """Convert ``value`` to a float tensor and reject non-finite input."""
  try:
    tensor = torch.as_tensor(value, dtype=torch.float32)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be numeric") from error
  if not torch.isfinite(tensor).all():
    raise ValueError(f"{name} must be finite")
  return tensor


def trajectory_quality_score(progress_ratio, speed_error, failed=None, success=None, *,
                             progress_weight=.5, error_weight=.3, failure_weight=.2,
                             error_ref=.5):
  """Weighted quality of one finished trajectory in ``[0, 1]``.

  ``progress_ratio`` is the fraction of the route covered (0 = start, 1 =
  goal), ``speed_error`` the mean absolute planar tracking error and
  ``failed``/``success`` the termination outcome.  A failed episode pays the
  full termination penalty; a timeout without success pays none here but has a
  low progress ratio, so it still scores low.  Weights are normalised, which
  keeps the score comparable across task revisions.
  """
  progress = _tensor(progress_ratio, "progress_ratio")
  error = _tensor(speed_error, "speed_error")
  weights = (float(progress_weight), float(error_weight), float(failure_weight))
  if any(not math.isfinite(weight) or weight < 0. for weight in weights):
    raise ValueError("Trajectory quality weights must be finite and non-negative")
  total = sum(weights)
  if total <= 0.:
    raise ValueError("Trajectory quality weights must not all be zero")
  error_ref = float(error_ref)
  if not math.isfinite(error_ref) or error_ref <= 0.:
    raise ValueError("Trajectory quality error_ref must be positive")
  if failed is not None:
    penalty = _tensor(failed, "failed").clamp(0., 1.)
  elif success is not None:
    penalty = 1.-_tensor(success, "success").clamp(0., 1.)
  else:
    penalty = torch.zeros_like(error)
  progress_term = progress.clamp(0., 1.)
  tracking_term = 1.-(error.clamp(min=0.)/error_ref).clamp(0., 1.)
  termination_term = 1.-penalty
  score = (weights[0]*progress_term + weights[1]*tracking_term
           + weights[2]*termination_term) / total
  return score.clamp(0., 1.)


def interpolate_bandwidth(initial, limit, progress):
  """Raise a bandwidth from ``initial`` towards ``limit`` with progress.

  ``progress`` is clamped to ``[0, 1]``; a decreasing limit is allowed so a
  task may also shrink a band, but the default use is monotone expansion.
  """
  initial, limit = float(initial), float(limit)
  if not math.isfinite(initial) or not math.isfinite(limit) or initial < 0.:
    raise ValueError("Bandwidth bounds must be finite and non-negative")
  progress = float(progress)
  if not math.isfinite(progress):
    raise ValueError("Bandwidth progress must be finite")
  progress = min(max(progress, 0.), 1.)
  return initial + progress*(limit-initial)


class TrajectoryDifficultyScheduler:
  """Promote/demote difficulty from consecutive trajectory-quality windows.

  Each ``observe`` call adds the quality scores of a completed batch to a
  pending window for one terrain group; ``update`` closes every window that
  holds at least ``min_samples`` samples.  A window mean above
  ``up_threshold`` extends the promotion streak, a mean below
  ``down_threshold`` extends the demotion streak, and anything in between
  clears both.  ``patience`` consecutive windows are required for a decision.

  The class owns no environment state, so ``state_dict``/``load_state_dict``
  are simple and old checkpoints (which never stored this object) keep working
  because the passage curriculum treats a missing state as a fresh scheduler.
  """

  def __init__(self, groups, *, up_threshold=.65, down_threshold=.35,
               patience=3, down_patience=None, min_samples=1):
    groups = tuple(groups)
    if not groups:
      raise ValueError("Trajectory scheduler needs at least one group")
    if len(set(groups)) != len(groups):
      raise ValueError("Trajectory scheduler groups must be unique")
    up_threshold, down_threshold = float(up_threshold), float(down_threshold)
    if not (math.isfinite(up_threshold) and math.isfinite(down_threshold)
            and 0. <= down_threshold < up_threshold <= 1.):
      raise ValueError("Trajectory scheduler needs 0 <= down_threshold < up_threshold <= 1")
    patience = int(patience)
    down_patience = patience if down_patience is None else int(down_patience)
    min_samples = int(min_samples)
    if patience < 1 or down_patience < 1:
      raise ValueError("Trajectory scheduler patience must be positive")
    if min_samples < 1:
      raise ValueError("Trajectory scheduler min_samples must be positive")
    self.groups = groups
    self.up_threshold = up_threshold
    self.down_threshold = down_threshold
    self.patience = patience
    self.down_patience = down_patience
    self.min_samples = min_samples
    self.up_streak = {name: 0 for name in groups}
    self.down_streak = {name: 0 for name in groups}
    self.last_window_quality = {name: None for name in groups}
    self.updates = 0
    self._pending = {name: [] for name in groups}

  def observe(self, group, scores):
    """Add one completed batch of trajectory quality scores for ``group``."""
    if group not in self._pending:
      raise KeyError(f"Unknown trajectory scheduler group {group!r}")
    values = _tensor(scores, "trajectory quality").flatten()
    if values.numel() == 0:
      return
    self._pending[group].append(values.clamp(0., 1.))

  def pending_samples(self, group):
    return int(sum(len(values) for values in self._pending[group]))

  def update(self):
    """Close ready windows and return ``{group: +1 | -1}`` decisions."""
    decisions = {}
    for name in self.groups:
      values = self._pending[name]
      if sum(len(value) for value in values) < self.min_samples:
        continue  # keep filling this window
      self._pending[name] = []
      window = torch.cat(values)
      quality = float(window.mean())
      self.last_window_quality[name] = quality
      if quality > self.up_threshold:
        self.up_streak[name] += 1
        self.down_streak[name] = 0
      elif quality < self.down_threshold:
        self.down_streak[name] += 1
        self.up_streak[name] = 0
      else:
        self.up_streak[name] = 0
        self.down_streak[name] = 0
      if self.up_streak[name] >= self.patience:
        decisions[name] = 1
        self.up_streak[name] = 0
      elif self.down_streak[name] >= self.down_patience:
        decisions[name] = -1
        self.down_streak[name] = 0
    self.updates += 1
    return decisions

  def state_dict(self):
    return {"version": 1, "groups": list(self.groups),
            "up_threshold": self.up_threshold, "down_threshold": self.down_threshold,
            "patience": self.patience, "down_patience": self.down_patience,
            "min_samples": self.min_samples, "updates": self.updates,
            "up_streak": dict(self.up_streak), "down_streak": dict(self.down_streak),
            "last_window_quality": dict(self.last_window_quality)}

  def load_state_dict(self, state):
    if (not isinstance(state, dict) or state.get("version") != 1
        or tuple(state.get("groups", ())) != self.groups):
      raise ValueError("Incompatible trajectory scheduler state")
    if (state.get("up_threshold") != self.up_threshold
        or state.get("down_threshold") != self.down_threshold
        or state.get("patience") != self.patience
        or state.get("down_patience") != self.down_patience
        or state.get("min_samples") != self.min_samples):
      raise ValueError("Incompatible trajectory scheduler configuration")
    updates = state.get("updates", 0)
    if not isinstance(updates, int) or updates < 0:
      raise ValueError("Invalid trajectory scheduler update count")
    streaks = {}
    for key, limit in (("up_streak", self.patience), ("down_streak", self.down_patience)):
      values = state.get(key, {})
      if set(values) != set(self.groups):
        raise ValueError("Incompatible trajectory scheduler groups")
      restored = {}
      for name, value in values.items():
        if not isinstance(value, int) or not 0 <= value <= limit:
          raise ValueError("Invalid trajectory scheduler streak")
        restored[name] = value
      streaks[key] = restored
    quality = state.get("last_window_quality", {})
    if set(quality) != set(self.groups):
      raise ValueError("Incompatible trajectory scheduler groups")
    for name, value in quality.items():
      if value is not None and (not isinstance(value, (int, float))
                                or not math.isfinite(value) or not 0. <= value <= 1.):
        raise ValueError("Invalid trajectory scheduler window quality")
    for value in self._pending.values():
      value.clear()
    self.updates = updates
    self.up_streak = streaks["up_streak"]
    self.down_streak = streaks["down_streak"]
    self.last_window_quality = dict(quality)


def _max_episode_steps(env):
  """Longest control-step horizon available on real and fake environments."""
  value = getattr(env, "max_episode_length", None)
  if value is not None:
    return max(float(value), 1.)
  cfg = getattr(env, "cfg", None)
  seconds = getattr(cfg, "episode_length_s", None) if cfg is not None else None
  step_dt = getattr(env, "step_dt", None)
  if seconds is not None and step_dt:
    return max(float(seconds)/float(step_dt), 1.)
  buffer = getattr(env, "episode_length_buf", None)
  if buffer is not None and buffer.numel():
    return max(float(buffer.max()), 1.)
  return 1.


def survival_safety_margin(env, env_ids=None, *, success_weight=.6, survival_weight=.4):
  """Sensor-free safety proxy in ``[0, 1]`` for completed episodes.

  A passage episode is "safe" when it succeeded and when it survived a large
  fraction of the timeout horizon before terminating.  Without perception or
  contact sensors this is the cheapest signal that separates a rare unsafe
  rollout from an average success; it is deliberately pessimistic for failed
  episodes (margin ``0``).
  """
  from .passage import failed, passage_success  # local import avoids a cycle

  success_weight, survival_weight = float(success_weight), float(survival_weight)
  if (not math.isfinite(success_weight) or not math.isfinite(survival_weight)
      or success_weight < 0. or survival_weight < 0.
      or success_weight + survival_weight <= 0.):
    raise ValueError("Safety margin weights must be finite, non-negative and non-zero")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=getattr(env, "device", "cpu"))
  env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                            device=getattr(env, "device", "cpu")).flatten()
  if len(env_ids) == 0:
    return torch.zeros(0, device=getattr(env, "device", "cpu"))
  buffer = getattr(env, "episode_length_buf", None)
  if buffer is None:
    survival = torch.zeros(len(env_ids), device=env_ids.device)
  else:
    survival = (buffer[env_ids].float()/_max_episode_steps(env)).clamp(0., 1.)
  success = passage_success(env)[env_ids].float()
  failure = failed(env)[env_ids].float()
  margin = (success_weight*success + survival_weight*survival)
  margin = margin/(success_weight+survival_weight)
  return (margin*(1.-failure)).clamp(0., 1.)
