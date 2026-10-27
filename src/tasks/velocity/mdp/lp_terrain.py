"""Reset-time terrain task sampling for mjlab; actor observations stay unchanged."""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase

from src.curriculum.lp import LearningProgressSampler, build_terrain_tasks


def _canonical(value):
  if is_dataclass(value) and not isinstance(value, type):
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}}
  if isinstance(value, dict):
    return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
  if isinstance(value, (tuple, list)):
    return [_canonical(v) for v in value]
  if isinstance(value, slice):
    return {"slice": [value.start, value.stop, value.step]}
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, Path):
    return str(value)
  raise ValueError(f"Unsupported curriculum signature value: {type(value).__name__}")


class LPTerrainCurriculum(ManagerTermBase):
  """One terrain task per episode; signed LP or uniform task-space baseline.

  Runs before scene/reward/metric reset. It owns row/type/origin selection and
  requires fixed command ranges; a separate global speed expander is excluded.
  """

  def __init__(self, cfg, env):
    super().__init__(env)
    params = cfg.params
    self.terrain = env.scene.terrain
    generator = self.terrain.cfg.terrain_generator if self.terrain is not None else None
    if generator is None or not generator.curriculum:
      raise ValueError("LP terrain sampling requires a curriculum generator grid")
    if "command_vel" in env.cfg.curriculum or "terrain_levels" in env.cfg.curriculum:
      raise ValueError("LP terrain sampling cannot share control with manual terrain/speed curricula")
    self.tasks = build_terrain_tasks(generator)
    self.mode = params.get("mode", "lp_acrl")
    self.sampler_options = {
      "mode": self.mode, "temperature": float(params.get("temperature", 2.0)),
      "uniform_mix": float(params.get("uniform_mix", 0.1)),
      "min_episodes": params.get("min_episodes", 32), "seed": int(env.cfg.seed or 0),
    }
    self.sampler = LearningProgressSampler(len(self.tasks), **self.sampler_options)
    interval = float(params.get("update_interval_s", 100.0))
    if not math.isfinite(interval) or interval <= 0:
      raise ValueError("LP update_interval_s must be positive")
    self.check_interval_steps = max(1, math.ceil(interval / env.step_dt))
    self.last_check_step = int(env.common_step_counter)
    self.env_task_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    self.failed_counts = np.zeros(len(self.tasks), dtype=np.int64)
    self.progress_sums = np.zeros(len(self.tasks), dtype=np.float64)
    self.distance_sums = np.zeros(len(self.tasks), dtype=np.float64)
    self.last_failure = 0.0
    self.last_progress = 0.0
    self.skipped_partial_episodes = 0
    self.status_path = Path(params["status_path"]) if params.get("status_path") else None
    signature = {"tasks": self.tasks, "generator": generator,
                 "rewards": env.cfg.rewards, "command": env.cfg.commands["twist"],
                 "step_dt": env.step_dt, "episode_s": env.max_episode_length_s,
                 "interval_steps": self.check_interval_steps,
                 "sampler": {k: v for k, v in self.sampler_options.items() if k != "seed"}}
    self.signature = hashlib.sha256(json.dumps(_canonical(signature), sort_keys=True,
                                              allow_nan=False).encode()).hexdigest()
    self.fixed_ranges = _canonical(env.cfg.commands["twist"].ranges)
    self.write_status()

  def __call__(self, env, env_ids, **params):
    del params
    if env_ids is None or isinstance(env_ids, slice):
      selection = slice(None) if env_ids is None else env_ids
      env_ids = torch.arange(self.num_envs, device=self.device)[selection]
    if not len(env_ids):
      return self._metrics()
    # Read old episode ownership and sums BEFORE assigning the next tile.
    completed = env_ids[(env.episode_length_buf[env_ids] > 0) & (self.env_task_ids[env_ids] >= 0)]
    if env.common_step_counter > 0 and len(completed):
      time_outs = getattr(env.termination_manager, "time_outs", None)
      if time_outs is not None:
        ended = env.termination_manager.terminated[completed] | time_outs[completed]
        completed = completed[ended]
      # RSL-RL may randomize the initial episode-length buffer to desynchronize
      # timeouts. Those shortened first episodes are not comparable LP returns.
      actual_steps = getattr(env.metrics_manager, "_step_count", None)
      if actual_steps is not None:
        full = actual_steps[completed] == env.episode_length_buf[completed]
        previous_count = len(completed)
        completed = completed[full]
        self.skipped_partial_episodes += previous_count - len(completed)
    if env.common_step_counter > 0 and len(completed):
      metric_sums = env.metrics_manager._episode_sums
      if not {"tracking_progress", "commanded_distance"}.issubset(metric_sums):
        raise ValueError("LP terrain sampling requires progress and distance metrics")
      old_tasks = self.env_task_ids[completed].detach().cpu().numpy()
      returns = torch.stack([values[completed] for values in env.reward_manager._episode_sums.values()]).sum(0)
      failed = env.termination_manager.terminated[completed].detach().cpu().numpy()
      progress = metric_sums["tracking_progress"][completed].detach().cpu().numpy()
      distance = metric_sums["commanded_distance"][completed].detach().cpu().numpy()
      if (not np.isfinite(progress).all() or not np.isfinite(distance).all()
          or (progress < 0).any() or (distance < 0).any()):
        raise ValueError("LP terrain sampling received invalid progress/distance metrics")
      self.sampler.observe(old_tasks, returns.detach().cpu().numpy())
      np.add.at(self.failed_counts, old_tasks, failed.astype(np.int64))
      np.add.at(self.progress_sums, old_tasks, progress)
      np.add.at(self.distance_sums, old_tasks, distance)
      self.last_failure = float(failed.mean())
      self.last_progress = float(progress.sum() / max(float(distance.sum()), 1e-6))
    if env.common_step_counter - self.last_check_step >= self.check_interval_steps:
      self.sampler.update()
      self.last_check_step = int(env.common_step_counter)
      self.write_status()
    task_ids = self.sampler.sample(len(env_ids))
    rows = np.asarray([self.sampler.rng.choice(self.tasks[i]["rows"]) for i in task_ids])
    cols = np.asarray([self.sampler.rng.choice(self.tasks[i]["columns"]) for i in task_ids])
    self.env_task_ids[env_ids] = torch.as_tensor(task_ids, device=self.device)
    self.terrain.terrain_levels[env_ids] = torch.as_tensor(rows, device=self.device)
    self.terrain.terrain_types[env_ids] = torch.as_tensor(cols, device=self.device)
    self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[
      self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
    ]
    return self._metrics()

  def _metrics(self):
    p = self.sampler.probabilities
    return {"level_mean": self.terrain.terrain_levels.float().mean(),
            "mode_is_lp": float(self.mode == "lp_acrl"), "task_count": len(self.tasks),
            "updates": self.sampler.updates, "episodes": int(self.sampler.counts.sum()),
            "ready_tasks": int(self.sampler.known.sum()),
            "skipped_partial_episodes": self.skipped_partial_episodes,
            "sampling_entropy": float(-(p * np.log(p.clip(1e-300))).sum()),
            "reset_batch_failure_fraction": self.last_failure, "tracking_ratio": self.last_progress}

  def state_dict(self):
    return {"version": 1, "signature": self.signature,
            "sampler": self.sampler.state_dict(), "last_check_step": self.last_check_step,
            "failed_counts": self.failed_counts.tolist(), "progress_sums": self.progress_sums.tolist(),
            "distance_sums": self.distance_sums.tolist()}

  def load_state_dict(self, state):
    if state.get("version") != 1 or state.get("signature") != self.signature:
      warnings.warn("LP task/reward/command configuration changed; starting fresh sampler statistics")
      self.sampler = LearningProgressSampler(len(self.tasks), **self.sampler_options)
      self.failed_counts.fill(0)
      self.progress_sums.fill(0)
      self.distance_sums.fill(0)
      self.last_check_step = int(self._env.common_step_counter)
      self.env_task_ids.fill_(-1)
      self.write_status()
      return False
    restored = LearningProgressSampler(len(self.tasks), **self.sampler_options)
    restored.load_state_dict(state["sampler"])
    diagnostics = {}
    for name in ("failed_counts", "progress_sums", "distance_sums"):
      value = np.asarray(state.get(name), dtype=np.float64)
      if value.shape != (len(self.tasks),) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"Invalid LP terrain checkpoint: {name}")
      diagnostics[name] = value
    if (diagnostics["failed_counts"] > restored.counts).any():
      raise ValueError("Invalid LP failure count")
    if (diagnostics["failed_counts"] != np.floor(diagnostics["failed_counts"])).any():
      raise ValueError("Invalid LP failure count")
    if (diagnostics["progress_sums"] > diagnostics["distance_sums"] +
        np.maximum(1., diagnostics["distance_sums"]) * 1e-5).any():
      raise ValueError("Invalid LP progress distance")
    step = state.get("last_check_step", -1)
    if not isinstance(step, int) or step < 0:
      raise ValueError("Invalid LP checkpoint control step")
    self.sampler = restored
    for name, value in diagnostics.items():
      getattr(self, name)[:] = value
    self.last_check_step = min(step, int(self._env.common_step_counter))
    # Physics episodes are not restored by OnPolicyRunner. Reassign only on reset.
    self.env_task_ids.fill_(-1)
    self.last_failure = self.last_progress = 0.0
    self.write_status()
    return True

  def write_status(self):
    if self.status_path is None:
      return
    try:
      self._write_status()
    except OSError as error:
      if not getattr(self, "_status_warning", False):
        warnings.warn(f"Cannot write LP monitoring file: {error}")
        self._status_warning = True

  def _write_status(self):
    self.status_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "mode": self.mode, "signature": self.signature,
            "updated": datetime.now(timezone.utc).isoformat(),
            "control_step": int(self._env.common_step_counter),
            "updates": self.sampler.updates, "fixed_velocity_ranges": self.fixed_ranges,
            "tasks": []}
    for i, task in enumerate(self.tasks):
      count = int(self.sampler.counts[i])
      data["tasks"].append({**task, "probability": float(self.sampler.probabilities[i]),
                            "learning_progress": float(self.sampler.learning_progress[i]),
                            "mean_return": float(self.sampler.mean_returns[i]) if self.sampler.known[i] else None,
                            "samples": count, "pending_samples": int(self.sampler.window_counts[i]),
                            "failure_fraction": float(self.failed_counts[i] / count) if count else None,
                            "progress_ratio": float(self.progress_sums[i] / self.distance_sums[i])
                            if self.distance_sums[i] > 0 else None})
    temp = self.status_path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(self.status_path)
