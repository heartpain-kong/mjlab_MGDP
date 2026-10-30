"""Signed learning-progress task sampling, independently implemented from Eq. 6–7.

Source: https://arxiv.org/html/2601.17428v1#S3.SS2
Baseline mixing, smoothing and probability caps are explicit engineering options.
"""

from copy import deepcopy
import math

import numpy as np


class LearningProgressSampler:
  def __init__(self, num_tasks, *, mode="lp_acrl", temperature=2.0,
               uniform_mix=0.1, min_episodes=32, seed=42, baseline_probabilities=None,
               progress_scale=1.0, progress_alpha=1.0, probability_alpha=1.0,
               max_task_probability=1.0):
    if int(num_tasks) != num_tasks or num_tasks < 1:
      raise ValueError("num_tasks must be a positive integer")
    if mode not in {"uniform", "lp_acrl"}:
      raise ValueError("Unknown terrain sampling mode")
    if not math.isfinite(temperature) or temperature <= 0:
      raise ValueError("LP temperature must be positive")
    if not math.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1:
      raise ValueError("LP uniform_mix must be between 0 and 1")
    if int(min_episodes) != min_episodes or min_episodes < 1:
      raise ValueError("LP min_episodes must be a positive integer")
    for name, value in (("progress_alpha", progress_alpha), ("probability_alpha", probability_alpha)):
      if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"LP {name} must be in (0, 1]")
    if not math.isfinite(progress_scale) or progress_scale <= 0:
      raise ValueError("LP progress_scale must be positive")
    if not math.isfinite(max_task_probability) or not 1 / num_tasks <= max_task_probability <= 1:
      raise ValueError("LP max_task_probability must allow a normalized distribution")
    self.num_tasks = int(num_tasks)
    self.mode = mode
    self.temperature = float(temperature)
    self.uniform_mix = float(uniform_mix)
    self.min_episodes = int(min_episodes)
    baseline = np.full(self.num_tasks, 1 / self.num_tasks) if baseline_probabilities is None else np.asarray(baseline_probabilities, dtype=float)
    if (baseline.shape != (self.num_tasks,) or not np.isfinite(baseline).all()
        or (baseline < 0).any() or not np.isclose(baseline.sum(), 1., rtol=0, atol=1e-10)):
      raise ValueError("Invalid LP baseline probabilities")
    if baseline.max() > max_task_probability + 1e-12:
      raise ValueError("LP task cap must include the baseline distribution")
    self.baseline_probabilities = baseline.copy()
    self.progress_scale = float(progress_scale)
    self.progress_alpha = float(progress_alpha)
    self.probability_alpha = float(probability_alpha)
    self.max_task_probability = float(max_task_probability)
    self.rng = np.random.default_rng(seed)
    self.probabilities = baseline.copy()
    self.counts = np.zeros(self.num_tasks, dtype=np.int64)
    self.window_counts = np.zeros(self.num_tasks, dtype=np.int64)
    self.window_returns = np.zeros(self.num_tasks, dtype=np.float64)
    self.mean_returns = np.zeros(self.num_tasks, dtype=np.float64)
    self.known = np.zeros(self.num_tasks, dtype=bool)
    self.learning_progress = np.zeros(self.num_tasks, dtype=np.float64)
    self.smoothed_progress = np.zeros(self.num_tasks, dtype=np.float64)
    self.updates = 0

  def observe(self, task_ids, returns, *, include_in_window=None):
    raw_ids = np.asarray(task_ids)
    values = np.asarray(returns, dtype=np.float64)
    if raw_ids.ndim != 1 or values.shape != raw_ids.shape:
      raise ValueError("Episode task IDs and returns must be matching vectors")
    if not np.issubdtype(raw_ids.dtype, np.integer):
      raise ValueError("Episode task IDs must be integers")
    ids = raw_ids.astype(np.int64)
    if (ids < 0).any() or (ids >= self.num_tasks).any() or not np.isfinite(values).all():
      raise ValueError("Invalid episode task IDs or nonfinite returns")
    include = np.ones(len(ids), dtype=bool) if include_in_window is None else np.asarray(include_in_window)
    if include.shape != ids.shape or include.dtype != np.bool_:
      raise ValueError("LP window membership must be a matching boolean vector")
    np.add.at(self.counts, ids, 1)
    np.add.at(self.window_counts, ids[include], 1)
    np.add.at(self.window_returns, ids[include], values[include])

  def update(self):
    eligible = self.window_counts >= self.min_episodes
    if not eligible.any():
      return False
    means = self.window_returns[eligible] / self.window_counts[eligible]
    # A first observation establishes a baseline, not progress from zero.
    delta = np.zeros(self.num_tasks, dtype=np.float64)
    ids = np.flatnonzero(eligible)
    established = self.known[ids]
    delta[ids[established]] = means[established] - self.mean_returns[ids[established]]
    self.mean_returns[ids] = means
    self.known[ids] = True
    self.learning_progress[:] = delta
    # Keep the signed LP observable. Smooth only the signal that controls sampling.
    self.smoothed_progress *= 1 - self.progress_alpha
    self.smoothed_progress += self.progress_alpha * delta / self.progress_scale
    self.window_counts[ids] = 0
    self.window_returns[ids] = 0.0
    if self.mode == "lp_acrl":
      # Subtract before dividing: a tiny temperature must not create inf-inf.
      with np.errstate(over="ignore"):
        weights = np.exp((self.smoothed_progress - self.smoothed_progress.max()) / self.temperature)
      weights /= weights.sum()
      floor = self.uniform_mix * self.baseline_probabilities
      target = floor + (1 - self.uniform_mix) * weights
      # Redistribute excess without weakening any baseline floor. A cap on the
      # final probabilities also bounds a task with an extreme LP outlier.
      for _ in range(self.num_tasks + 1):
        excess = np.maximum(target - self.max_task_probability, 0.).sum()
        target = np.minimum(target, self.max_task_probability)
        if excess <= 1e-14:
          break
        room = np.maximum(self.max_task_probability - target, 0.)
        target += excess * room / room.sum()
      self.probabilities *= 1 - self.probability_alpha
      self.probabilities += self.probability_alpha * target
    else:
      self.probabilities[:] = self.baseline_probabilities
    self.updates += 1
    return True

  def sample(self, count):
    return self.rng.choice(self.num_tasks, size=count, replace=True, p=self.probabilities)

  def state_dict(self):
    arrays = ("probabilities", "counts", "window_counts", "window_returns",
              "mean_returns", "known", "learning_progress", "smoothed_progress")
    return {
      "version": 2, "num_tasks": self.num_tasks, "mode": self.mode,
      "temperature": self.temperature, "uniform_mix": self.uniform_mix,
      "min_episodes": self.min_episodes, "updates": self.updates,
      "baseline_probabilities": self.baseline_probabilities.tolist(),
      "progress_scale": self.progress_scale, "progress_alpha": self.progress_alpha,
      "probability_alpha": self.probability_alpha, "max_task_probability": self.max_task_probability,
      "rng_state": deepcopy(self.rng.bit_generator.state),
      **{key: getattr(self, key).tolist() for key in arrays},
    }

  def load_state_dict(self, state):
    for key, expected in (("version", 2), ("num_tasks", self.num_tasks), ("mode", self.mode),
                          ("temperature", self.temperature), ("uniform_mix", self.uniform_mix),
                          ("min_episodes", self.min_episodes),
                          ("baseline_probabilities", self.baseline_probabilities.tolist()),
                          ("progress_scale", self.progress_scale), ("progress_alpha", self.progress_alpha),
                          ("probability_alpha", self.probability_alpha), ("max_task_probability", self.max_task_probability)):
      if state.get(key) != expected:
        raise ValueError(f"Incompatible LP sampler checkpoint: {key}")
    arrays = {}
    for key in ("probabilities", "counts", "window_counts", "window_returns",
                "mean_returns", "known", "learning_progress", "smoothed_progress"):
      value = np.asarray(state.get(key), dtype=np.float64)
      if value.shape != (self.num_tasks,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid LP checkpoint vector: {key}")
      if key in {"counts", "window_counts", "known"}:
        if (value < 0).any() or (value >= 2**63).any() or (value != np.floor(value)).any():
          raise ValueError(f"Invalid LP checkpoint counts: {key}")
        if key == "known" and (value > 1).any():
          raise ValueError("Invalid LP checkpoint baseline flags")
      arrays[key] = value
    probs = arrays["probabilities"]
    if (probs < 0).any() or (probs > 1).any() or not np.isclose(probs.sum(), 1.0, rtol=0, atol=1e-10):
      raise ValueError("Invalid LP checkpoint probabilities")
    if self.mode == "lp_acrl" and ((probs > self.max_task_probability + 1e-10).any()
        or (probs < self.uniform_mix * self.baseline_probabilities - 1e-10).any()):
      raise ValueError("LP checkpoint probabilities violate sampling bounds")
    if (arrays["window_counts"] > arrays["counts"]).any():
      raise ValueError("Invalid LP checkpoint episode counts")
    if ((arrays["window_counts"] == 0) & (arrays["window_returns"] != 0)).any():
      raise ValueError("Invalid LP checkpoint empty window returns")
    updates = state.get("updates", -1)
    if not isinstance(updates, int) or updates < 0:
      raise ValueError("Invalid LP checkpoint update count")
    rng = np.random.default_rng()
    try:
      rng.bit_generator.state = deepcopy(state["rng_state"])
    except (KeyError, TypeError, ValueError) as error:
      raise ValueError("Invalid LP checkpoint random state") from error
    for key, value in arrays.items():
      getattr(self, key)[:] = value
    self.rng = rng
    self.updates = updates


def build_terrain_tasks(generator):
  """Match mjlab 1.2 curriculum column allocation, then group equivalent rows."""
  names = list(generator.sub_terrains)
  configs = list(generator.sub_terrains.values())
  proportions = np.asarray([cfg.proportion for cfg in configs], dtype=np.float64)
  if (not np.isfinite(proportions).all() or (proportions < 0).any()
      or proportions.sum() <= 0 or generator.num_rows < 1 or generator.num_cols < 1):
    raise ValueError("Invalid terrain grid/proportions for task sampling")
  cumulative = np.cumsum(proportions / proportions.sum())
  columns = [[] for _ in names]
  for col in range(generator.num_cols):
    match = np.flatnonzero(col / generator.num_cols + 0.001 < cumulative)
    if not len(match):
      raise ValueError("Terrain column cannot be mapped to a generator type")
    columns[int(match[0])].append(col)
  tasks = []
  for name, config, cols in zip(names, configs, columns, strict=True):
    if not cols:
      continue
    independent = type(config).__name__ in {"BoxFlatTerrainCfg", "HfRandomUniformTerrainCfg"}
    row_groups = [tuple(range(generator.num_rows))] if independent else [
      (row,) for row in range(generator.num_rows)
    ]
    for rows in row_groups:
      level = "all" if independent else str(rows[0])
      tasks.append({"name": f"{name}/row_{level}", "terrain": name,
                    "rows": list(rows), "columns": cols})
  return tasks


def terrain_baseline_probabilities(tasks, generator):
  """Preserve requested type proportions; divide each type across its tasks."""
  names = {task["terrain"] for task in tasks}
  masses = {name: float(generator.sub_terrains[name].proportion) for name in names}
  total = sum(masses.values())
  counts = {name: sum(task["terrain"] == name for task in tasks) for name in names}
  return [masses[task["terrain"]] / total / counts[task["terrain"]] for task in tasks]
