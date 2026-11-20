"""Teacher-student DAgger distillation for the LgDog1 high-wall task.

Motivation (CMU Extreme Parkour style)
--------------------------------------
The high-wall course currently has ~0% success.  Pure RL cannot break through:
the blind (depth-only) policy never sees a successful wall crossing, so it has
no gradient signal to learn the jump.  The teacher-student trick builds a
*privileged teacher* which additionally sees the 187-dim ``height_scan``
(mjlab puts that term in the ``critic`` observation group today), lets RL learn
the skill with privileged information, and then distils the teacher's actions
into the student with DAgger (data comes from the *student's* own state
distribution, labels come from the teacher).

Design: one environment, input masks
------------------------------------
Two simulators (privileged teacher env + student env) have to be kept in sync,
which is fragile at mjlab scale.  Instead this module keeps **one** env whose
flat observation vector contains both modalities::

    flat_obs = [ student_part (student_dim) | privileged_part (privileged_dim) ]

For the wall task (``LgDog1-Race-MGDP-WallTraining``)::

    student_part    = 188 proprio + 64 depth feature = 252   (exactly today's policy input)
    privileged_part = 187 height scan
    total           = 439

``build_distill_env`` appends the ``height_scan`` term (copied from the
``critic`` group) to the **actor** observation group.  The actor group has a
four-frame proprio history, so the group-level history is moved down to the
individual terms first; without that the height scan would be expanded to
4 x 187 dimensions.  ``DistillEnv`` then re-orders the raw observation groups
into the flat layout above.  Teacher *RL training* uses
:func:`register_teacher_task` / :func:`build_teacher_env_cfg` instead, which adds
the scan as a separate ``"privileged"`` group so the MGDP history/perception
contracts (and ``--warm-start-checkpoint``) keep working; both configurations
produce the same flat observation.

``MaskedActor`` is a plain MLP actor (the same ``rsl_rl.modules.MLP`` /
``EmpiricalNormalization`` modules the rsl_rl ``MLPModel`` uses, so
``actor_state_dict`` checkpoints stay loadable) that zeroes one half of the
input according to the role it is asked to play:

* teacher: sees ``privileged_part`` (by design; ``teacher_sees_student=True``
  switches to "teacher sees everything", which is what a teacher trained with
  plain PPO on the height-augmented task actually expects -- see the note below);
* student: sees ``student_part`` only.

Because the student is a *fixed-shape* 439-input network whose privileged input
slots are always zero, the trained student can be exported as a normal 252-dim
rsl_rl actor by dropping the last layer-0 columns and the last normalizer
entries (``MaskedActor.deployable_state_dict``).  The 252-dim order is identical
to today's wall policy input, so the exported student can be loaded with the
existing wall task checkpoint machinery.

IMPORTANT note about the teacher input mask
-------------------------------------------
The design brief asks the teacher to see *only* ``privileged_part``.  That is
implemented as the ``MaskedActor`` default (``teacher_sees_student=False``) and
is what the unit tests check.  A teacher that cannot see the velocity command /
proprioception can only learn a near-useless mapping, though.  A teacher that
is meant to be useful in practice should be trained by RL on exactly the
height-augmented task registered by :func:`register_teacher_task`, which means
it was trained on the full 439-dim vector; use ``teacher_sees_student=True``
(``scripts/distill_wall.py --teacher-input full``, the CLI default) in that case.
Loading the checkpoint is identical for both modes because both networks have
the same 439-dim input; only the distillation-time mask changes.

Per-lesson teachers (``--lesson``)
----------------------------------
By default the teacher sees the whole ten-lesson wall course, which mixes a
36 cm wall with stepping stones, beams, tyres and so on.  Some lessons are much
easier to learn in isolation because their terrain is fixed and their difficulty
is concentrated (``high_wall`` and ``stepping_stones`` are the two the WMP review
asked for).  Every cfg builder therefore accepts ``lesson=<name>``: it filters
``cfg.scene.terrain.terrain_generator.sub_terrains`` down to that single lesson
and sets its ``proportion=1.0``::

    register_teacher_task(lesson="high_wall")
    # -> "LgDog1-Race-MGDP-Wall-Teacher-high_wall"  (lesson=None keeps "...-Teacher")

    build_teacher_env_cfg(WALL_TASK_ID, lesson="high_wall")
    build_distill_env(WALL_TASK_ID, lesson="high_wall", ...)

    python scripts/distill_wall.py --lesson high_wall --teacher ... --out ...
    python scripts/distill_wall.py --list-lessons   # lessons + teacher task ids

Pinning the difficulty stage (``level``)
----------------------------------------
The terrain curriculum ramps difficulty over eight rows, so a freshly sampled
teacher spends most envs on the easy stages.  Every cfg builder and
``register_teacher_task`` therefore accepts ``level=1..8``: it sets
``generator.num_rows = 1`` and ``generator.difficulty_range = (d, d)`` with
``d = (level - 0.5) / 8`` (the mid-point of the 8-stage curriculum, so
``stage = int(d * 8) == level - 1``), i.e. every env is generated at exactly that
stage::

    register_teacher_task(lesson="high_wall", level=8)   # competition stage, pinned
    build_teacher_env_cfg(WALL_TASK_ID, lesson="high_wall", level=8)
    build_distill_env(WALL_TASK_ID, lesson="high_wall", level=8, ...)

    python scripts/distill_wall.py --lesson high_wall --teacher-level 8 ...

The printed summary is ``[distill] level=8 difficulty=0.9375``.  ``level=None``
(and the CLI default) keeps the ramping curriculum; the task id rule is
unchanged, so the same teacher task id is reused for a pinned level.

``lesson=None`` (or the CLI's ``--lesson all``) keeps the current ten-lesson
behaviour, so the existing task id / checkpoints remain valid.  Known
limitation: the wall *course* managers still hard-require several lesson names
(``BodyAwarePassageCommand`` needs ``high_wall`` + ``slalom_brick``,
``WallGeometry`` rewards need ``high_wall``, ``StoneGroundContact`` needs
``stepping_stones``, ``SlalomBodyProjection`` needs ``slalom_brick``), so a
single-lesson env config still has to have those terms pruned before the env can
be constructed end-to-end; the filter in this module only reshapes the terrain
roster.

How to get the teacher weights
------------------------------
The existing wall checkpoints (e.g. ``.../lgdog1_velocity_race_mgdp_wall_v5/``
``model_*.pt``) contain ``actor_state_dict`` for the 252-dim blind actor.  Those
weights are *not* a privileged teacher yet: the height columns do not exist.
The supported recipe is:

1. Expand the blind actor to the teacher width (the privileged columns are
   appended zeroed, so the teacher initially ignores the height scan)::

       from src.tasks.velocity.rl.distill import (
           DistillLayout, expand_blind_actor_state_dict, extract_actor_state_dict,
           save_teacher_init_checkpoint,
       )
       state_dict, _ = extract_actor_state_dict(".../model_10000.pt")
       layout = DistillLayout(student_dim=252, privileged_dim=187, actor_student_dim=188)
       expanded = expand_blind_actor_state_dict(state_dict, layout)
       save_teacher_init_checkpoint("logs/distill/teacher_init.pt", expanded,
                                    source_checkpoint=".../model_10000.pt")

   (``scripts/distill_wall.py --teacher <blind ckpt> --write-teacher-init <path>``
   does the same thing after building the env, i.e. with the real layout.)

2. RL fine-tune the teacher on the height-augmented task so the height columns
   become non-zero::

       from src.tasks.velocity.rl.distill import register_teacher_task
       register_teacher_task()   # registers "LgDog1-Race-MGDP-Wall-Teacher"
       # scripts/train.py imports src.tasks but not this module, so register the
       # task in the same process and then run train.py's main():
       #   python -c "import sys, runpy; import src.tasks; \
       #     from src.tasks.velocity.rl.distill import register_teacher_task; \
       #     register_teacher_task(); sys.argv=['scripts/train.py', \
       #     'LgDog1-Race-MGDP-Wall-Teacher']; \
       #     runpy.run_path('scripts/train.py', run_name='__main__')" \
       #     --warm-start-checkpoint logs/distill/teacher_init.pt ...

   The registered teacher task appends the height scan as a *separate*
   observation group (``"privileged"``) instead of appending it to the actor
   group.  The resulting policy input is still ``[proprio | depth | height]``
   (identical to the canonical layout here), but the actor group keeps its
   four-frame history, so the existing MGDP history/perception contracts and
   the ``--warm-start-checkpoint`` path keep working.

3. Distil::  ``python scripts/distill_wall.py --teacher <teacher model_*.pt> ...``

Checkpoint format written by this module
----------------------------------------
``save_student_checkpoint`` writes the usual rsl_rl checkpoint dict::

    actor_state_dict      standard 252-dim actor (mlp.* + obs_normalizer.* +
                          distribution.std_param) -> loadable by scripts/play.py
    critic_state_dict     empty (distillation has no critic)
    optimizer_state_dict  optional Adam state
    iter, infos

plus ``distill_student_state_dict`` (the full 439-dim masked student) and
``distill_meta`` / ``distill_layout`` for resuming / inspection.  The MGDP
perception contract/sidecar of the teacher checkpoint is inherited (and the
sidecar file is copied next to the student checkpoint) so the distilled policy
stays loadable through the existing MGDP task machinery.

Nothing in this module import-mutates the task registry at import time, and the
simulation stack (mjlab) is imported lazily inside :func:`build_distill_env` /
:func:`build_distill_env_cfg` / :func:`register_teacher_task`, so the tensor
utilities below are importable without GPU / MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rsl_rl.modules import EmpiricalNormalization, MLP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_TASK_ID = "LgDog1-Race-MGDP-WallTraining"
"""The high-wall training task this module is written for (single 12-DoF dog)."""

TEACHER_TASK_ID = "LgDog1-Race-MGDP-Wall-Teacher"
"""Base teacher task id (all ten wall lessons), registered by :func:`register_teacher_task`.

A per-lesson teacher appends ``-<lesson>``, e.g.
``LgDog1-Race-MGDP-Wall-Teacher-high_wall`` for ``lesson="high_wall"``.
"""

LESSON_ALL = "all"
"""CLI value meaning "do not filter the lesson roster" (maps to ``lesson=None``)."""

TERRAIN_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8)
"""Difficulty stages a lesson is graded into (1 = easiest, 8 = hardest).

The passage terrains quantise difficulty with ``stage = int(difficulty * 8)``
(``passage_terrain_cfg.py``), and the wall course has one row per stage, so the
1-based stage ``n`` spans ``[(n - 1) / 8, n / 8)`` with mid-point
``(n - 0.5) / 8``.
"""

HEIGHT_SCAN_TERM = "height_scan"
"""Observation term holding the 187-dim terrain height scan (in the critic group)."""

HEIGHT_SCAN_GROUP = "actor"
"""Actor group the privileged height scan is appended to for the *distillation* env."""

TEACHER_PRIVILEGED_GROUP = "privileged"
"""Observation group holding the height scan for *teacher RL training*.

The teacher task keeps the actor group untouched (four-frame proprio history)
and adds the privileged scan as a separate group, so the MGDP observation
history / perception contracts stay valid and an existing 252-dim blind actor
can be warm-started via ``scripts/train.py --warm-start-checkpoint``.  The flat
distillation observation is identical to the one produced by appending the scan
to the actor group (proprio, depth, height), so teacher checkpoints trained this
way load into the canonical layout without any reordering.
"""

ACTION_DIM = 12
"""LgDog1 joint-position action dimension."""

DEFAULT_HIDDEN_DIMS = (512, 256, 128)
"""Hidden layer sizes of the wall actor (see ``lgdog1_ppo_runner_cfg``)."""

DEFAULT_ACTIVATION = "elu"
"""Activation used by every existing LgDog1 checkpoint."""

ROLE_TEACHER = "teacher"
ROLE_STUDENT = "student"

INPUT_ORDER_CANONICAL = "canonical"
"""``[student_part | privileged_part]`` ordering used everywhere in this module."""

INPUT_ORDER_ENV = "env"
"""Ordering produced by plain PPO on the height-augmented task.

The RL policy sees ``obs_groups = ("actor", "perception")``; after appending the
height scan to the actor group its vector is
``[proprio | height_scan | depth_feature]``, while this module's canonical
ordering is ``[proprio | depth_feature | height_scan]``.
"""

DEFAULT_INIT_STD = 0.25
"""Gaussian std written into the exported student so ``scripts/play.py`` can build
a stochastic ``MLPModel`` around it (the value is irrelevant for playback)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _prod(values: Sequence[int]) -> int:
  total = 1
  for value in values:
    total *= int(value)
  return total


def _to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
  """Detach/copy tensors to CPU so saved checkpoints are device independent."""
  result: dict[str, Any] = {}
  for key, value in state_dict.items():
    if torch.is_tensor(value):
      result[key] = value.detach().cpu().clone()
    else:
      result[key] = value
  return result


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillLayout:
  """Dimensions of the flat ``[student_part | privileged_part]`` observation.

  Attributes:
    student_dim: Width of the student part (no privileged information).
    privileged_dim: Width of the privileged part (height scan).
    actor_student_dim: Width of the student part that comes from the same
      observation group as the privileged term (the proprioception history).
      Needed to map a checkpoint trained with plain PPO in the env's natural
      ordering onto the canonical ordering; only relevant for privileged groups
      that are the first policy group (which holds for the wall task).
  """

  student_dim: int
  privileged_dim: int
  actor_student_dim: int = 0
  privileged_group: str = HEIGHT_SCAN_GROUP
  privileged_term: str = HEIGHT_SCAN_TERM

  def __post_init__(self) -> None:
    self._require_positive("student_dim", self.student_dim)
    self._require_positive("privileged_dim", self.privileged_dim)
    if not isinstance(self.actor_student_dim, int) or self.actor_student_dim < 0:
      raise ValueError(
        f"actor_student_dim must be a non-negative int, got {self.actor_student_dim!r}"
      )
    if self.actor_student_dim > self.student_dim:
      raise ValueError(
        f"actor_student_dim ({self.actor_student_dim}) cannot exceed "
        f"student_dim ({self.student_dim})"
      )

  @staticmethod
  def _require_positive(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
      raise ValueError(f"{name} must be a positive int, got {value!r}")

  @property
  def total_dim(self) -> int:
    return self.student_dim + self.privileged_dim

  @property
  def student_slice(self) -> slice:
    return slice(0, self.student_dim)

  @property
  def privileged_slice(self) -> slice:
    return slice(self.student_dim, self.total_dim)

  def check_observation(self, observation: torch.Tensor, name: str = "observation") -> torch.Tensor:
    """Validate the last dimension of ``observation`` and return it unchanged."""
    if not torch.is_tensor(observation):
      raise TypeError(f"{name} must be a torch.Tensor, got {type(observation)!r}")
    if observation.shape[-1] != self.total_dim:
      raise ValueError(
        f"{name} last dim is {observation.shape[-1]}, expected {self.total_dim} "
        f"({self.student_dim} student + {self.privileged_dim} privileged)"
      )
    return observation

  def split(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(student_part, privileged_part)`` views/copies of ``observation``."""
    observation = self.check_observation(observation)
    return (
      observation[..., self.student_slice],
      observation[..., self.privileged_slice],
    )

  def join(self, student_part: torch.Tensor, privileged_part: torch.Tensor) -> torch.Tensor:
    """Concatenate a student and privileged part back into the flat layout."""
    if student_part.shape[:-1] != privileged_part.shape[:-1]:
      raise ValueError(
        "student_part and privileged_part must share leading dims, got "
        f"{tuple(student_part.shape)} vs {tuple(privileged_part.shape)}"
      )
    if student_part.shape[-1] != self.student_dim:
      raise ValueError(
        f"student_part last dim is {student_part.shape[-1]}, expected {self.student_dim}"
      )
    if privileged_part.shape[-1] != self.privileged_dim:
      raise ValueError(
        f"privileged_part last dim is {privileged_part.shape[-1]}, expected {self.privileged_dim}"
      )
    return torch.cat((student_part, privileged_part), dim=-1)

  def role_masks(self, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Return multiplicative masks for the two roles (1 = keep, 0 = masked out)."""
    student_mask = torch.zeros(self.total_dim, dtype=dtype)
    student_mask[: self.student_dim] = 1.0
    teacher_mask = torch.zeros(self.total_dim, dtype=dtype)
    teacher_mask[self.student_dim :] = 1.0
    return {ROLE_STUDENT: student_mask, ROLE_TEACHER: teacher_mask}

  def env_order_permutation(self) -> torch.Tensor:
    """Index vector mapping the env-natural observation order to the canonical one.

    Env-natural (plain PPO on :func:`build_distill_env_cfg`)::

        [actor_student | privileged | other_policy_groups]

    Canonical (this module)::

        [actor_student | other_policy_groups | privileged]

    ``canonical = env_natural[..., permutation]``.  This assumes the privileged
    observation group is the first policy group, which holds for the wall task
    (``obs_groups = ("actor", "perception")``).
    """
    head = self.actor_student_dim
    privileged = self.privileged_dim
    other = self.student_dim - head
    if other < 0:
      raise ValueError("invalid layout: actor_student_dim > student_dim")
    indices = (
      list(range(head))
      + list(range(head + privileged, head + privileged + other))
      + list(range(head, head + privileged))
    )
    permutation = torch.tensor(indices, dtype=torch.long)
    if permutation.numel() != self.total_dim:
      raise ValueError("internal error: permutation length does not match total_dim")
    return permutation

  def describe(self) -> dict[str, Any]:
    return {
      "student_dim": self.student_dim,
      "privileged_dim": self.privileged_dim,
      "total_dim": self.total_dim,
      "actor_student_dim": self.actor_student_dim,
      "privileged_group": self.privileged_group,
      "privileged_term": self.privileged_term,
    }


# ---------------------------------------------------------------------------
# Env cfg construction: append the privileged term to the actor group
# ---------------------------------------------------------------------------


def append_height_scan_to_actor(
  cfg: Any,
  term_name: str = HEIGHT_SCAN_TERM,
  source_group: str = "critic",
  target_group: str = HEIGHT_SCAN_GROUP,
) -> Any:
  """Append a copy of the critic's ``height_scan`` term to the actor group.

  The actor group carries a multi-frame proprio history via the *group-level*
  ``history_length``.  mjlab applies that override to every term of the group,
  which would turn the 187-dim height scan into ``history_length * 187``.  The
  history is therefore pushed down to the individual terms first, so that the
  appended privileged term stays a single 187-dim frame.  The resulting actor
  observation is bit-identical to the old proprio history (same term-major
  ordering) plus the height scan at the end.

  Idempotent: calling it twice returns ``cfg`` unchanged.

  Returns the (mutated) env config.
  """
  observations = getattr(cfg, "observations", None)
  if not isinstance(observations, Mapping):
    raise TypeError("cfg.observations must be a mapping of observation groups")
  if source_group not in observations:
    raise ValueError(
      f"source observation group {source_group!r} not found; available: {sorted(observations)}"
    )
  if target_group not in observations:
    raise ValueError(
      f"target observation group {target_group!r} not found; available: {sorted(observations)}"
    )
  source_terms = observations[source_group].terms
  target_terms = observations[target_group].terms
  if term_name not in source_terms:
    raise ValueError(
      f"term {term_name!r} not found in group {source_group!r}; "
      f"available: {sorted(source_terms)}"
    )
  if term_name in target_terms:
    return cfg  # already augmented

  import copy

  term = copy.deepcopy(source_terms[term_name])
  # Privileged teacher input should be clean terrain heights, not corrupted.
  term.noise = None
  term.delay_min_lag = 0
  term.delay_max_lag = 0
  term.history_length = 0
  term.flatten_history_dim = True
  target_terms[term_name] = term

  group = observations[target_group]
  group_history = getattr(group, "history_length", None)
  if group_history:
    for name, existing in target_terms.items():
      if name == term_name:
        continue
      existing.history_length = int(group_history)
      existing.flatten_history_dim = bool(getattr(group, "flatten_history_dim", True))
    # Per-term history now carries the group's frames; the group override must
    # be disabled or it would also apply to the new privileged term.
    group.history_length = None
  return cfg


def add_privileged_observation_group(
  cfg: Any,
  group_name: str = TEACHER_PRIVILEGED_GROUP,
  term_name: str = HEIGHT_SCAN_TERM,
  source_group: str = "critic",
  group_factory: Any = None,
) -> Any:
  """Add a separate observation group holding the privileged height scan.

  This is what :func:`register_teacher_task` uses: unlike
  :func:`append_height_scan_to_actor` it leaves the ``actor`` group (and its
  four-frame history / MGDP contracts) completely untouched, which keeps
  ``scripts/train.py --warm-start-checkpoint`` and the perception sidecar
  machinery working.  The teacher policy observation is
  ``[actor | perception | privileged]`` = ``[proprio | depth | height]``, i.e.
  exactly this module's canonical flat layout.

  The group is added with ``enable_corruption=False`` and no term history, so a
  single 187-dim height frame is appended.  Idempotent.

  ``group_factory(terms)`` can inject a fake group type in tests; by default
  ``mjlab.managers.observation_manager.ObservationGroupCfg`` is used.
  """
  import copy

  observations = getattr(cfg, "observations", None)
  if not isinstance(observations, Mapping):
    raise TypeError("cfg.observations must be a mapping of observation groups")
  if group_name in observations:
    return cfg  # already added
  if source_group not in observations:
    raise ValueError(
      f"source observation group {source_group!r} not found; available: {sorted(observations)}"
    )
  source_terms = observations[source_group].terms
  if term_name not in source_terms:
    raise ValueError(
      f"term {term_name!r} not found in group {source_group!r}; "
      f"available: {sorted(source_terms)}"
    )
  term = copy.deepcopy(source_terms[term_name])
  term.noise = None
  term.delay_min_lag = 0
  term.delay_max_lag = 0
  term.history_length = 0
  term.flatten_history_dim = True
  terms = {term_name: term}
  if group_factory is None:
    from mjlab.managers.observation_manager import ObservationGroupCfg

    group = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      flatten_history_dim=True,
    )
  else:
    group = group_factory(terms)
  observations[group_name] = group
  return cfg



# ---------------------------------------------------------------------------
# Distill env: raw observation dict -> flat [student | privileged] tensor
# ---------------------------------------------------------------------------


class DistillEnv:
  """Thin adapter turning a mjlab env's observation dict into the flat layout.

  The wrapped manager-based env must provide ``observation_manager.compute()``,
  ``num_envs``, ``device``, ``action_manager.total_action_dim``, ``reset()`` and
  ``step(actions)`` (mjlab's ``ManagerBasedRlEnv`` does; ``train_distill`` only
  needs this small protocol, which is why it can be unit-tested with a fake
  env).
  """

  def __init__(
    self,
    env: Any,
    layout: DistillLayout,
    group_dims: Mapping[str, int],
    student_slices: Sequence[tuple[str, slice]],
    privileged_slice: tuple[str, slice],
  ) -> None:
    self.env = env
    self.layout = layout
    self._group_dims = {name: int(dim) for name, dim in group_dims.items()}
    self._student_slices = tuple(student_slices)
    self._privileged_slice = privileged_slice
    self.num_envs = int(env.num_envs)
    self.device = torch.device(getattr(env, "device", "cpu"))
    self.num_actions = int(env.action_manager.total_action_dim)

  # -- construction --------------------------------------------------------

  @classmethod
  def from_env(
    cls,
    env: Any,
    policy_groups: Sequence[str],
    privileged_group: str = HEIGHT_SCAN_GROUP,
    privileged_term: str = HEIGHT_SCAN_TERM,
  ) -> "DistillEnv":
    """Build the adapter, inferring the split from the env's observation manager.

    ``policy_groups`` is the ordered list of observation groups fed to the
    policy (``rl_cfg.obs_groups["actor"]``, e.g. ``("actor", "perception")``).
    The privileged term must be the last term of ``privileged_group`` and that
    group must come first in ``policy_groups`` (true for the wall task); this
    keeps the privileged part a contiguous tail.
    """
    groups = tuple(str(group) for group in policy_groups)
    if not groups:
      raise ValueError("policy_groups must not be empty")
    if privileged_group not in groups:
      raise ValueError(
        f"privileged group {privileged_group!r} is not in policy groups {groups}"
      )
    if groups[0] != privileged_group:
      raise ValueError(
        "the privileged observation group must be the first policy group, got "
        f"{groups}"
      )
    manager = getattr(env, "observation_manager", None)
    if manager is None:
      raise TypeError("env has no observation_manager; DistillEnv needs mjlab's manager")

    group_dims: dict[str, int] = {}
    for group in groups:
      if not manager.group_obs_concatenate[group]:
        raise ValueError(
          f"observation group {group!r} must use concatenate_terms=True"
        )
      dims = tuple(manager.group_obs_dim[group])
      if len(dims) != 1:
        raise ValueError(
          f"observation group {group!r} is not flat (shape {dims}); "
          "only [batch, width] observations are supported"
        )
      group_dims[group] = int(dims[0])

    terms = list(manager.active_terms[privileged_group])
    if privileged_term not in terms:
      raise ValueError(
        f"term {privileged_term!r} not found in group {privileged_group!r}; "
        f"available: {terms}. Did you call build_distill_env_cfg()?"
      )
    index = terms.index(privileged_term)
    if index != len(terms) - 1:
      raise ValueError(
        f"privileged term {privileged_term!r} must be the last term of "
        f"{privileged_group!r} (got index {index} of {len(terms)})"
      )
    term_dims = [tuple(dim) for dim in manager.group_obs_term_dim[privileged_group]]
    term_widths = [_prod(dim) for dim in term_dims]
    group_width = group_dims[privileged_group]
    if sum(term_widths) != group_width:
      raise ValueError(
        f"term widths {term_widths} of group {privileged_group!r} do not sum to "
        f"its width {group_width}"
      )

    privileged_dim = int(term_widths[index])
    actor_student_dim = group_width - privileged_dim
    other_groups = groups[1:]
    student_dim = actor_student_dim + sum(group_dims[group] for group in other_groups)

    student_slices: list[tuple[str, slice]] = [(privileged_group, slice(0, actor_student_dim))]
    for group in other_groups:
      student_slices.append((group, slice(0, group_dims[group])))
    privileged_slice = (privileged_group, slice(actor_student_dim, group_width))

    layout = DistillLayout(
      student_dim=student_dim,
      privileged_dim=privileged_dim,
      actor_student_dim=actor_student_dim,
      privileged_group=privileged_group,
      privileged_term=privileged_term,
    )
    return cls(
      env=env,
      layout=layout,
      group_dims=group_dims,
      student_slices=student_slices,
      privileged_slice=privileged_slice,
    )

  # -- observation assembly ------------------------------------------------

  def _group_tensor(self, observation: Mapping[str, Any], group: str) -> torch.Tensor:
    try:
      tensor = observation[group]
    except Exception as error:  # TensorDict / dict both raise KeyError-ish
      raise KeyError(
        f"observation group {group!r} missing from env observations: "
        f"{sorted(observation.keys()) if hasattr(observation, 'keys') else observation!r}"
      ) from error
    if not torch.is_tensor(tensor):
      raise TypeError(f"observation group {group!r} is not a tensor: {type(tensor)!r}")
    expected = self._group_dims[group]
    if tensor.shape[-1] != expected:
      raise ValueError(
        f"observation group {group!r} has width {tensor.shape[-1]}, expected {expected}"
      )
    return tensor

  def flatten_observation(self, observation: Mapping[str, Any]) -> torch.Tensor:
    """Return the flat ``[student_part | privileged_part]`` observation tensor."""
    student_parts = [
      self._group_tensor(observation, group)[..., selection]
      for group, selection in self._student_slices
    ]
    if not student_parts:
      raise ValueError("internal error: no student segments configured")
    student = (
      student_parts[0]
      if len(student_parts) == 1
      else torch.cat(student_parts, dim=-1)
    )
    group, selection = self._privileged_slice
    privileged = self._group_tensor(observation, group)[..., selection]
    flat = torch.cat((student, privileged), dim=-1)
    if flat.shape[-1] != self.layout.total_dim:
      raise ValueError(
        f"assembled observation width {flat.shape[-1]} != "
        f"layout total {self.layout.total_dim}"
      )
    return flat

  # -- vec-env protocol ----------------------------------------------------

  def get_observations(self) -> torch.Tensor:
    return self.flatten_observation(self.env.observation_manager.compute())

  def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
    result = self.env.reset()
    if isinstance(result, tuple) and len(result) >= 2:
      observation, extras = result[0], result[1]
    else:
      observation, extras = result, {}
    return self.flatten_observation(observation), extras

  def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    result = self.env.step(actions.to(self.device))
    if len(result) == 5:
      observation, rewards, terminated, truncated, extras = result
      dones = (terminated | truncated).to(dtype=torch.long)
    elif len(result) == 4:
      observation, rewards, dones, extras = result
      dones = dones.to(dtype=torch.long) if torch.is_tensor(dones) else torch.as_tensor(dones)
    else:
      raise ValueError(f"env.step returned {len(result)} values, expected 4 or 5")
    return self.flatten_observation(observation), rewards, dones, extras

  def close(self) -> None:
    close = getattr(self.env, "close", None)
    if callable(close):
      close()

  @property
  def unwrapped(self) -> Any:
    return getattr(self.env, "unwrapped", self.env)

  def __getattr__(self, name: str) -> Any:
    # Delegate everything else (cfg, render, seed, ...) to the wrapped env.
    if name.startswith("_"):
      raise AttributeError(name)
    env = self.__dict__.get("env")
    if env is None:
      raise AttributeError(name)
    return getattr(env, name)


# ---------------------------------------------------------------------------
# Masked actor
# ---------------------------------------------------------------------------


class MaskedActor(nn.Module):
  """MLP actor that masks one half of the flat observation according to its role.

  The network has the standard rsl_rl ``MLPModel`` actor layout
  (``mlp.<i>.weight`` + ``obs_normalizer.{_mean,_var,_std,count}``), so existing
  ``actor_state_dict`` checkpoints can be loaded after shape adaptation
  (:func:`prepare_actor_state_dict`).  The role masks are non-persistent buffers
  and therefore do not appear in ``state_dict()``.

  Args:
    layout: Observation split.
    hidden_dims: MLP hidden layer sizes.
    activation: MLP activation name (``"elu"`` for every LgDog1 checkpoint).
    output_dim: Action dimension (12 for LgDog1).
    obs_normalization: Whether to hold an ``EmpiricalNormalization`` buffer set.
    teacher_sees_student: When True the teacher role receives the *whole* flat
      observation (student + privileged).  Default False follows the design
      brief's complementary masks (teacher sees only the privileged part).
  """

  def __init__(
    self,
    layout: DistillLayout,
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    activation: str = DEFAULT_ACTIVATION,
    output_dim: int = ACTION_DIM,
    obs_normalization: bool = True,
    teacher_sees_student: bool = False,
  ) -> None:
    super().__init__()
    if not isinstance(layout, DistillLayout):
      raise TypeError(f"layout must be a DistillLayout, got {type(layout)!r}")
    hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
      raise ValueError(f"hidden_dims must be positive, got {hidden_dims!r}")
    self.layout = layout
    self.hidden_dims = hidden_dims
    self.activation = str(activation)
    self.output_dim = int(output_dim)
    if self.output_dim <= 0:
      raise ValueError(f"output_dim must be positive, got {self.output_dim!r}")
    self.obs_normalization = bool(obs_normalization)
    self.teacher_sees_student = bool(teacher_sees_student)

    self.mlp = MLP(layout.total_dim, self.output_dim, self.hidden_dims, self.activation)
    self.obs_normalizer: nn.Module = (
      EmpiricalNormalization(layout.total_dim) if self.obs_normalization else nn.Identity()
    )

    masks = layout.role_masks()
    self.register_buffer("_student_mask", masks[ROLE_STUDENT], persistent=False)
    self.register_buffer("_teacher_mask", masks[ROLE_TEACHER], persistent=False)

  # -- masks / forward -----------------------------------------------------

  @property
  def input_dim(self) -> int:
    return self.layout.total_dim

  def masked_obs(self, observation: torch.Tensor, role: str = ROLE_STUDENT) -> torch.Tensor:
    """Zero the part of the observation the given role must not see."""
    observation = self.layout.check_observation(observation)
    role = str(role).lower()
    if role == ROLE_TEACHER:
      if self.teacher_sees_student:
        return observation
      mask = self._teacher_mask
    elif role == ROLE_STUDENT:
      mask = self._student_mask
    else:
      raise ValueError(
        f"unknown role {role!r}; expected {ROLE_TEACHER!r} or {ROLE_STUDENT!r}"
      )
    return observation * mask.to(device=observation.device, dtype=observation.dtype)

  def forward(self, observation: torch.Tensor, role: str = ROLE_STUDENT) -> torch.Tensor:
    latent = self.obs_normalizer(self.masked_obs(observation, role))
    return self.mlp(latent)

  @torch.no_grad()
  def update_normalization(self, observation: torch.Tensor, role: str = ROLE_STUDENT) -> None:
    """Update the running observation statistics from masked observations."""
    if not self.obs_normalization:
      return
    self.obs_normalizer.update(self.masked_obs(observation, role))

  # -- checkpoint helpers --------------------------------------------------

  def parameter_count(self, trainable_only: bool = False) -> int:
    return sum(
      parameter.numel()
      for parameter in self.parameters()
      if (parameter.requires_grad or not trainable_only)
    )

  def deployable_state_dict(self, init_std: float | None = DEFAULT_INIT_STD) -> dict[str, Any]:
    """Return a 252-dim rsl_rl actor state dict for deployment.

    Training always feeds ``0`` into the privileged input slots (the student
    mask zeroes them before normalization; because every observed value is
    zero the normalizer output for those slots is exactly zero as well), so the
    corresponding first-layer columns contribute nothing and can be dropped.
    The remaining ``student_dim`` inputs are exactly today's wall policy input
    (proprio history then depth feature).
    """
    student_dim = self.layout.student_dim
    total_dim = self.layout.total_dim
    state_dict = self.state_dict()
    exported: dict[str, Any] = {}
    for key, value in state_dict.items():
      if key == "mlp.0.weight":
        exported[key] = value[:, :student_dim].detach().cpu().clone()
      elif key.startswith("obs_normalizer.") and value.dim() >= 2 and value.shape[-1] == total_dim:
        exported[key] = value[..., :student_dim].detach().cpu().clone()
      else:
        exported[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    if init_std is not None:
      exported["distribution.std_param"] = torch.full(
        (self.output_dim,), float(init_std), dtype=torch.float32
      )
    return exported

  def deployable_parameter_count(self) -> int:
    return sum(
      value.numel() for value in self.deployable_state_dict(init_std=None).values()
      if torch.is_tensor(value)
    )


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def distill_loss(
  student_action: torch.Tensor,
  teacher_action: torch.Tensor,
  loss: str = "mse",
  huber_delta: float = 1.0,
) -> torch.Tensor:
  """Scalar imitation loss between the student's and teacher's actions.

  Args:
    student_action: Student action tensor (any leading dims, last dim action).
    teacher_action: Teacher action labels, same shape as ``student_action``.
    loss: ``"mse"`` (default) or ``"huber"``.
    huber_delta: Transition point of the Huber loss (``"huber"`` only).

  Returns:
    A 0-dim tensor.
  """
  if not torch.is_tensor(student_action) or not torch.is_tensor(teacher_action):
    raise TypeError("student_action and teacher_action must be torch.Tensors")
  if student_action.shape != teacher_action.shape:
    raise ValueError(
      "student_action and teacher_action shapes differ: "
      f"{tuple(student_action.shape)} vs {tuple(teacher_action.shape)}"
    )
  kind = str(loss).lower()
  if kind == "mse":
    return F.mse_loss(student_action, teacher_action)
  if kind == "huber":
    delta = float(huber_delta)
    if delta <= 0.0:
      raise ValueError(f"huber_delta must be positive, got {delta!r}")
    return F.huber_loss(student_action, teacher_action, delta=delta)
  raise ValueError(f"unknown loss {loss!r}; expected 'mse' or 'huber'")


# ---------------------------------------------------------------------------
# Checkpoint loading / adaptation
# ---------------------------------------------------------------------------


def extract_actor_state_dict(source: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return ``(actor_state_dict, infos)`` from an rsl_rl/mjlab checkpoint.

  ``source`` may be a checkpoint path or an already loaded checkpoint dict.
  The supported layouts are:

  * current rsl-rl / mjlab: ``{"actor_state_dict": ..., "infos": ...}``;
  * legacy mjlab (rsl-rl < 4): ``model_state_dict`` with ``actor.*`` /
    ``actor_obs_normalizer.*`` keys;
  * a raw actor state dict (``mlp.*`` / ``obs_normalizer.*`` / bare ``0.weight``).
  """
  if isinstance(source, (str, Path)):
    path = Path(source)
    if not path.is_file():
      raise FileNotFoundError(f"checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
  elif isinstance(source, Mapping):
    data = source
  else:
    raise TypeError(f"unsupported checkpoint source: {type(source)!r}")

  if not isinstance(data, Mapping):
    raise TypeError(f"checkpoint must contain a mapping, got {type(data)!r}")

  if "actor_state_dict" in data:
    actor = dict(data["actor_state_dict"])
    infos = dict(data.get("infos") or {})
    return actor, infos

  if "model_state_dict" in data:
    legacy = data["model_state_dict"]
    migrated: dict[str, Any] = {}
    for key, value in legacy.items():
      if key.startswith("actor."):
        migrated["mlp." + key[len("actor."):]] = value
      elif key.startswith("actor_obs_normalizer."):
        migrated["obs_normalizer." + key[len("actor_obs_normalizer."):]] = value
      elif key in ("std", "log_std"):
        migrated["distribution." + ("std_param" if key == "std" else "log_std_param")] = value
      else:
        migrated[key] = value
    return migrated, dict(data.get("infos") or {})

  # A raw actor state dict.
  keys = list(data.keys())
  looks_like_actor = any(
    key.startswith(("mlp.", "obs_normalizer.", "distribution.")) or key[:1].isdigit()
    for key in keys
  )
  if looks_like_actor:
    migrated = {}
    for key, value in data.items():
      migrated["mlp." + key if key[:1].isdigit() else key] = value
    return migrated, dict(data.get("infos") or {})

  raise ValueError(
    "cannot find an actor state dict in the checkpoint; expected 'actor_state_dict' "
    "(rsl-rl/mjlab) or a raw actor state dict (mlp.* keys). "
    f"Top-level keys: {keys[:10]}"
  )


@dataclass(frozen=True)
class ActorArchitecture:
  """Network shape recovered from an actor state dict."""

  obs_dim: int
  hidden_dims: tuple[int, ...]
  output_dim: int


def infer_actor_architecture(state_dict: Mapping[str, Any]) -> ActorArchitecture:
  """Recover ``(obs_dim, hidden_dims, output_dim)`` from an actor state dict."""
  layers: list[tuple[int, torch.Tensor]] = []
  for key, value in state_dict.items():
    if not key.startswith("mlp.") or not key.endswith(".weight"):
      continue
    index_text = key[len("mlp."): -len(".weight")]
    if not index_text.isdigit() or not torch.is_tensor(value):
      continue
    layers.append((int(index_text), value))
  if not layers:
    raise ValueError(
      "actor state dict has no 'mlp.<i>.weight' entries; cannot infer the network "
      f"shape (keys: {list(state_dict)[:10]})"
    )
  layers.sort(key=lambda item: item[0])
  obs_dim = int(layers[0][1].shape[1])
  hidden_dims = tuple(int(weight.shape[0]) for _, weight in layers[:-1])
  output_dim = int(layers[-1][1].shape[0])
  return ActorArchitecture(obs_dim=obs_dim, hidden_dims=hidden_dims, output_dim=output_dim)


def _actor_input_dim(state_dict: Mapping[str, Any]) -> int:
  for key in ("mlp.0.weight", "0.weight"):
    value = state_dict.get(key)
    if torch.is_tensor(value) and value.dim() == 2:
      return int(value.shape[1])
  for key in ("obs_normalizer._mean", "_mean"):
    value = state_dict.get(key)
    if torch.is_tensor(value) and value.dim() >= 2:
      return int(value.shape[-1])
  raise ValueError(
    "cannot determine the actor observation width from the state dict "
    f"(keys: {list(state_dict)[:10]})"
  )


def permute_actor_input_state_dict(
  state_dict: Mapping[str, Any], permutation: Sequence[int] | torch.Tensor
) -> dict[str, Any]:
  """Reorder the actor input dimension: ``result = state_dict[..., permutation]``."""
  if not torch.is_tensor(permutation):
    permutation = torch.tensor(list(permutation), dtype=torch.long)
  else:
    permutation = permutation.to(dtype=torch.long)
  dim = _actor_input_dim(state_dict)
  if permutation.numel() != dim:
    raise ValueError(
      f"permutation has {permutation.numel()} entries, actor input dim is {dim}"
    )
  if not torch.equal(permutation.sort().values, torch.arange(dim)):
    raise ValueError("permutation is not a valid reordering of the actor input dim")
  result: dict[str, Any] = {}
  for key, value in state_dict.items():
    if torch.is_tensor(value) and (
      key == "mlp.0.weight"
      or (key.startswith("obs_normalizer.") and value.dim() >= 2 and value.shape[-1] == dim)
    ):
      result[key] = value.index_select(-1, permutation.to(value.device))
    else:
      result[key] = value
  return result


def expand_actor_input_state_dict(
  state_dict: Mapping[str, Any],
  new_input_dim: int,
  *,
  fill_mean: float = 0.0,
  fill_std: float = 1.0,
) -> dict[str, Any]:
  """Zero-pad the actor input width (warm start from a narrower checkpoint).

  Existing columns keep their meaning and the new columns are appended (the
  canonical layout puts the privileged part last).  The first-layer columns are
  zeroed and the normalizer mean/std are set to ``fill_mean`` / ``fill_std`` so
  the added inputs initially contribute nothing.
  """
  old_input_dim = _actor_input_dim(state_dict)
  new_input_dim = int(new_input_dim)
  if new_input_dim < old_input_dim:
    raise ValueError(
      f"cannot shrink actor input from {old_input_dim} to {new_input_dim}"
    )
  extra = new_input_dim - old_input_dim
  result: dict[str, Any] = {}
  for key, value in state_dict.items():
    if not torch.is_tensor(value):
      result[key] = value
      continue
    if key == "mlp.0.weight" and extra > 0:
      padding = value.new_zeros((value.shape[0], extra))
      result[key] = torch.cat((value, padding), dim=-1)
    elif key == "obs_normalizer._mean" and extra > 0:
      padding = value.new_full((value.shape[0], extra), float(fill_mean))
      result[key] = torch.cat((value, padding), dim=-1)
    elif key in ("obs_normalizer._var", "obs_normalizer._std") and extra > 0:
      padding = value.new_full((value.shape[0], extra), float(fill_std) ** 2 if key.endswith("_var") else float(fill_std))
      result[key] = torch.cat((value, padding), dim=-1)
    else:
      result[key] = value
  return result


def expand_blind_actor_state_dict(
  state_dict: Mapping[str, Any],
  layout: DistillLayout,
  *,
  keep_distribution: bool = True,
) -> dict[str, Any]:
  """Expand today's 252-dim blind actor into the canonical teacher width.

  The privileged columns are appended zeroed, so the resulting actor ignores
  the height scan until RL fine-tuning makes those columns non-zero.  The
  ``distribution.*`` entries (Gaussian std) are kept by default so the result
  loads *strictly* into an rsl_rl ``MLPModel`` actor (needed by
  ``scripts/train.py --warm-start-checkpoint``).
  """
  if not isinstance(layout, DistillLayout):
    raise TypeError(f"layout must be a DistillLayout, got {type(layout)!r}")
  kept = {
    key: value
    for key, value in state_dict.items()
    if keep_distribution or not key.startswith("distribution.")
  }
  input_dim = _actor_input_dim(kept)
  if input_dim != layout.student_dim:
    raise ValueError(
      f"blind actor input dim is {input_dim}, expected the layout's student_dim "
      f"({layout.student_dim}); pass the 252-dim wall actor checkpoint here "
      "(the teacher task has 439 inputs)"
    )
  return expand_actor_input_state_dict(kept, layout.total_dim)


def prepare_actor_state_dict(
  state_dict: Mapping[str, Any],
  layout: DistillLayout,
  input_order: str = INPUT_ORDER_ENV,
  *,
  allow_expand: bool = True,
  label: str = "actor",
) -> dict[str, Any]:
  """Adapt a checkpoint actor state dict to this module's canonical 439-dim layout.

  Three cases are supported:

  * width == ``layout.total_dim`` and ``input_order == "canonical"`` -- used as-is;
  * width == ``layout.total_dim`` and ``input_order == "env"`` -- reorders the
    columns from ``[proprio | height | depth]`` to ``[proprio | depth | height]``;
  * width == ``layout.student_dim`` (today's wall actor, 252) -- appends zeroed
    privileged columns, i.e. the resulting teacher ignores the height scan until
    it is RL fine-tuned on the height-augmented task.

  ``distribution.*`` entries are dropped (the distillation networks are
  deterministic; the exported student re-adds a fixed std).
  """
  if not isinstance(layout, DistillLayout):
    raise TypeError(f"layout must be a DistillLayout, got {type(layout)!r}")
  prepared = {
    key: value for key, value in state_dict.items() if not key.startswith("distribution.")
  }
  dim = _actor_input_dim(prepared)
  order = str(input_order).lower()
  if dim == layout.total_dim:
    if order == INPUT_ORDER_CANONICAL:
      return prepared
    if order == INPUT_ORDER_ENV:
      return permute_actor_input_state_dict(prepared, layout.env_order_permutation())
    raise ValueError(
      f"unknown input_order {input_order!r}; expected "
      f"{INPUT_ORDER_ENV!r} or {INPUT_ORDER_CANONICAL!r}"
    )
  if allow_expand and dim == layout.student_dim:
    print(
      f"[distill] {label} checkpoint actor input is {dim} = student_dim; "
      f"zero-padding to {layout.total_dim} (privileged columns start at 0 weight, "
      "i.e. this checkpoint ignores the height scan until RL fine-tuned)",
      flush=True,
    )
    return expand_actor_input_state_dict(prepared, layout.total_dim)
  raise ValueError(
    f"{label} checkpoint actor input dim {dim} does not match the distillation "
    f"layout (student_dim={layout.student_dim}, total_dim={layout.total_dim}). "
    "A 252-dim actor can be warm-started (zero-padded); a 439-dim teacher has to "
    "come from the height-augmented task. For the latter use "
    f"input_order={INPUT_ORDER_ENV!r} (default; checkpoint trained by plain PPO "
    f"on build_distill_env_cfg) or {INPUT_ORDER_CANONICAL!r}."
  )


def load_actor_state_dict(model: MaskedActor, state_dict: Mapping[str, Any], strict: bool = True) -> dict[str, Any]:
  """Load an actor state dict into a :class:`MaskedActor`, ignoring ``distribution.*``."""
  if not isinstance(model, MaskedActor):
    raise TypeError(f"model must be a MaskedActor, got {type(model)!r}")
  model_keys = set(model.state_dict().keys())
  filtered = {key: value for key, value in state_dict.items() if key in model_keys}
  ignored = [key for key in state_dict if key not in model_keys]
  missing = [key for key in model.state_dict() if key not in filtered]
  if strict and missing:
    raise ValueError(
      f"actor state dict is missing {len(missing)} keys, e.g. {missing[:5]}"
    )
  try:
    model.load_state_dict(filtered, strict=strict)
  except RuntimeError as error:
    raise ValueError(f"actor state dict is incompatible with the MaskedActor: {error}") from error
  return {"loaded": len(filtered), "ignored": ignored, "missing": missing}


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(out: str | Path, iteration: int | None = None, final: bool = False) -> Path:
  """Resolve ``--out`` to a concrete file path.

  * ``out`` ending in ``.pt``/``.pth`` is the final checkpoint file; periodic
    saves become ``<stem>_<iteration>.pt`` next to it.
  * anything else is treated as a directory: final = ``model_final.pt``,
    periodic = ``model_<iteration>.pt``.
  """
  path = Path(out)
  if path.suffix:
    if final:
      return path
    if iteration is None:
      raise ValueError("iteration is required for a periodic checkpoint path")
    return path.with_name(f"{path.stem}_{int(iteration)}{path.suffix}")
  if final:
    return path / "model_final.pt"
  if iteration is None:
    raise ValueError("iteration is required for a periodic checkpoint path")
  return path / f"model_{int(iteration)}.pt"


def save_student_checkpoint(
  path: str | Path,
  student: MaskedActor,
  *,
  optimizer: torch.optim.Optimizer | None = None,
  iteration: int = 0,
  stats: Mapping[str, Any] | None = None,
  init_std: float | None = DEFAULT_INIT_STD,
  extra_infos: Mapping[str, Any] | None = None,
) -> Path:
  """Save the student in the usual rsl_rl checkpoint format.

  The ``actor_state_dict`` entry is the *deployable* 252-dim actor
  (:meth:`MaskedActor.deployable_state_dict`) so ``scripts/play.py`` can load it
  with the normal wall task; the full 439-dim masked student is stored under
  ``distill_student_state_dict`` for resuming distillation.
  """
  if not isinstance(student, MaskedActor):
    raise TypeError(f"student must be a MaskedActor, got {type(student)!r}")
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  meta = {
    "format": "lgdog_distill_student_v1",
    "iteration": int(iteration),
    "layout": student.layout.describe(),
    "action_dim": int(student.output_dim),
    "hidden_dims": list(student.hidden_dims),
    "activation": student.activation,
    "obs_normalization": bool(student.obs_normalization),
    "teacher_sees_student": bool(student.teacher_sees_student),
    "init_std": None if init_std is None else float(init_std),
    "last_stats": dict(stats or {}),
    "deploy_note": (
      "actor_state_dict is a standard rsl_rl actor for the student_dim blind "
      "policy (same input order as today's LgDog1 wall policy)"
    ),
  }
  infos: dict[str, Any] = dict(extra_infos or {})
  infos["distill"] = meta
  payload = {
    "actor_state_dict": _to_cpu(student.deployable_state_dict(init_std=init_std)),
    "critic_state_dict": {},
    "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else {}),
    "iter": int(iteration),
    "infos": infos,
    "distill_student_state_dict": _to_cpu(student.state_dict()),
    "distill_meta": meta,
    "distill_layout": student.layout.describe(),
  }
  torch.save(payload, path)
  return path


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _copy_perception_sidecar(source_checkpoint: str | Path, target_directory: Path) -> str | None:
  """Copy the MGDP perception sidecar referenced by ``source_checkpoint``.

  ``scripts/train.py --warm-start-checkpoint`` (via
  ``src.mgdp_checkpoint.bind_checkpoint_encoder``) resolves the perception
  module from a file sitting next to the checkpoint, so a teacher-init
  checkpoint has to carry that file along.
  """
  try:
    _, infos = extract_actor_state_dict(source_checkpoint)
  except (FileNotFoundError, ValueError, TypeError):
    return None
  contract = infos.get("mgdp_perception")
  if not isinstance(contract, Mapping) or "file" not in contract:
    return None
  name = str(contract["file"])
  if Path(name).name != name:
    raise ValueError(f"invalid MGDP sidecar name in checkpoint: {name!r}")
  source = Path(source_checkpoint).resolve().parent / name
  if not source.is_file():
    print(
      f"[distill][warn] MGDP perception sidecar {source} is missing; "
      "warm-starting the teacher will need --perception-checkpoint",
      flush=True,
    )
    return None
  target = Path(target_directory) / name
  expected = contract.get("sha256")
  if target.is_file() and (expected is None or _file_sha256(target) == expected):
    return name
  shutil.copy2(source, target)
  actual = _file_sha256(target)
  if expected is not None and actual != expected:
    raise ValueError(
      f"copied MGDP perception sidecar {target} does not match the checkpoint contract"
    )
  return name


def save_teacher_init_checkpoint(
  path: str | Path,
  state_dict: Mapping[str, Any],
  *,
  source_checkpoint: str | Path | None = None,
  iteration: int = 0,
  extra_infos: Mapping[str, Any] | None = None,
) -> Path:
  """Write an rsl_rl checkpoint that initialises the privileged teacher RL run.

  ``state_dict`` must be the expanded (439-dim) actor produced by
  :func:`expand_blind_actor_state_dict` (or any teacher-width actor state dict).
  When ``source_checkpoint`` is given, its ``infos`` (MGDP perception contract,
  course contract, ...) are inherited and the perception sidecar file is copied
  next to the new checkpoint, which is what ``scripts/train.py
  --warm-start-checkpoint`` requires.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  infos: dict[str, Any] = {}
  source: Path | None = None
  if source_checkpoint is not None:
    source = Path(source_checkpoint)
    infos.update(_inherit_checkpoint_infos(source))
    sidecar = _copy_perception_sidecar(source, path.parent)
    if sidecar is not None:
      print(f"[distill] copied MGDP perception sidecar {sidecar} to {path.parent}", flush=True)
  for key, value in (extra_infos or {}).items():
    infos[key] = value
  infos["distill_teacher_init"] = {
    "format": "lgdog_teacher_init_v1",
    "iteration": int(iteration),
    "source_checkpoint": None if source is None else str(source),
  }
  payload = {
    "actor_state_dict": _to_cpu(state_dict),
    "critic_state_dict": {},
    "optimizer_state_dict": {},
    "iter": int(iteration),
    "infos": infos,
  }
  torch.save(payload, path)
  return path


# ---------------------------------------------------------------------------
# DAgger training loop
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
  if torch.is_tensor(value):
    return float(value.detach().mean().cpu())
  return float(value)


def train_distill(
  env: Any,
  teacher: MaskedActor,
  student: MaskedActor,
  iterations: int = 500,
  lr: float = 3e-4,
  *,
  steps_per_iteration: int = 32,
  updates_per_iteration: int = 1,
  mini_batch_size: int = 64,
  loss: str = "mse",
  huber_delta: float = 1.0,
  action_noise_std: float = 0.0,
  clip_actions: float | None = 1.0,
  max_grad_norm: float = 1.0,
  save_every: int = 50,
  out: str | Path | None = None,
  log_every: int = 10,
  log_path: str | Path | None = None,
  device: str | torch.device | None = None,
  update_normalizer: bool = True,
  weight_decay: float = 0.0,
  seed: int | None = None,
  log_fn: Any = print,
  teacher_checkpoint: str | Path | None = None,
  extra_infos: Mapping[str, Any] | None = None,
  init_std: float | None = DEFAULT_INIT_STD,
  optimizer: torch.optim.Optimizer | None = None,
  save_optimizer: bool = True,
  reset_env: bool = True,
) -> dict[str, Any]:
  """DAgger distillation: roll out the *student*, label with the *teacher*.

  Every iteration collects ``steps_per_iteration`` environment steps driven by
  the student policy (with optional Gaussian action noise), asks the frozen
  teacher for labels on exactly those observations, and then performs
  ``updates_per_iteration`` passes of mini-batch MSE (or Huber) regression from
  student action to teacher action.  Because the data is produced by the
  student, the student is trained on its own state distribution (DAgger) rather
  than on the teacher's.

  Args:
    env: Either a :class:`DistillEnv` or any object with the same small
      protocol (``get_observations``/``step``/``reset``/``num_actions``/
      ``device``).  The env must auto-reset episodes (mjlab does).
    teacher: Frozen :class:`MaskedActor`; labels are ``teacher(obs, role="teacher")``.
    student: Trainable :class:`MaskedActor`; actions are ``student(obs, role="student")``.
    iterations: Number of DAgger rounds.
    lr: Adam learning rate.
    steps_per_iteration: Environment steps collected per round.
    updates_per_iteration: Gradient passes over each collected buffer.
    mini_batch_size: Mini-batch size for the regression updates.
    loss: ``"mse"`` or ``"huber"``.
    huber_delta: Huber transition point.
    action_noise_std: Gaussian noise added to the student action *executed* in
      the env (labels stay clean); 0 = pure student policy.
    clip_actions: Clip executed actions to ``[-clip_actions, clip_actions]``
      (matching the wall task's ``clip_actions=1.0``); ``None`` disables.
    max_grad_norm: Gradient clipping norm; ``0`` disables.
    save_every: Save a checkpoint every N iterations (0 disables periodic saves).
    out: Output checkpoint path (see :func:`resolve_checkpoint_path`); ``None``
      disables saving.
    log_every: Print a line every N iterations (0 disables printing).
    log_path: Optional JSONL file that receives one record per iteration.
    device: Torch device for both networks; defaults to the student's device.
    update_normalizer: Update the student's observation statistics from the
      student rollouts (rsl_rl behaviour).
    weight_decay: Adam weight decay.
    seed: Optional torch seed.
    log_fn: Callable used for progress lines (pass ``None`` to silence).
    teacher_checkpoint: Teacher checkpoint path; only used to inherit
      ``infos`` (e.g. the MGDP perception sidecar contract) into student saves.
    extra_infos: Extra ``infos`` entries for every saved checkpoint.
    init_std: Gaussian std written into the exported student actor.
    optimizer: Optional existing optimizer (e.g. when resuming).
    save_optimizer: Whether to store the Adam state in checkpoints.
    reset_env: Reset the env before the first rollout.

  Returns:
    A dict with ``iterations``, ``history`` (one record per iteration),
    ``saved_checkpoints`` and ``final_checkpoint``.
  """
  iterations = int(iterations)
  steps_per_iteration = int(steps_per_iteration)
  updates_per_iteration = int(updates_per_iteration)
  mini_batch_size = int(mini_batch_size)
  save_every = int(save_every)
  log_every = int(log_every)
  if iterations <= 0:
    raise ValueError(f"iterations must be positive, got {iterations}")
  if steps_per_iteration <= 0:
    raise ValueError(f"steps_per_iteration must be positive, got {steps_per_iteration}")
  if updates_per_iteration <= 0:
    raise ValueError(f"updates_per_iteration must be positive, got {updates_per_iteration}")
  if mini_batch_size <= 0:
    raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")
  if save_every < 0:
    raise ValueError(f"save_every must be non-negative, got {save_every}")
  if not isinstance(teacher, MaskedActor) or not isinstance(student, MaskedActor):
    raise TypeError("teacher and student must be MaskedActor instances")
  if teacher.layout != student.layout:
    raise ValueError(
      f"teacher layout {teacher.layout.describe()} != student layout "
      f"{student.layout.describe()}"
    )
  if teacher.output_dim != student.output_dim:
    raise ValueError(
      f"teacher output_dim {teacher.output_dim} != student output_dim {student.output_dim}"
    )
  layout = student.layout
  env_layout = getattr(env, "layout", None)
  if isinstance(env_layout, DistillLayout) and env_layout != layout:
    raise ValueError(
      f"env layout {env_layout.describe()} != student layout {layout.describe()}"
    )
  num_actions = getattr(env, "num_actions", None)
  if num_actions is not None and int(num_actions) != student.output_dim:
    raise ValueError(
      f"env has {int(num_actions)} actions, student outputs {student.output_dim}; "
      "check the task/checkpoint"
    )

  if seed is not None:
    torch.manual_seed(int(seed))
  device = torch.device(device) if device is not None else next(student.parameters()).device
  teacher.to(device).eval()
  student.to(device).train()
  for parameter in teacher.parameters():
    parameter.requires_grad_(False)
  if optimizer is None:
    optimizer = torch.optim.Adam(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))

  extra_infos = dict(extra_infos or {})
  inherited = _inherit_checkpoint_infos(teacher_checkpoint) if teacher_checkpoint is not None else {}
  for key, value in inherited.items():
    extra_infos.setdefault(key, value)

  env_device = torch.device(getattr(env, "device", device))

  def emit(message: str) -> None:
    if log_fn is not None:
      log_fn(message)

  # Carry the teacher run's MGDP perception sidecar next to the student
  # checkpoints so ``scripts/play.py`` can resolve the encoder contract.
  if out is not None and teacher_checkpoint is not None:
    sidecar = _copy_perception_sidecar(
      teacher_checkpoint, resolve_checkpoint_path(out, final=True).parent
    )
    if sidecar is not None:
      emit(f"[distill] copied MGDP perception sidecar {sidecar} to the checkpoint directory")

  if reset_env:
    observation = env.reset()[0]
  else:
    observation = env.get_observations()
  observation = torch.as_tensor(observation).to(device=device, dtype=torch.float32)
  if observation.shape[-1] != layout.total_dim:
    raise ValueError(
      f"env observation width {observation.shape[-1]} != layout total {layout.total_dim}"
    )

  history: list[dict[str, Any]] = []
  saved: list[str] = []
  log_file = None
  if log_path is not None:
    log_file_path = Path(log_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_file_path.open("a", encoding="utf8")

  try:
    for iteration in range(1, iterations + 1):
      observations: list[torch.Tensor] = []
      labels: list[torch.Tensor] = []
      student_action_abs = 0.0
      teacher_action_abs = 0.0
      for _ in range(steps_per_iteration):
        with torch.no_grad():
          teacher_action = teacher(observation, role=ROLE_TEACHER)
          student_action = student(observation, role=ROLE_STUDENT)
        observations.append(observation.detach())
        labels.append(teacher_action.detach())
        student_action_abs += _as_float(student_action.abs().mean())
        teacher_action_abs += _as_float(teacher_action.abs().mean())
        action = student_action
        if action_noise_std > 0.0:
          action = action + torch.randn_like(action) * float(action_noise_std)
        if clip_actions is not None:
          action = torch.clamp(action, -float(clip_actions), float(clip_actions))
        next_observation, _reward, _dones, _extras = env.step(action.detach().to(env_device))
        observation = torch.as_tensor(next_observation).to(device=device, dtype=torch.float32)

      observation_batch = torch.cat(observations, dim=0)
      label_batch = torch.cat(labels, dim=0)

      if update_normalizer:
        student.update_normalization(observation_batch, role=ROLE_STUDENT)

      num_samples = int(observation_batch.shape[0])
      batch_size = min(max(1, mini_batch_size), num_samples)
      loss_sum = 0.0
      grad_norm_sum = 0.0
      num_updates = 0
      for _ in range(updates_per_iteration):
        order = torch.randperm(num_samples, device=observation_batch.device)
        for start in range(0, num_samples, batch_size):
          ids = order[start : start + batch_size]
          if ids.numel() == 0:
            continue
          prediction = student(observation_batch[ids], role=ROLE_STUDENT)
          loss_value = distill_loss(
            prediction, label_batch[ids], loss=loss, huber_delta=huber_delta
          )
          optimizer.zero_grad(set_to_none=True)
          loss_value.backward()
          if max_grad_norm and float(max_grad_norm) > 0.0:
            grad_norm = float(
              nn.utils.clip_grad_norm_(student.parameters(), float(max_grad_norm))
            )
          else:
            squared = [
              parameter.grad.detach().pow(2).sum()
              for parameter in student.parameters()
              if parameter.grad is not None
            ]
            grad_norm = float(torch.stack(squared).sum().sqrt()) if squared else 0.0
          optimizer.step()
          loss_sum += float(loss_value.detach())
          grad_norm_sum += grad_norm
          num_updates += 1

      with torch.no_grad():
        student_reference = student(observation_batch, role=ROLE_STUDENT)
        teacher_student_mse = float(F.mse_loss(student_reference, label_batch))

      record: dict[str, Any] = {
        "iteration": iteration,
        "distill_loss": loss_sum / max(1, num_updates),
        "teacher_student_mse": teacher_student_mse,
        "teacher_action_abs_mean": teacher_action_abs / steps_per_iteration,
        "student_action_abs_mean": student_action_abs / steps_per_iteration,
        "grad_norm": grad_norm_sum / max(1, num_updates),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
      }
      history.append(record)
      if log_file is not None:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

      should_log = log_every > 0 and (
        iteration == 1 or iteration == iterations or iteration % log_every == 0
      )
      if should_log:
        emit(
          f"[distill] it={iteration}/{iterations} "
          f"loss={record['distill_loss']:.6f} "
          f"teacher_student_mse={record['teacher_student_mse']:.6f} "
          f"teacher_abs={record['teacher_action_abs_mean']:.4f} "
          f"student_abs={record['student_action_abs_mean']:.4f} "
          f"grad={record['grad_norm']:.3f}"
        )

      if out is not None and save_every > 0 and iteration % save_every == 0:
        path = save_student_checkpoint(
          resolve_checkpoint_path(out, iteration),
          student,
          optimizer=optimizer if save_optimizer else None,
          iteration=iteration,
          stats=record,
          init_std=init_std,
          extra_infos=extra_infos,
        )
        saved.append(str(path))
        emit(f"[distill] saved {path}")

    final_checkpoint: str | None = None
    if out is not None:
      final_path = save_student_checkpoint(
        resolve_checkpoint_path(out, iterations, final=True),
        student,
        optimizer=optimizer if save_optimizer else None,
        iteration=iterations,
        stats=history[-1] if history else {},
        init_std=init_std,
        extra_infos=extra_infos,
      )
      final_checkpoint = str(final_path)
      saved.append(final_checkpoint)
      emit(f"[distill] final student checkpoint: {final_path}")
  finally:
    if log_file is not None:
      log_file.close()

  return {
    "iterations": iterations,
    "history": history,
    "saved_checkpoints": saved,
    "final_checkpoint": final_checkpoint if out is not None else None,
  }


def _inherit_checkpoint_infos(checkpoint: str | Path) -> dict[str, Any]:
  """Copy the teacher checkpoint's ``infos`` (MGDP perception contract etc.).

  Keeps the distilled student playable through the existing MGDP task machinery:
  the contract names a perception sidecar file that has to sit next to the
  checkpoint -- the caller is responsible for copying that file (see the report /
  script output).
  """
  try:
    _, infos = extract_actor_state_dict(checkpoint)
  except (FileNotFoundError, ValueError, TypeError):
    return {}
  # Only text/contract-ish values are safe to carry over.
  keep = (
    "mgdp_perception",
    "mgdp_vision_mode",
    "mgdp_course_contract",
    "mgdp_obstacle_speed",
    "mgdp_rotation_stall",
    "mgdp_passage_curriculum",
    "mgdp_skill_training",
    "mgdp_tire_hole_guidance",
  )
  inherited: dict[str, Any] = {}
  for key in keep:
    if key in infos:
      inherited[key] = infos[key]
  return inherited


# ---------------------------------------------------------------------------
# Task / env construction (lazy simulation imports)
# ---------------------------------------------------------------------------


def _load_task_env_cfg(task_id: str, play: bool = False) -> Any:
  """Load a registered task's (deep-copied) env config with lazy sim imports."""
  import mjlab.tasks  # noqa: F401  (populates task-relevant imports)
  import src.tasks  # noqa: F401  (registers the LgDog1 tasks)
  from mjlab.tasks.registry import load_env_cfg

  try:
    return load_env_cfg(task_id, play=play)
  except KeyError as error:
    from mjlab.tasks.registry import list_tasks

    raise ValueError(
      f"unknown task {task_id!r}; available tasks: {list_tasks()}"
    ) from error


def _terrain_generator_of(cfg: Any) -> Any:
  """Return ``cfg.scene.terrain.terrain_generator`` or raise a clear ``ValueError``."""
  scene = getattr(cfg, "scene", None)
  terrain = getattr(scene, "terrain", None)
  generator = getattr(terrain, "terrain_generator", None)
  if generator is None:
    raise ValueError(
      "env config has no scene.terrain.terrain_generator; a lesson filter only "
      "applies to generator-based terrain tasks"
    )
  return generator


def terrain_lessons(cfg: Any) -> list[str]:
  """Return the sub-terrain lesson names of a loaded env config."""
  generator = _terrain_generator_of(cfg)
  sub_terrains = getattr(generator, "sub_terrains", None)
  if not isinstance(sub_terrains, Mapping) or not sub_terrains:
    raise ValueError(
      "env config has an empty scene.terrain.terrain_generator.sub_terrains"
    )
  return [str(name) for name in sub_terrains]


def _terrain_level_difficulty(level: int) -> float:
  """Return the generator difficulty that lands in the 1-based level ``level``.

  ``passage_terrain_cfg`` maps a generator difficulty to a 0-based stage with
  ``stage = int(difficulty * 8)`` (clipped to ``[0, 7]``).  The 1-based stage
  ``n`` therefore spans ``[(n - 1) / 8, n / 8)``; its mid-point
  ``(n - 0.5) / 8`` is used so the per-cell row jitter in
  ``TerrainGeneratorCfg`` cannot push the value into a neighbouring stage.

  Raises:
    ValueError: if ``level`` is not an integer in ``1..8`` (the message says so).
  """
  valid = (
    isinstance(level, (int, float))
    and not isinstance(level, bool)
    and float(level) == int(level)
  )
  if not valid or not TERRAIN_LEVELS[0] <= int(level) <= TERRAIN_LEVELS[-1]:
    raise ValueError(
      f"level must be an integer between {TERRAIN_LEVELS[0]} and "
      f"{TERRAIN_LEVELS[-1]} (inclusive; the wall course has "
      f"{len(TERRAIN_LEVELS)} difficulty stages), got {level!r}"
    )
  return (int(level) - 0.5) / 8.0


def filter_terrain_lessons(
  cfg: Any, lesson: str | None = None, level: int | None = None
) -> Any:
  """Restrict a loaded env config to a single terrain lesson and/or difficulty.

  ``lesson=None`` keeps the full ten-lesson wall course.  Otherwise
  ``cfg.scene.terrain.terrain_generator.sub_terrains`` is replaced by just
  ``{lesson: cfg}`` with that lesson's ``proportion`` set to ``1.0``, so the whole
  terrain grid becomes that one lesson.  A one-line summary is printed::

      [distill] lesson=high_wall rows=8 cols=20

  ``level`` (1..8, ``None`` = do not pin) pins the difficulty stage: the
  generator is reduced to a single row (``num_rows = 1``) with
  ``difficulty_range = (d, d)``, where ``d = (level - 0.5) / 8`` is the mid-point
  of the 1-based 8-stage curriculum (``stage = int(difficulty * 8)`` in
  ``passage_terrain_cfg``), so every env starts on the requested stage::

      [distill] level=8 difficulty=0.9375

  Pinning only ``level`` (with ``lesson=None``) is allowed and does not need a
  ``sub_terrains`` roster.  Both filters compose; the lesson line is printed
  after the level pin, so its ``rows=`` field reports the pinned single row.

  Raises:
    ValueError: if the config has no terrain generator, the roster is empty,
      ``lesson`` is not one of the available lessons (the message lists them), or
      ``level`` is outside ``1..8``.
  """
  difficulty: float | None = None
  if level is not None:
    difficulty = _terrain_level_difficulty(level)
  if lesson is None and difficulty is None:
    return cfg
  generator = _terrain_generator_of(cfg)
  if difficulty is not None:
    generator.num_rows = 1
    generator.difficulty_range = (difficulty, difficulty)
    print(f"[distill] level={int(level)} difficulty={difficulty}", flush=True)
  if lesson is None:
    return cfg
  sub_terrains = getattr(generator, "sub_terrains", None)
  if not isinstance(sub_terrains, Mapping) or not sub_terrains:
    raise ValueError(
      "env config has an empty scene.terrain.terrain_generator.sub_terrains; "
      "cannot filter to a lesson"
    )
  names = [str(name) for name in sub_terrains]
  name = str(lesson)
  if name not in names:
    raise ValueError(f"unknown lesson {name!r}; available lessons: {names}")
  selected = sub_terrains[name]
  selected.proportion = 1.0
  generator.sub_terrains = {name: selected}
  rows = getattr(generator, "num_rows", "?")
  cols = getattr(generator, "num_cols", "?")
  print(f"[distill] lesson={name} rows={rows} cols={cols}", flush=True)
  return cfg


def available_lessons(task_id: str = WALL_TASK_ID) -> list[str]:
  """List the terrain-lesson names ``task_id`` can be filtered to.

  Loads the task's *training* env config (no simulation / MuJoCo scene is built),
  so it is safe to call from a CLI listing mode.
  """
  return terrain_lessons(_load_task_env_cfg(task_id, play=False))


def build_distill_env_cfg(
  task_id: str = WALL_TASK_ID,
  lesson: str | None = None,
  play: bool = False,
  level: int | None = None,
) -> Any:
  """Load a task's env config and append the privileged height scan to the actor.

  This is the config of the *distillation* env: the 187-dim ``height_scan`` term
  (copied from the critic group) is appended to the actor observation group,
  with the four-frame history pushed down to the individual actor terms so the
  scan stays a single frame.  Teacher RL training uses
  :func:`build_teacher_env_cfg` instead (separate observation group, so the MGDP
  contracts / warm start keep working); both produce the same flat
  ``[proprio | depth | height]`` policy input.

  ``lesson`` optionally restricts the terrain roster to one lesson and ``level``
  (1..8) optionally pins the difficulty stage (see
  :func:`filter_terrain_lessons`); ``None`` keeps the full course / curriculum.
  """
  cfg = _load_task_env_cfg(task_id, play=play)
  filter_terrain_lessons(cfg, lesson, level=level)
  append_height_scan_to_actor(cfg)
  return cfg


def build_teacher_env_cfg(
  task_id: str = WALL_TASK_ID,
  lesson: str | None = None,
  play: bool = False,
  level: int | None = None,
) -> Any:
  """Load a task's env config with the privileged scan as a separate group.

  The actor group is left untouched (history / MGDP contract compatible); the
  height scan becomes the ``"privileged"`` observation group, and
  :func:`register_teacher_task` adds it to ``rl_cfg.obs_groups["actor"]``.

  ``lesson`` optionally restricts the terrain roster to one lesson and ``level``
  (1..8) optionally pins the difficulty stage (see
  :func:`filter_terrain_lessons`); ``None`` keeps the full course / curriculum.
  """
  cfg = _load_task_env_cfg(task_id, play=play)
  filter_terrain_lessons(cfg, lesson, level=level)
  add_privileged_observation_group(cfg)
  return cfg


def build_distill_env(
  task_id: str = WALL_TASK_ID,
  device: str = "cpu",
  num_envs: int = 1,
  play: bool = False,
  seed: int | None = None,
  policy_groups: Sequence[str] | None = None,
  perception_checkpoint: str | Path | None = None,
  lesson: str | None = None,
  level: int | None = None,
) -> DistillEnv:
  """Build a single privileged distillation environment for ``task_id``.

  The returned :class:`DistillEnv` yields flat
  ``[student_part | privileged_part]`` observations.  ``policy_groups`` defaults
  to the task's RL actor observation groups (``rl_cfg.obs_groups["actor"]``).
  ``perception_checkpoint`` optionally overrides the frozen MGDP depth encoder
  (the wall task defaults to ``logs/mgdp/perception/mgdp_wall_v5.pt``; the
  encoder must match the current perception architecture and camera/sensing
  contract).  ``lesson`` optionally restricts the terrain roster to a single
  lesson and ``level`` (1..8) optionally pins the difficulty stage (see
  :func:`filter_terrain_lessons`); ``None`` keeps the full course / curriculum.
  """
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_rl_cfg

  cfg = build_distill_env_cfg(task_id, lesson=lesson, play=play, level=level)
  if int(num_envs) <= 0:
    raise ValueError(f"num_envs must be positive, got {num_envs}")
  cfg.scene.num_envs = int(num_envs)
  if seed is not None:
    cfg.seed = int(seed)
  if perception_checkpoint is not None:
    if "perception" not in cfg.observations:
      raise ValueError(
        f"task {task_id!r} has no MGDP perception group; "
        "--perception-checkpoint does not apply"
      )
    cfg.observations["perception"].terms["depth_feature"].params["checkpoint"] = str(
      perception_checkpoint
    )
  rl_cfg = load_rl_cfg(task_id)
  groups = tuple(policy_groups) if policy_groups is not None else tuple(rl_cfg.obs_groups["actor"])
  print(
    f"[distill] building {task_id} (num_envs={int(num_envs)}, device={device}, "
    f"policy groups={groups})",
    flush=True,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
  except Exception:
    env.close()
    raise
  return DistillEnv.from_env(env, groups)


def teacher_task_id(lesson: str | None = None) -> str:
  """Task id :func:`register_teacher_task` registers for ``lesson``.

  * ``lesson=None``  -> ``LgDog1-Race-MGDP-Wall-Teacher`` (all ten lessons);
  * ``lesson="high_wall"`` -> ``LgDog1-Race-MGDP-Wall-Teacher-high_wall``.

  The id rule is pure (no registry access), so it can be used to print the id
  that *will* be registered.
  """
  if lesson is None:
    return TEACHER_TASK_ID
  return f"{TEACHER_TASK_ID}-{str(lesson)}"


def register_teacher_task(
  lesson: str | None = None,
  level: int | None = None,
  source_task_id: str = WALL_TASK_ID,
) -> str:
  """Register a privileged teacher RL task on top of ``source_task_id``.

  The registered task is the wall task plus a ``"privileged"`` observation group
  holding the 187-dim height scan; the RL actor observation becomes
  ``[actor (proprio history) | perception (depth) | privileged (height)]``, i.e.
  the canonical layout of this module.  The actor group itself is untouched, so
  ``scripts/train.py --warm-start-checkpoint <teacher-init>`` (an expanded blind
  checkpoint, see :func:`save_teacher_init_checkpoint`) and the MGDP perception
  contracts keep working::

      from src.tasks.velocity.rl.distill import register_teacher_task, teacher_task_id
      register_teacher_task()                           # idempotent
      register_teacher_task(lesson="high_wall")         # -> "...-Wall-Teacher-high_wall"
      register_teacher_task(level=8)                     # pinned to difficulty stage 8
      register_teacher_task(lesson="high_wall", level=8) # both filters
      teacher_task_id("high_wall")                      # same id, without registering

      # scripts/train.py imports ``src.tasks`` but not this module, so run it in
      # one process after registering the task:
      #   python -c "import sys, runpy; import src.tasks; \
      #     from src.tasks.velocity.rl.distill import register_teacher_task; \
      #     register_teacher_task(lesson='high_wall'); sys.argv=['scripts/train.py', \
      #     'LgDog1-Race-MGDP-Wall-Teacher-high_wall']; \
      #     runpy.run_path('scripts/train.py', run_name='__main__')"

  ``lesson`` filters the terrain roster to that single lesson and ``level``
  (1..8) pins the terrain difficulty stage (see :func:`filter_terrain_lessons`)
  for both the training and play config.  ``lesson`` also selects the task id
  (see :func:`teacher_task_id`); the id rule is unchanged by ``level``, so
  registering the same lesson at two different levels reuses the first
  registration (idempotent).  Registration is idempotent: an already registered
  id is reused and returned without re-building or re-registering it, and a
  concurrent/duplicate ``register`` ValueError is swallowed.

  Teacher checkpoints end up with a 439-dim actor in canonical order, which is
  what ``scripts/distill_wall.py`` expects by default
  (``--teacher-order canonical``).

  Returns:
    The registered task id.

  Raises:
    ValueError: if ``lesson`` is not an available lesson (the message lists the
      available lessons), ``level`` is outside ``1..8``, or the source task is
      unknown.
  """
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task
  from src.tasks.velocity.rl import PPOHIMRunner

  # Validate eagerly so an out-of-range level fails even when the task id is
  # already registered (the idempotent path below returns before building cfgs).
  if level is not None:
    _terrain_level_difficulty(level)

  task_id = teacher_task_id(lesson)
  if task_id in list_tasks():
    print(f"[distill] teacher task {task_id!r} is already registered; reusing it", flush=True)
    return task_id

  env_cfg = build_teacher_env_cfg(source_task_id, lesson=lesson, play=False, level=level)
  play_env_cfg = build_teacher_env_cfg(source_task_id, lesson=lesson, play=True, level=level)
  rl_cfg = load_rl_cfg(source_task_id)
  actor_groups = tuple(rl_cfg.obs_groups["actor"])
  if TEACHER_PRIVILEGED_GROUP not in actor_groups:
    rl_cfg.obs_groups = {
      **rl_cfg.obs_groups,
      "actor": (*actor_groups, TEACHER_PRIVILEGED_GROUP),
    }
  try:
    register_mjlab_task(
      task_id=task_id,
      env_cfg=env_cfg,
      play_env_cfg=play_env_cfg,
      rl_cfg=rl_cfg,
      runner_cls=PPOHIMRunner,
    )
    print(
      f"[distill] registered teacher task {task_id!r} "
      f"(lesson={lesson!r}, level={level!r}, "
      f"actor obs groups: {tuple(rl_cfg.obs_groups['actor'])})",
      flush=True,
    )
  except ValueError as error:
    if "already registered" not in str(error):
      raise
    print(f"[distill] teacher task {task_id!r} was already registered", flush=True)
  return task_id


__all__ = [
  "ACTION_DIM",
  "ActorArchitecture",
  "DEFAULT_ACTIVATION",
  "DEFAULT_HIDDEN_DIMS",
  "DEFAULT_INIT_STD",
  "DistillEnv",
  "DistillLayout",
  "HEIGHT_SCAN_TERM",
  "INPUT_ORDER_CANONICAL",
  "INPUT_ORDER_ENV",
  "LESSON_ALL",
  "MaskedActor",
  "ROLE_STUDENT",
  "ROLE_TEACHER",
  "TEACHER_PRIVILEGED_GROUP",
  "TEACHER_TASK_ID",
  "TERRAIN_LEVELS",
  "WALL_TASK_ID",
  "add_privileged_observation_group",
  "append_height_scan_to_actor",
  "available_lessons",
  "build_distill_env",
  "build_distill_env_cfg",
  "build_teacher_env_cfg",
  "distill_loss",
  "expand_actor_input_state_dict",
  "expand_blind_actor_state_dict",
  "extract_actor_state_dict",
  "filter_terrain_lessons",
  "infer_actor_architecture",
  "load_actor_state_dict",
  "permute_actor_input_state_dict",
  "prepare_actor_state_dict",
  "register_teacher_task",
  "resolve_checkpoint_path",
  "save_student_checkpoint",
  "save_teacher_init_checkpoint",
  "teacher_task_id",
  "terrain_lessons",
  "train_distill",
]
