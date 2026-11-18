"""Left-right mirror symmetry augmentation for the LgDog1 velocity policy.

Reference: robot_lab's ``.../mdp/symmetry/anymal.py`` (Apache-2.0).  Only the
left-right transform is implemented here; front-back and diagonal transforms
do not apply to this task, so the augmented mini-batch grows by x2.

Joint layout (``src/assets/robots/lgdog1/xmls/lgdog1.xml`` and
``src/assets/robots/lgdog1/lgdog1_constants.py``)::

    [FR, FL, RR, RL] x [hip, thigh, calf]

The left-right mirror is:

* ``FL <-> FR`` and ``RL <-> RR`` (the legs swap);
* ``hip`` reverses sign (axis ``+X``, a roll joint);
* ``thigh`` / ``calf`` keep their sign (axis ``+Y``, pitch joints).

Actor observation layout (read this before reshaping!)
------------------------------------------------------
The actor group is produced by
``ObservationManager.compute_group`` with
``ObservationGroupCfg.history_length = 4`` and
``flatten_history_dim = True`` (see
``src/tasks/velocity/config/lgdog1/mgdp_env_cfg.py``).  For every term the
manager flattens the per-term circular buffer
``(num_envs, history, term_dim) -> (num_envs, history * term_dim)`` and only
afterwards concatenates all terms (``observation_manager.py``).  The resulting
188-wide vector is therefore **term-major**, not ``4 frames x 47``::

    [ang_vel t0..t3][gravity t0..t3][command t0..t3][phase t0..t3]
    [joint_pos t0..t3][joint_vel t0..t3][actions t0..t3]
    = 12 + 12 + 12 + 8 + 48 + 48 + 48 = 188

A plain ``obs.reshape(4, 47)`` would mix frames and terms and silently corrupt
the augmented data.  ``mirror_actor_obs`` re-groups the term-major vector into
``(..., history, 47)`` frames before applying the per-frame transform.  It also
accepts a single frame (47) and frame-major ``(..., history, 47)`` batches
(the HIM history layout).  The input layout is never changed in place.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

__all__ = [
  "compute_symmetric_states",
  "mirror_actions",
  "mirror_actor_obs",
  "mirror_joint_vector",
  "mirror_obs_dict",
]

# ---- Joint layout -----------------------------------------------------------
JOINT_DIM = 12
# new index -> old index for [FR, FL, RR, RL] x [hip, thigh, calf].
_JOINT_MIRROR_PERM = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
_HIP_INDICES = (0, 3, 6, 9)

# ---- Actor observation layout ----------------------------------------------
FRAME_DIM = 47
# Contiguous term blocks of one frame, in ObservationManager concatenation order.
_TERM_DIMS = (3, 3, 3, 2, 12, 12, 12)
assert sum(_TERM_DIMS) == FRAME_DIM

_ANG_VEL_SIGN = (-1.0, 1.0, -1.0)
_GRAVITY_SIGN = (1.0, -1.0, 1.0)
_COMMAND_SIGN = (1.0, -1.0, -1.0)


def _vec(values: tuple[float, ...], ref: torch.Tensor) -> torch.Tensor:
  return torch.tensor(values, device=ref.device, dtype=ref.dtype)


def mirror_joint_vector(x: torch.Tensor) -> torch.Tensor:
  """Mirror a LgDog1 joint vector left-to-right.

  Args:
    x: Tensor whose last dimension is 12 (``[FR, FL, RR, RL] x [hip, thigh,
      calf]``).  Any number of leading dimensions is allowed.

  Returns:
    A new tensor with the same shape: FL/FR and RL/RR swapped and the four hip
    entries negated.
  """
  if x.shape[-1] != JOINT_DIM:
    raise ValueError(f"expected last dim {JOINT_DIM}, got {tuple(x.shape)}")
  perm = torch.tensor(_JOINT_MIRROR_PERM, device=x.device, dtype=torch.long)
  out = x[..., perm].clone()
  hips = torch.tensor(_HIP_INDICES, device=x.device, dtype=torch.long)
  out[..., hips] = -out[..., hips]
  return out


def _mirror_frame(frame: torch.Tensor) -> torch.Tensor:
  """Mirror actor observation frames whose last dimension is 47."""
  if frame.shape[-1] != FRAME_DIM:
    raise ValueError(f"expected last dim {FRAME_DIM}, got {tuple(frame.shape)}")
  out = frame.clone()
  # base_ang_vel: flip x and z (0:3)
  out[..., 0:3] = out[..., 0:3] * _vec(_ANG_VEL_SIGN, out)
  # projected_gravity: flip y (3:6)
  out[..., 3:6] = out[..., 3:6] * _vec(_GRAVITY_SIGN, out)
  # command [vx, vy, wz]: flip y and yaw (6:9)
  out[..., 6:9] = out[..., 6:9] * _vec(_COMMAND_SIGN, out)
  # phase (9:11) is invariant: it is a clock signal, not a spatial quantity.
  # joint_pos (11:23), joint_vel (23:35) and last actions (35:47) use the joint
  # permutation.
  out[..., 11:23] = mirror_joint_vector(out[..., 11:23])
  out[..., 23:35] = mirror_joint_vector(out[..., 23:35])
  out[..., 35:47] = mirror_joint_vector(out[..., 35:47])
  return out


def mirror_actor_obs(obs: torch.Tensor) -> torch.Tensor:
  """Mirror actor observations left-to-right.

  Supported layouts (determined from the last dimension):

  * ``(B, 47)`` / ``(B, history, 47)``: frame-major, e.g. the HIM history.
  * ``(B, 188)`` (or any other ``history * 47``): mjlab's term-major flattened
    history described in the module docstring.
  """
  width = obs.shape[-1]
  if width == FRAME_DIM:
    # Single frame, or frame-major (..., history, 47): each row of 47 is one
    # frame and can be mirrored independently.
    return _mirror_frame(obs)
  if width % FRAME_DIM != 0:
    raise ValueError(
      f"actor observation width {width} is neither {FRAME_DIM} nor a multiple "
      f"of {FRAME_DIM}"
    )

  history = width // FRAME_DIM
  lead = obs.shape[:-1]
  # Re-group term-major [term A t0..tH-1 | term B t0..tH-1 | ...] into
  # frame-major (..., history, 47) so one transform handles every term.
  frames = obs.new_empty(*lead, history, FRAME_DIM)
  src = dst = 0
  for term_dim in _TERM_DIMS:
    block = obs[..., src : src + history * term_dim]
    frames[..., :, dst : dst + term_dim] = block.reshape(
      *lead, history, term_dim
    )
    src += history * term_dim
    dst += term_dim

  mirrored = _mirror_frame(frames)

  # Scatter back into the original term-major layout.
  out = obs.clone()
  src = dst = 0
  for term_dim in _TERM_DIMS:
    block = mirrored[..., dst : dst + term_dim]
    out[..., src : src + history * term_dim] = block.reshape(
      *lead, history * term_dim
    )
    src += history * term_dim
    dst += term_dim
  return out


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  """Mirror a ``(B, 12)`` action tensor (same joint permutation)."""
  return mirror_joint_vector(actions)


def _is_tensordict(value: Any) -> bool:
  return hasattr(value, "batch_size") and hasattr(value, "repeat")


def mirror_obs_dict(obs: Mapping[str, torch.Tensor] | Any) -> Any:
  """Double a per-group observation dict, mirroring only the actor group.

  The returned object keeps the input type (``TensorDict`` or plain ``dict``)
  and has twice the batch size.  The first half repeats the original data; the
  second half mirrors the actor group while every other group (e.g. the critic
  with ``height_scan``) is simply repeated, matching robot_lab: the critic is
  never evaluated on mirrored privileged observations.
  """
  if _is_tensordict(obs):
    batch_size = int(obs.batch_size[0])
    obs_aug = obs.repeat(2)
  else:
    values = list(obs.values())
    if not values:
      raise ValueError("observation dict is empty")
    batch_size = int(values[0].shape[0])
    obs_aug = {
      name: torch.cat((value, value), dim=0) for name, value in obs.items()
    }

  for actor_group in ("actor", "policy"):
    if actor_group not in obs:
      continue
    actor_obs = obs[actor_group]
    mirrored = mirror_actor_obs(actor_obs)
    if _is_tensordict(obs):
      obs_aug[actor_group][:batch_size] = actor_obs
      obs_aug[actor_group][batch_size:] = mirrored
    else:
      obs_aug[actor_group] = torch.cat((actor_obs, mirrored), dim=0)
    break
  return obs_aug


def compute_symmetric_states(
  obs: Mapping[str, torch.Tensor] | Any | None = None,
  actions: torch.Tensor | None = None,
  env: Any | None = None,
) -> tuple[Any | None, torch.Tensor | None]:
  """Return ``(obs_aug, actions_aug)`` with the batch doubled by mirroring.

  Matches robot_lab's ``compute_symmetric_states`` calling convention; the
  optional ``env`` argument is accepted for compatibility with rsl_rl's
  symmetry hook but is unused.  The batch layout is ``[original, mirrored]``.
  """
  obs_aug = None if obs is None else mirror_obs_dict(obs)
  if actions is None:
    actions_aug = None
  else:
    actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
  return obs_aug, actions_aug
