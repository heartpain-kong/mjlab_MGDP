"""Tests for the LgDog1 left-right symmetry augmentation."""

import pytest
import torch

from src.tasks.velocity.mdp.symmetry import (
  FRAME_DIM,
  JOINT_DIM,
  compute_symmetric_states,
  mirror_actions,
  mirror_actor_obs,
  mirror_joint_vector,
  mirror_obs_dict,
)


def _random(frame_shape):
  return torch.randn(*frame_shape, generator=torch.Generator().manual_seed(0))


def test_joint_mirror_swaps_left_right_and_negates_hips():
  x = torch.arange(JOINT_DIM, dtype=torch.float32).unsqueeze(0)
  mirrored = mirror_joint_vector(x)

  # New FR <- old FL (hip negated); new FL <- old FR (hip negated); same for
  # hind legs.  thigh/calf keep their sign.
  expected = torch.tensor(
    [[-3.0, 4.0, 5.0, -0.0, 1.0, 2.0, -9.0, 10.0, 11.0, -6.0, 7.0, 8.0]]
  )
  torch.testing.assert_close(mirrored, expected)

  # FL <-> FR explicit check (legs are [FR, FL, RR, RL] x [hip, thigh, calf]).
  assert mirrored[0, 0] == -x[0, 3]  # FR hip <- FL hip
  assert mirrored[0, 1] == x[0, 4]  # FR thigh <- FL thigh
  assert mirrored[0, 2] == x[0, 5]  # FR calf <- FL calf
  assert mirrored[0, 3] == -x[0, 0]  # FL hip <- FR hip
  assert mirrored[0, 6] == -x[0, 9]  # RR hip <- RL hip
  assert mirrored[0, 9] == -x[0, 6]  # RL hip <- RR hip
  # thigh / calf never flip.
  assert torch.all(mirrored[:, [1, 2, 4, 5, 7, 8, 10, 11]] >= 0)


def test_joint_mirror_is_an_involution():
  x = _random((5, JOINT_DIM))
  torch.testing.assert_close(mirror_joint_vector(mirror_joint_vector(x)), x)


def test_actor_frame_sign_flips_and_phase_untouched():
  frame = torch.zeros(2, FRAME_DIM)
  frame[0, 0:3] = torch.tensor([1.0, 2.0, 3.0])  # ang vel
  frame[0, 3:6] = torch.tensor([4.0, 5.0, 6.0])  # gravity
  frame[0, 6:9] = torch.tensor([7.0, 8.0, 9.0])  # command
  frame[0, 9:11] = torch.tensor([0.25, -0.5])  # phase
  frame[0, 11:23] = torch.arange(JOINT_DIM, dtype=torch.float32)  # joint pos
  frame[0, 23:35] = torch.arange(JOINT_DIM, dtype=torch.float32) + 100  # vel
  frame[0, 35:47] = torch.arange(JOINT_DIM, dtype=torch.float32) + 200  # act
  frame[1] = torch.arange(FRAME_DIM, dtype=torch.float32)

  mirrored = mirror_actor_obs(frame)
  torch.testing.assert_close(
    mirrored[0, 0:3], torch.tensor([-1.0, 2.0, -3.0])
  )
  torch.testing.assert_close(
    mirrored[0, 3:6], torch.tensor([4.0, -5.0, 6.0])
  )
  torch.testing.assert_close(
    mirrored[0, 6:9], torch.tensor([7.0, -8.0, -9.0])
  )
  torch.testing.assert_close(mirrored[0, 9:11], frame[0, 9:11])
  torch.testing.assert_close(
    mirrored[0, 11:23], mirror_joint_vector(frame[0, 11:23])
  )
  torch.testing.assert_close(
    mirrored[0, 23:35], mirror_joint_vector(frame[0, 23:35])
  )
  torch.testing.assert_close(
    mirrored[0, 35:47], mirror_joint_vector(frame[0, 35:47])
  )


def test_actor_frame_mirror_is_an_involution():
  for shape in ((3, FRAME_DIM), (3, 4, FRAME_DIM), (3, 4 * FRAME_DIM)):
    x = torch.randn(*shape)
    torch.testing.assert_close(mirror_actor_obs(mirror_actor_obs(x)), x)


def test_term_major_188_layout_keeps_phase_and_mirrors_each_term():
  # One 188-wide actor observation.  Because mjlab flattens each term's history
  # before concatenating, the layout is term-major, not 4 frames x 47.
  x = torch.arange(4 * FRAME_DIM, dtype=torch.float32).unsqueeze(0)
  out = mirror_actor_obs(x)

  terms = {
    "base_ang_vel": (0, 3),
    "projected_gravity": (12, 3),
    "command": (24, 3),
    "phase": (36, 2),
    "joint_pos": (44, 12),
    "joint_vel": (92, 12),
    "actions": (140, 12),
  }
  # phase block is invariant for every history frame
  torch.testing.assert_close(out[0, 36:44], x[0, 36:44])

  for name, (start, dim) in terms.items():
    if name == "phase":
      continue
    frames = x[0, start : start + 4 * dim].reshape(4, dim)
    mirrored_frames = out[0, start : start + 4 * dim].reshape(4, dim)
    if dim == 12:
      for i in range(4):
        torch.testing.assert_close(
          mirrored_frames[i], mirror_joint_vector(frames[i])
        )
    else:
      sign = {
        "base_ang_vel": torch.tensor([-1.0, 1.0, -1.0]),
        "projected_gravity": torch.tensor([1.0, -1.0, 1.0]),
        "command": torch.tensor([1.0, -1.0, -1.0]),
      }[name]
      torch.testing.assert_close(mirrored_frames, frames * sign)


def test_mirror_actor_obs_rejects_unknown_width():
  with pytest.raises(ValueError):
    mirror_actor_obs(torch.zeros(2, 50))
  with pytest.raises(ValueError):
    mirror_joint_vector(torch.zeros(2, 11))


def test_compute_symmetric_states_doubles_batch_with_original_first():
  batch = 4
  obs = {
    "actor": torch.randn(batch, 4 * FRAME_DIM),
    "critic": torch.randn(batch, 234),
  }
  actions = torch.randn(batch, JOINT_DIM)

  obs_aug, actions_aug = compute_symmetric_states(obs, actions)

  assert obs_aug["actor"].shape == (2 * batch, 4 * FRAME_DIM)
  assert obs_aug["critic"].shape == (2 * batch, 234)
  assert actions_aug.shape == (2 * batch, JOINT_DIM)

  # First half is the untouched original data.
  torch.testing.assert_close(obs_aug["actor"][:batch], obs["actor"])
  torch.testing.assert_close(obs_aug["critic"][:batch], obs["critic"])
  torch.testing.assert_close(actions_aug[:batch], actions)

  # Second half mirrors the actor/actions and repeats the critic unchanged.
  torch.testing.assert_close(
    obs_aug["actor"][batch:], mirror_actor_obs(obs["actor"])
  )
  torch.testing.assert_close(obs_aug["critic"][batch:], obs["critic"])
  torch.testing.assert_close(actions_aug[batch:], mirror_actions(actions))


def test_tensordict_obs_is_supported():
  from tensordict import TensorDict

  batch = 3
  obs = TensorDict(
    {
      "actor": torch.randn(batch, 4 * FRAME_DIM),
      "critic": torch.randn(batch, 234),
    },
    batch_size=[batch],
  )
  obs_aug = mirror_obs_dict(obs)

  assert obs_aug.batch_size[0] == 2 * batch
  torch.testing.assert_close(obs_aug["actor"][:batch], obs["actor"])
  torch.testing.assert_close(
    obs_aug["actor"][batch:], mirror_actor_obs(obs["actor"])
  )
  torch.testing.assert_close(obs_aug["critic"][batch:], obs["critic"])


def test_compute_symmetric_states_handles_missing_inputs():
  obs = {"actor": torch.zeros(2, FRAME_DIM)}
  obs_aug, actions_aug = compute_symmetric_states(obs=obs)
  assert obs_aug is not None
  assert actions_aug is None

  obs_aug, actions_aug = compute_symmetric_states(
    actions=torch.zeros(2, JOINT_DIM)
  )
  assert obs_aug is None
  assert actions_aug.shape == (4, JOINT_DIM)


def test_mirror_actions_equals_mirror_joint_vector():
  actions = torch.randn(6, JOINT_DIM)
  torch.testing.assert_close(mirror_actions(actions), mirror_joint_vector(actions))
