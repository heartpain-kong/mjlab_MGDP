"""Unit tests for teacher-student DAgger distillation (no GPU / simulation).

The tests cover the parts of ``src/tasks/velocity/rl/distill.py`` that do not
need MuJoCo/mjlab: the observation split, the role masks, the imitation loss,
network forward/backward, checkpoint adaptation and (with a tiny fake env) the
DAgger loop itself.
"""

from __future__ import annotations

import hashlib
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization

from src.tasks.velocity.rl.distill import (
  DistillEnv,
  DistillLayout,
  INPUT_ORDER_CANONICAL,
  INPUT_ORDER_ENV,
  LESSON_ALL,
  ROLE_STUDENT,
  ROLE_TEACHER,
  TEACHER_TASK_ID,
  WALL_TASK_ID,
  MaskedActor,
  add_privileged_observation_group,
  append_height_scan_to_actor,
  available_lessons,
  build_distill_env_cfg,
  build_teacher_env_cfg,
  distill_loss,
  expand_actor_input_state_dict,
  expand_blind_actor_state_dict,
  extract_actor_state_dict,
  filter_terrain_lessons,
  infer_actor_architecture,
  load_actor_state_dict,
  permute_actor_input_state_dict,
  prepare_actor_state_dict,
  register_teacher_task,
  resolve_checkpoint_path,
  save_student_checkpoint,
  save_teacher_init_checkpoint,
  teacher_task_id,
  terrain_lessons,
  train_distill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout(student_dim: int = 5, privileged_dim: int = 3, actor_student_dim: int = 2) -> DistillLayout:
  return DistillLayout(
    student_dim=student_dim,
    privileged_dim=privileged_dim,
    actor_student_dim=actor_student_dim,
  )


def _tiny_actor(layout: DistillLayout, *, hidden_dims=(8,), output_dim=4, **kwargs) -> MaskedActor:
  return MaskedActor(layout, hidden_dims=hidden_dims, output_dim=output_dim, **kwargs)


def _lesson_cfg(
  names=("start_flat", "high_wall", "stepping_stones"), rows: int = 8, cols: int = 20
) -> SimpleNamespace:
  """Minimal stand-in for a loaded env cfg with a terrain lesson roster."""
  sub_terrains = {
    name: SimpleNamespace(proportion=1.0 / len(names)) for name in names
  }
  generator = SimpleNamespace(sub_terrains=sub_terrains, num_rows=rows, num_cols=cols)
  return SimpleNamespace(
    scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=generator))
  )


def _blind_actor_state_dict(layout: DistillLayout, **kwargs) -> dict:
  """Build a checkpoint-like actor state dict narrowed to ``layout.student_dim``."""
  model = _tiny_actor(layout, **kwargs)
  narrow = {}
  for key, value in model.state_dict().items():
    if key == "mlp.0.weight":
      narrow[key] = value[:, : layout.student_dim].clone()
    elif key.startswith("obs_normalizer.") and value.dim() >= 2:
      narrow[key] = value[..., : layout.student_dim].clone()
    else:
      narrow[key] = value.clone()
  narrow["distribution.std_param"] = torch.full((model.output_dim,), 0.5)
  return narrow


def _observation_term(**overrides):
  term = SimpleNamespace(
    history_length=0,
    flatten_history_dim=True,
    noise=None,
    scale=None,
    params={},
    func=None,
  )
  for key, value in overrides.items():
    setattr(term, key, value)
  return term


class _FakeObservationManager:
  """Minimal stand-in for mjlab's ObservationManager (dict of flat groups)."""

  def __init__(self) -> None:
    self.active_terms = {"actor": ["proprio", "height_scan"], "perception": ["depth"]}
    self.group_obs_term_dim = {"actor": [(2,), (3,)], "perception": [(4,)]}
    self.group_obs_dim = {"actor": (5,), "perception": (4,)}
    self.group_obs_concatenate = {"actor": True, "perception": True}
    self.value = {
      "actor": torch.arange(5, dtype=torch.float32).reshape(1, 5),
      "perception": torch.arange(5, 9, dtype=torch.float32).reshape(1, 4),
    }

  def compute(self) -> dict[str, torch.Tensor]:
    return self.value


class _FakeEnv:
  """Tiny auto-resetting env with the DistillEnv protocol (no simulation)."""

  def __init__(self) -> None:
    self.num_envs = 1
    self.device = "cpu"
    self.action_manager = SimpleNamespace(total_action_dim=4)
    self.observation_manager = _FakeObservationManager()
    self.step_count = 0

  def reset(self):
    return self.observation_manager.value, {}

  def step(self, actions):
    del actions
    self.step_count += 1
    offset = float(self.step_count)
    actor = self.observation_manager.value["actor"] + offset
    perception = self.observation_manager.value["perception"] + offset
    self.observation_manager.value = {"actor": actor, "perception": perception}
    reward = torch.zeros(self.num_envs)
    dones = torch.zeros(self.num_envs, dtype=torch.bool)
    return self.observation_manager.value, reward, dones, {"step": self.step_count}


class _NarrowPolicy(torch.nn.Module):
  """Plain rsl_rl-style actor used to check the deployable student export."""

  def __init__(self, input_dim: int, hidden_dims=(8,), output_dim=4) -> None:
    super().__init__()
    self.mlp = MLP(input_dim, output_dim, hidden_dims, "elu")
    self.obs_normalizer = EmpiricalNormalization(input_dim)

  def forward(self, observation):
    return self.mlp(self.obs_normalizer(observation))


# ---------------------------------------------------------------------------
# 1. Observation split
# ---------------------------------------------------------------------------


class ObservationSplitTests(unittest.TestCase):
  def test_student_and_privileged_dims_sum_to_total(self):
    layout = _layout()
    observation = torch.randn(4, layout.total_dim)
    student_part, privileged_part = layout.split(observation)
    self.assertEqual(tuple(student_part.shape), (4, layout.student_dim))
    self.assertEqual(tuple(privileged_part.shape), (4, layout.privileged_dim))
    self.assertEqual(
      student_part.shape[-1] + privileged_part.shape[-1], layout.total_dim
    )
    self.assertTrue(torch.equal(layout.join(student_part, privileged_part), observation))

  def test_slices_are_contiguous(self):
    layout = _layout(student_dim=5, privileged_dim=3)
    self.assertEqual(layout.student_slice, slice(0, 5))
    self.assertEqual(layout.privileged_slice, slice(5, 8))

  def test_wrong_width_is_rejected(self):
    layout = _layout()
    with self.assertRaises(ValueError):
      layout.split(torch.zeros(2, layout.total_dim - 1))
    with self.assertRaises(TypeError):
      layout.split([0.0, 1.0])

  def test_join_rejects_wrong_parts(self):
    layout = _layout()
    with self.assertRaises(ValueError):
      layout.join(torch.zeros(2, layout.student_dim), torch.zeros(3, layout.privileged_dim))
    with self.assertRaises(ValueError):
      layout.join(torch.zeros(2, layout.student_dim - 1), torch.zeros(2, layout.privileged_dim))


# ---------------------------------------------------------------------------
# 2. Role masks
# ---------------------------------------------------------------------------


class MaskTests(unittest.TestCase):
  def setUp(self):
    self.layout = _layout()
    self.actor = _tiny_actor(self.layout)
    self.observation = torch.randn(6, self.layout.total_dim)

  def test_teacher_input_zeroes_student_part(self):
    teacher_input = self.actor.masked_obs(self.observation, ROLE_TEACHER)
    self.assertEqual(teacher_input[:, : self.layout.student_dim].abs().sum().item(), 0.0)
    self.assertTrue(torch.equal(
      teacher_input[:, self.layout.student_dim :],
      self.observation[:, self.layout.student_dim :],
    ))

  def test_student_input_zeroes_privileged_part(self):
    student_input = self.actor.masked_obs(self.observation, ROLE_STUDENT)
    self.assertEqual(student_input[:, self.layout.student_dim :].abs().sum().item(), 0.0)
    self.assertTrue(torch.equal(
      student_input[:, : self.layout.student_dim],
      self.observation[:, : self.layout.student_dim],
    ))

  def test_forward_uses_the_role_mask(self):
    output = self.actor(self.observation, role=ROLE_STUDENT)
    self.assertEqual(tuple(output.shape), (6, 4))
    self.assertTrue(torch.isfinite(output).all().item())

  def test_teacher_sees_full_observation_when_enabled(self):
    actor = _tiny_actor(self.layout, teacher_sees_student=True)
    teacher_input = actor.masked_obs(self.observation, ROLE_TEACHER)
    self.assertTrue(torch.equal(teacher_input, self.observation))

  def test_unknown_role_is_rejected(self):
    with self.assertRaises(ValueError):
      self.actor.masked_obs(self.observation, "referee")

  def test_mask_buffers_are_not_in_the_state_dict(self):
    keys = list(self.actor.state_dict().keys())
    self.assertIn("mlp.0.weight", keys)
    self.assertIn("obs_normalizer._mean", keys)
    self.assertFalse(any("mask" in key for key in keys))


# ---------------------------------------------------------------------------
# 3. Distillation loss
# ---------------------------------------------------------------------------


class DistillLossTests(unittest.TestCase):
  def test_identical_actions_give_zero(self):
    actions = torch.randn(8, 12)
    loss = distill_loss(actions, actions.clone())
    self.assertEqual(loss.dim(), 0)
    self.assertEqual(loss.item(), 0.0)

  def test_different_actions_are_positive(self):
    student = torch.zeros(8, 12)
    teacher = torch.ones(8, 12)
    self.assertGreater(distill_loss(student, teacher).item(), 0.0)
    self.assertAlmostEqual(distill_loss(student, teacher).item(), 1.0, places=6)

  def test_huber_and_mse_differ_for_large_errors(self):
    student = torch.zeros(4, 3)
    teacher = torch.full((4, 3), 5.0)
    mse = distill_loss(student, teacher, loss="mse").item()
    huber = distill_loss(student, teacher, loss="huber", huber_delta=1.0).item()
    self.assertAlmostEqual(mse, 25.0, places=6)
    self.assertLess(huber, mse)
    self.assertGreater(huber, 0.0)

  def test_shape_mismatch_and_unknown_loss_are_rejected(self):
    with self.assertRaises(ValueError):
      distill_loss(torch.zeros(2, 3), torch.zeros(2, 4))
    with self.assertRaises(TypeError):
      distill_loss([0.0, 1.0], torch.zeros(2))
    with self.assertRaises(ValueError):
      distill_loss(torch.zeros(2, 3), torch.zeros(2, 3), loss="l1")
    with self.assertRaises(ValueError):
      distill_loss(torch.zeros(2, 3), torch.zeros(2, 3), loss="huber", huber_delta=0.0)


# ---------------------------------------------------------------------------
# 4. Forward + backward
# ---------------------------------------------------------------------------


class BackwardTests(unittest.TestCase):
  def test_forward_backward_produces_gradients(self):
    layout = _layout()
    student = _tiny_actor(layout)
    observation = torch.randn(8, layout.total_dim)
    teacher_target = torch.randn(8, 4)

    prediction = student(observation, role=ROLE_STUDENT)
    loss = distill_loss(prediction, teacher_target)
    loss.backward()

    parameters = list(student.parameters())
    self.assertTrue(parameters)
    for parameter in parameters:
      self.assertIsNotNone(parameter.grad, "every parameter should receive a gradient")
      self.assertTrue(torch.isfinite(parameter.grad).all().item())

  def test_optimizer_step_changes_parameters(self):
    layout = _layout()
    student = _tiny_actor(layout)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    observation = torch.randn(8, layout.total_dim)
    teacher_target = torch.ones(8, 4)
    before = [parameter.detach().clone() for parameter in student.parameters()]

    optimizer.zero_grad()
    distill_loss(student(observation, role=ROLE_STUDENT), teacher_target).backward()
    optimizer.step()

    changed = [
      not torch.equal(previous, current.detach())
      for previous, current in zip(before, student.parameters())
    ]
    self.assertTrue(any(changed))

  def test_frozen_teacher_output_has_no_grad(self):
    layout = _layout()
    teacher = _tiny_actor(layout)
    teacher.requires_grad_(False)
    output = teacher(torch.randn(4, layout.total_dim), role=ROLE_TEACHER)
    self.assertFalse(output.requires_grad)


# ---------------------------------------------------------------------------
# 5. Actor ordering / checkpoint adaptation
# ---------------------------------------------------------------------------


class CheckpointAdaptationTests(unittest.TestCase):
  def test_env_order_permutation_maps_to_canonical(self):
    # env order = [a, p0, p1, r0] across groups; canonical = [a, r0, p0, p1]
    layout = DistillLayout(student_dim=2, privileged_dim=2, actor_student_dim=1)
    permutation = layout.env_order_permutation().tolist()
    self.assertEqual(permutation, [0, 3, 1, 2])
    env_ordered = torch.arange(layout.total_dim).reshape(1, -1)
    self.assertEqual(env_ordered[:, permutation].tolist(), [[0, 3, 1, 2]])

  def test_permute_roundtrip_restores_canonical_state_dict(self):
    layout = DistillLayout(student_dim=3, privileged_dim=2, actor_student_dim=1)
    model = _tiny_actor(layout, hidden_dims=(4,), output_dim=2)
    canonical = {key: value.clone() for key, value in model.state_dict().items()}
    permutation = layout.env_order_permutation()
    env_ordered = permute_actor_input_state_dict(canonical, torch.argsort(permutation))
    restored = prepare_actor_state_dict(env_ordered, layout, input_order=INPUT_ORDER_ENV)
    for key, value in canonical.items():
      self.assertTrue(torch.equal(restored[key], value), key)
    # canonical checkpoints are passed through unchanged
    passthrough = prepare_actor_state_dict(canonical, layout, input_order=INPUT_ORDER_CANONICAL)
    for key, value in canonical.items():
      self.assertTrue(torch.equal(passthrough[key], value), key)

  def test_student_width_checkpoint_is_zero_padded(self):
    layout = _layout(student_dim=5, privileged_dim=3, actor_student_dim=2)
    full = _tiny_actor(layout, hidden_dims=(4,), output_dim=2)
    full_state = full.state_dict()
    narrow = {
      key: (
        value[..., : layout.student_dim].clone()
        if key.startswith("obs_normalizer.") and value.dim() >= 2
        else value[:, : layout.student_dim].clone()
        if key == "mlp.0.weight"
        else value.clone()
      )
      for key, value in full_state.items()
    }
    self.assertEqual(narrow["mlp.0.weight"].shape[1], layout.student_dim)
    expanded = prepare_actor_state_dict(narrow, layout, input_order=INPUT_ORDER_CANONICAL)
    self.assertEqual(expanded["mlp.0.weight"].shape[1], layout.total_dim)
    self.assertTrue(torch.equal(
      expanded["mlp.0.weight"][:, : layout.student_dim], narrow["mlp.0.weight"]
    ))
    self.assertEqual(
      expanded["mlp.0.weight"][:, layout.student_dim :].abs().sum().item(), 0.0
    )
    self.assertEqual(
      expanded["obs_normalizer._mean"][:, layout.student_dim :].abs().sum().item(), 0.0
    )
    # and it loads into a full-width MaskedActor
    target = _tiny_actor(layout, hidden_dims=(4,), output_dim=2)
    load_actor_state_dict(target, expanded, strict=True)

  def test_expand_rejects_shrinking(self):
    state = _tiny_actor(_layout()).state_dict()
    with self.assertRaises(ValueError):
      expand_actor_input_state_dict(state, new_input_dim=2)

  def test_expand_blind_actor_keeps_distribution_for_warm_start(self):
    layout = _layout()
    blind = _blind_actor_state_dict(layout)
    expanded = expand_blind_actor_state_dict(blind, layout)
    self.assertEqual(expanded["mlp.0.weight"].shape[1], layout.total_dim)
    self.assertIn("distribution.std_param", expanded)
    self.assertTrue(torch.equal(
      expanded["mlp.0.weight"][:, : layout.student_dim], blind["mlp.0.weight"]
    ))
    self.assertEqual(
      expanded["mlp.0.weight"][:, layout.student_dim :].abs().sum().item(), 0.0
    )
    # the expanded actor loads strictly into a teacher-width MaskedActor
    teacher = _tiny_actor(layout)
    load_actor_state_dict(teacher, expanded, strict=True)
    with self.assertRaises(ValueError):
      expand_blind_actor_state_dict(
        blind, DistillLayout(student_dim=layout.student_dim - 1,
                             privileged_dim=layout.privileged_dim)
      )

  def test_save_teacher_init_inherits_infos_and_sidecar(self):
    layout = _layout()
    blind = _blind_actor_state_dict(layout)
    expanded = expand_blind_actor_state_dict(blind, layout)
    sidecar_bytes = b"perception-sidecar"
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_dir = root / "source"
      source_dir.mkdir()
      (source_dir / "perception_abc.pt").write_bytes(sidecar_bytes)
      source = {
        "actor_state_dict": blind,
        "infos": {
          "mgdp_perception": {
            "version": 1,
            "file": "perception_abc.pt",
            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
          }
        },
      }
      torch.save(source, source_dir / "model_10.pt")
      init_path = save_teacher_init_checkpoint(
        root / "teacher" / "teacher_init.pt", expanded,
        source_checkpoint=source_dir / "model_10.pt",
      )
      self.assertTrue((root / "teacher" / "perception_abc.pt").is_file())
      payload = torch.load(init_path, map_location="cpu", weights_only=False)
      self.assertIn("mgdp_perception", payload["infos"])
      self.assertIn("distill_teacher_init", payload["infos"])
      self.assertEqual(
        payload["actor_state_dict"]["mlp.0.weight"].shape[1], layout.total_dim
      )

  def test_infer_architecture_from_state_dict(self):
    model = _tiny_actor(_layout(), hidden_dims=(8, 6), output_dim=12)
    architecture = infer_actor_architecture(model.state_dict())
    self.assertEqual(architecture.obs_dim, _layout().total_dim)
    self.assertEqual(architecture.hidden_dims, (8, 6))
    self.assertEqual(architecture.output_dim, 12)

  def test_extract_actor_state_dict_from_path_and_legacy(self):
    legacy = {
      "model_state_dict": {
        "actor.0.weight": torch.zeros(4, 6),
        "actor.0.bias": torch.zeros(4),
        "actor_obs_normalizer._mean": torch.zeros(1, 6),
        "std": torch.ones(3),
      }
    }
    actor, infos = extract_actor_state_dict(legacy)
    self.assertEqual(infos, {})
    self.assertIn("mlp.0.weight", actor)
    self.assertIn("obs_normalizer._mean", actor)
    self.assertIn("distribution.std_param", actor)

  def test_resolve_checkpoint_path(self):
    self.assertEqual(
      resolve_checkpoint_path("run/student.pt", final=True), Path("run/student.pt")
    )
    self.assertEqual(
      resolve_checkpoint_path("run/student.pt", iteration=10), Path("run/student_10.pt")
    )
    self.assertEqual(
      resolve_checkpoint_path("run/distill", final=True), Path("run/distill/model_final.pt")
    )
    self.assertEqual(
      resolve_checkpoint_path("run/distill", iteration=10), Path("run/distill/model_10.pt")
    )


# ---------------------------------------------------------------------------
# 6. Env cfg augmentation + flat observation adapter
# ---------------------------------------------------------------------------


class EnvConfigurationTests(unittest.TestCase):
  def _cfg(self):
    actor = SimpleNamespace(
      terms={"joint_pos": _observation_term(), "command": _observation_term()},
      history_length=4,
      flatten_history_dim=True,
      concatenate_terms=True,
    )
    critic = SimpleNamespace(
      terms={"height_scan": _observation_term(scale=0.2)},
      history_length=1,
      flatten_history_dim=True,
      concatenate_terms=True,
    )
    return SimpleNamespace(observations={"actor": actor, "critic": critic})

  def test_height_scan_is_appended_without_history(self):
    cfg = self._cfg()
    actor = cfg.observations["actor"]
    append_height_scan_to_actor(cfg)
    self.assertEqual(list(actor.terms), ["joint_pos", "command", "height_scan"])
    self.assertIsNone(actor.history_length)
    self.assertEqual(actor.terms["joint_pos"].history_length, 4)
    self.assertEqual(actor.terms["joint_pos"].flatten_history_dim, True)
    self.assertEqual(actor.terms["command"].history_length, 4)
    self.assertEqual(actor.terms["height_scan"].history_length, 0)
    self.assertIsNone(actor.terms["height_scan"].noise)
    # critic is untouched
    self.assertEqual(cfg.observations["critic"].history_length, 1)

  def test_append_is_idempotent(self):
    cfg = self._cfg()
    append_height_scan_to_actor(cfg)
    append_height_scan_to_actor(cfg)
    self.assertEqual(
      list(cfg.observations["actor"].terms),
      ["joint_pos", "command", "height_scan"],
    )

  def test_privileged_group_leaves_actor_untouched(self):
    cfg = self._cfg()
    actor = cfg.observations["actor"]

    def factory(terms):
      return SimpleNamespace(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=None,
        flatten_history_dim=True,
      )

    add_privileged_observation_group(cfg, group_factory=factory)
    self.assertIs(cfg.observations["actor"], actor)
    self.assertEqual(actor.history_length, 4)
    self.assertEqual(actor.terms["joint_pos"].history_length, 0)
    privileged = cfg.observations["privileged"]
    self.assertEqual(list(privileged.terms), ["height_scan"])
    self.assertEqual(privileged.terms["height_scan"].history_length, 0)
    self.assertIsNone(privileged.terms["height_scan"].noise)
    add_privileged_observation_group(cfg, group_factory=factory)
    self.assertEqual(list(cfg.observations), ["actor", "critic", "privileged"])

  def test_missing_source_term_is_rejected(self):
    cfg = self._cfg()
    with self.assertRaises(ValueError):
      append_height_scan_to_actor(cfg, term_name="not_there")
    with self.assertRaises(ValueError):
      append_height_scan_to_actor(cfg, source_group="not_there")
    with self.assertRaises(TypeError):
      append_height_scan_to_actor(SimpleNamespace())


class DistillEnvTests(unittest.TestCase):
  def test_flat_observation_layout(self):
    fake = _FakeEnv()
    env = DistillEnv.from_env(fake, ("actor", "perception"))
    self.assertEqual(env.layout.student_dim, 6)  # 2 proprio + 4 depth
    self.assertEqual(env.layout.privileged_dim, 3)
    self.assertEqual(env.layout.actor_student_dim, 2)
    self.assertEqual(env.layout.total_dim, 9)
    self.assertEqual(env.num_actions, 4)

    actor = fake.observation_manager.value["actor"]
    perception = fake.observation_manager.value["perception"]
    expected = torch.cat((actor[:, :2], perception, actor[:, 2:]), dim=-1)
    flat = env.get_observations()
    self.assertTrue(torch.equal(flat, expected))

  def test_step_returns_flat_observation(self):
    fake = _FakeEnv()
    env = DistillEnv.from_env(fake, ("actor", "perception"))
    observation, reward, dones, extras = env.step(torch.zeros(1, 4))
    self.assertEqual(tuple(observation.shape), (1, env.layout.total_dim))
    self.assertEqual(tuple(reward.shape), (1,))
    self.assertEqual(tuple(dones.shape), (1,))
    self.assertEqual(extras["step"], 1)

  def test_missing_group_is_reported(self):
    fake = _FakeEnv()
    with self.assertRaises(ValueError):
      DistillEnv.from_env(fake, ("perception", "actor"))


# ---------------------------------------------------------------------------
# 7. Deployable student export
# ---------------------------------------------------------------------------


class DeployableStudentTests(unittest.TestCase):
  def test_deployable_state_dict_strips_privileged_inputs(self):
    layout = _layout()
    student = _tiny_actor(layout)
    full = student.state_dict()
    deployed = student.deployable_state_dict(init_std=0.3)
    self.assertEqual(deployed["mlp.0.weight"].shape[1], layout.student_dim)
    self.assertTrue(torch.equal(
      deployed["mlp.0.weight"], full["mlp.0.weight"][:, : layout.student_dim]
    ))
    self.assertTrue(torch.equal(
      deployed["obs_normalizer._mean"], full["obs_normalizer._mean"][:, : layout.student_dim]
    ))
    self.assertEqual(deployed["distribution.std_param"].shape, (4,))
    self.assertAlmostEqual(float(deployed["distribution.std_param"][0]), 0.3)

  def test_stripped_network_reproduces_masked_forward(self):
    layout = _layout()
    student = _tiny_actor(layout)
    student.update_normalization(torch.randn(16, layout.total_dim), role=ROLE_STUDENT)
    observation = torch.randn(5, layout.total_dim)
    with torch.no_grad():
      reference = student(observation, role=ROLE_STUDENT)
    deployed = student.deployable_state_dict(init_std=None)
    narrow = _NarrowPolicy(input_dim=layout.student_dim)
    narrow.load_state_dict({key: value for key, value in deployed.items()})
    with torch.no_grad():
      stripped = narrow(observation[:, : layout.student_dim])
    self.assertTrue(torch.allclose(reference, stripped, atol=1e-6), (reference - stripped).abs().max())

  def test_save_student_checkpoint_roundtrip(self):
    layout = _layout()
    student = _tiny_actor(layout)
    with tempfile.TemporaryDirectory() as directory:
      path = save_student_checkpoint(
        Path(directory) / "student.pt", student, iteration=7, stats={"distill_loss": 0.5}
      )
      self.assertTrue(path.is_file())
      actor, infos = extract_actor_state_dict(path)
      self.assertEqual(actor["mlp.0.weight"].shape[1], layout.student_dim)
      self.assertIn("distill", infos)
      self.assertEqual(infos["distill"]["iteration"], 7)
      payload = torch.load(path, map_location="cpu", weights_only=False)
      self.assertIn("distill_student_state_dict", payload)
      self.assertEqual(
        payload["distill_student_state_dict"]["mlp.0.weight"].shape[1], layout.total_dim
      )


# ---------------------------------------------------------------------------
# 8. DAgger loop (fake env)
# ---------------------------------------------------------------------------


class TrainDistillTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    self.fake = _FakeEnv()
    self.env = DistillEnv.from_env(self.fake, ("actor", "perception"))
    self.layout = self.env.layout
    self.teacher = _tiny_actor(self.layout)
    self.student = _tiny_actor(self.layout)

  def test_loop_logs_and_saves(self):
    before = [parameter.detach().clone() for parameter in self.student.parameters()]
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "student.pt"
      log_path = Path(directory) / "distill_log.jsonl"
      result = train_distill(
        self.env,
        self.teacher,
        self.student,
        iterations=2,
        lr=1e-3,
        steps_per_iteration=3,
        updates_per_iteration=1,
        mini_batch_size=2,
        save_every=1,
        out=str(output),
        log_every=0,
        log_path=log_path,
        log_fn=None,
        seed=0,
      )
      self.assertEqual(result["iterations"], 2)
      self.assertEqual(len(result["history"]), 2)
      for record in result["history"]:
        self.assertTrue(math.isfinite(record["distill_loss"]))
        self.assertTrue(math.isfinite(record["teacher_student_mse"]))
        self.assertGreaterEqual(record["teacher_student_mse"], 0.0)
      self.assertTrue(output.is_file())
      self.assertTrue(log_path.is_file())
      self.assertGreaterEqual(len(result["saved_checkpoints"]), 2)
      self.assertEqual(result["final_checkpoint"], str(output))

      payload = torch.load(output, map_location="cpu", weights_only=False)
      self.assertEqual(
        payload["actor_state_dict"]["mlp.0.weight"].shape[1], self.layout.student_dim
      )
      self.assertIn("distill", payload["infos"])

    changed = any(
      not torch.equal(previous, current.detach())
      for previous, current in zip(before, self.student.parameters())
    )
    self.assertTrue(changed, "the student should be updated by the DAgger loop")

  def test_student_drives_the_env(self):
    train_distill(
      self.env,
      self.teacher,
      self.student,
      iterations=1,
      lr=1e-3,
      steps_per_iteration=4,
      updates_per_iteration=1,
      mini_batch_size=2,
      save_every=0,
      out=None,
      log_every=0,
      log_fn=None,
      seed=0,
    )
    self.assertEqual(self.fake.step_count, 4)

  def test_layout_mismatch_is_rejected(self):
    other = _tiny_actor(DistillLayout(student_dim=9, privileged_dim=3))
    with self.assertRaises(ValueError):
      train_distill(self.env, other, self.student, iterations=1, log_fn=None)

  def test_env_action_dim_mismatch_is_rejected(self):
    wrong = MaskedActor(self.layout, hidden_dims=(8,), output_dim=5)
    with self.assertRaises(ValueError):
      train_distill(self.env, wrong, wrong, iterations=1, log_fn=None)


# ---------------------------------------------------------------------------
# 9. Per-lesson teachers (terrain filter, task ids, CLI)
# ---------------------------------------------------------------------------


class LessonFilterTests(unittest.TestCase):
  def test_none_keeps_the_whole_roster_and_prints_nothing(self):
    cfg = _lesson_cfg()
    names_before = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    output = io.StringIO()
    with redirect_stdout(output):
      result = filter_terrain_lessons(cfg, None)
    self.assertIs(result, cfg)
    self.assertEqual(
      list(cfg.scene.terrain.terrain_generator.sub_terrains), names_before
    )
    self.assertEqual(output.getvalue(), "")

  def test_high_wall_becomes_the_only_lesson(self):
    cfg = _lesson_cfg()
    output = io.StringIO()
    with redirect_stdout(output):
      filter_terrain_lessons(cfg, "high_wall")
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    self.assertEqual(list(sub_terrains), ["high_wall"])
    self.assertAlmostEqual(float(sub_terrains["high_wall"].proportion), 1.0, places=6)
    self.assertEqual(terrain_lessons(cfg), ["high_wall"])
    self.assertIn("[distill] lesson=high_wall rows=8 cols=20", output.getvalue())

  def test_stepping_stones_becomes_the_only_lesson(self):
    cfg = _lesson_cfg()
    with redirect_stdout(io.StringIO()):
      filter_terrain_lessons(cfg, "stepping_stones")
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    self.assertEqual(list(sub_terrains), ["stepping_stones"])
    self.assertAlmostEqual(
      float(sub_terrains["stepping_stones"].proportion), 1.0, places=6
    )

  def test_unknown_lesson_raises_with_the_available_lessons(self):
    cfg = _lesson_cfg()
    with self.assertRaises(ValueError) as raised:
      filter_terrain_lessons(cfg, "moon_wall")
    message = str(raised.exception)
    self.assertIn("moon_wall", message)
    self.assertIn("high_wall", message)
    self.assertIn("stepping_stones", message)
    # the roster is left untouched on a bad name
    self.assertEqual(
      list(cfg.scene.terrain.terrain_generator.sub_terrains),
      ["start_flat", "high_wall", "stepping_stones"],
    )

  def test_cfg_without_a_terrain_generator_is_rejected(self):
    with self.assertRaises(ValueError):
      filter_terrain_lessons(SimpleNamespace(), "high_wall")
    empty = SimpleNamespace(
      scene=SimpleNamespace(
        terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={}))
      )
    )
    with self.assertRaises(ValueError):
      filter_terrain_lessons(empty, "high_wall")


class TeacherTaskIdTests(unittest.TestCase):
  def test_lesson_task_id_rule(self):
    self.assertEqual(teacher_task_id(), TEACHER_TASK_ID)
    self.assertEqual(TEACHER_TASK_ID, "LgDog1-Race-MGDP-Wall-Teacher")
    self.assertEqual(
      teacher_task_id("high_wall"), "LgDog1-Race-MGDP-Wall-Teacher-high_wall"
    )
    self.assertEqual(
      teacher_task_id("stepping_stones"),
      "LgDog1-Race-MGDP-Wall-Teacher-stepping_stones",
    )

  def test_lesson_all_is_the_cli_sentinel_for_no_filter(self):
    self.assertEqual(LESSON_ALL, "all")


class RealWallTaskLessonTests(unittest.TestCase):
  """Integration checks against the registered wall task (no simulation is built)."""

  def test_wall_task_cfg_can_be_filtered_to_one_lesson(self):
    cfg = build_teacher_env_cfg(WALL_TASK_ID, lesson="high_wall")
    generator = cfg.scene.terrain.terrain_generator
    self.assertEqual(list(generator.sub_terrains), ["high_wall"])
    self.assertAlmostEqual(
      float(generator.sub_terrains["high_wall"].proportion), 1.0, places=6
    )
    # the privileged group is still added on top of the filtered cfg
    self.assertIn("privileged", cfg.observations)

    distill_cfg = build_distill_env_cfg(WALL_TASK_ID, lesson="stepping_stones")
    sub_terrains = distill_cfg.scene.terrain.terrain_generator.sub_terrains
    self.assertEqual(list(sub_terrains), ["stepping_stones"])
    self.assertAlmostEqual(
      float(sub_terrains["stepping_stones"].proportion), 1.0, places=6
    )
    self.assertIn("height_scan", distill_cfg.observations["actor"].terms)

  def test_unknown_lesson_on_the_real_task_lists_the_available_lessons(self):
    cfg = build_teacher_env_cfg(WALL_TASK_ID)  # unfiltered roster
    available = set(terrain_lessons(cfg))
    self.assertIn("high_wall", available)
    self.assertIn("stepping_stones", available)
    with self.assertRaises(ValueError) as raised:
      build_distill_env_cfg(WALL_TASK_ID, lesson="not_a_lesson")
    message = str(raised.exception)
    self.assertIn("not_a_lesson", message)
    for name in available:
      self.assertIn(name, message)


class RegisterTeacherTaskTests(unittest.TestCase):
  def test_register_three_task_ids_is_idempotent(self):
    from mjlab.tasks.registry import _REGISTRY, list_tasks

    pre_existing = set(list_tasks())
    registered_here: list[str] = []
    for lesson in (None, "high_wall", "stepping_stones"):
      task_id = register_teacher_task(lesson=lesson)
      self.assertEqual(task_id, teacher_task_id(lesson))
      if task_id not in pre_existing:
        registered_here.append(task_id)
      self.assertIn(task_id, list_tasks())
      # a repeated call must not raise and must return the same id
      self.assertEqual(register_teacher_task(lesson=lesson), task_id)
    try:
      self.assertIn(teacher_task_id("high_wall"), list_tasks())
      self.assertIn(teacher_task_id("stepping_stones"), list_tasks())
      self.assertEqual(register_teacher_task(lesson="high_wall"), teacher_task_id("high_wall"))
    finally:
      for task_id in registered_here:
        _REGISTRY.pop(task_id, None)


class DistillWallCliTests(unittest.TestCase):
  def test_list_lessons_prints_lessons_and_teacher_task_ids(self):
    from scripts.distill_wall import main as distill_wall_main

    output = io.StringIO()
    with redirect_stdout(output):
      return_code = distill_wall_main(["--list-lessons"])
    text = output.getvalue()
    self.assertEqual(return_code, 0)
    self.assertIn("high_wall", text)
    self.assertIn("stepping_stones", text)
    self.assertIn(teacher_task_id("high_wall"), text)
    self.assertIn(teacher_task_id("stepping_stones"), text)
    self.assertIn(TEACHER_TASK_ID, text)

  def test_teacher_is_required_outside_the_listing_modes(self):
    from scripts.distill_wall import main as distill_wall_main

    with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
      distill_wall_main([])
    self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
  unittest.main()
