"""CPU tests for the optional perception encoders (DreamWaQ VAE / DELTA attention).

No simulator or MuJoCo runtime is exercised: both encoders are pure PyTorch and
are tested directly on synthetic history / height-scan tensors.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

try:  # Conventional repo import (also exercises the mdp package wiring).
  from src.tasks.velocity.mdp.perception_encoders import (
    HeightAttentionEncoder,
    TerrainVAE,
  )
except ImportError:  # pragma: no cover - torch-only env without mjlab/MuJoCo.
  # The encoder module itself is pure PyTorch, so load it from disk when the
  # ``src.tasks.velocity.mdp`` package cannot import because mjlab is absent.
  _MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tasks"
    / "velocity"
    / "mdp"
    / "perception_encoders.py"
  )
  _spec = importlib.util.spec_from_file_location(
    "perception_encoders", _MODULE_PATH
  )
  assert _spec is not None and _spec.loader is not None
  _module = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(_module)
  HeightAttentionEncoder = _module.HeightAttentionEncoder
  TerrainVAE = _module.TerrainVAE


def _history(batch=5, length=4, dim=47, seed=0):
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(batch, length, dim, generator=generator)


def _scan(batch=3, side=8, seed=0):
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(batch, side * side, generator=generator)


class TerrainVAETests(unittest.TestCase):
  def test_output_shapes(self):
    model = TerrainVAE(
      history_dim=47,
      history_len=4,
      latent_dim=16,
      feature_dim=32,
      hidden=(128, 64),
    ).eval()
    history = _history()
    z, mu, logvar, recon = model(history)
    self.assertEqual(tuple(z.shape), (5, 16))
    self.assertEqual(tuple(mu.shape), (5, 16))
    self.assertEqual(tuple(logvar.shape), (5, 16))
    self.assertEqual(tuple(recon.shape), (5, 4, 47))

    # Flattened (B, H * D) input is accepted and gives the same shapes.
    z2, mu2, logvar2, recon2 = model(history.reshape(5, -1))
    self.assertEqual(tuple(z2.shape), (5, 16))
    self.assertEqual(tuple(recon2.shape), (5, 4, 47))

  def test_kl_non_negative_and_losses_finite(self):
    history = _history()
    for latent_dim in (16, 32):
      model = TerrainVAE(
        history_dim=47,
        history_len=4,
        latent_dim=latent_dim,
        feature_dim=64,
        hidden=128,
      ).train()
      total, recon_loss, kl_loss = model.loss(history)
      for value in (total, recon_loss, kl_loss):
        self.assertTrue(bool(torch.isfinite(value)), value)
      self.assertGreaterEqual(float(kl_loss.detach()), 0.0)

      # Gradients reach every parameter through the combined objective.
      total.backward()
      for name, parameter in model.named_parameters():
        self.assertIsNotNone(parameter.grad, name)
        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

  def test_loss_beta_override(self):
    history = _history()
    model = TerrainVAE(
      history_dim=47, history_len=4, latent_dim=16, feature_dim=32, hidden=64
    ).eval()
    total, recon_loss, kl_loss = model.loss(history, beta=0.0)
    torch.testing.assert_close(total, recon_loss)
    total_beta, _, _ = model.loss(history, beta=2.0)
    torch.testing.assert_close(total_beta, recon_loss + 2.0 * kl_loss)

  def test_eval_mu_is_deterministic(self):
    model = TerrainVAE(
      history_dim=42, history_len=4, latent_dim=32, feature_dim=64, hidden=128
    ).eval()
    history = _history(dim=42)
    _, mu_a, logvar_a, recon_a = model(history)
    _, mu_b, logvar_b, recon_b = model(history)
    self.assertTrue(torch.equal(mu_a, mu_b))
    self.assertTrue(torch.equal(logvar_a, logvar_b))
    self.assertTrue(torch.equal(recon_a, recon_b))

  def test_eval_does_not_sample_and_train_does(self):
    history = _history()
    model = TerrainVAE(
      history_dim=47, history_len=4, latent_dim=16, feature_dim=32, hidden=64
    ).eval()
    z_eval_a = model(history)[0]
    z_eval_b = model(history)[0]
    torch.testing.assert_close(z_eval_a, z_eval_b)

    model.train()
    z_train_a = model(history)[0]
    z_train_b = model(history)[0]
    self.assertFalse(torch.allclose(z_train_a, z_train_b))

  def test_logvar_is_clamped(self):
    history = _history()
    model = TerrainVAE(
      history_dim=47,
      history_len=4,
      latent_dim=8,
      feature_dim=8,
      hidden=16,
      logvar_min=-10.0,
      logvar_max=4.0,
    ).eval()
    with torch.no_grad():
      model.logvar_head.weight.zero_()
      model.logvar_head.bias.fill_(1_000.0)
      logvar_high = model(history)[2]
      model.logvar_head.bias.fill_(-1_000.0)
      logvar_low = model(history)[2]
    self.assertLessEqual(float(logvar_high.max()), 4.0)
    self.assertGreaterEqual(float(logvar_high.min()), 4.0 - 1e-5)
    self.assertGreaterEqual(float(logvar_low.min()), -10.0)
    self.assertLessEqual(float(logvar_low.max()), -10.0 + 1e-5)

  def test_invalid_dimensions_raise(self):
    invalid_cases = (
      {"history_dim": 0},
      {"history_dim": -3},
      {"history_dim": 47, "history_len": 0},
      {"history_dim": 47, "latent_dim": 0},
      {"history_dim": 47, "latent_dim": -1},
      {"history_dim": 47, "feature_dim": 0},
      {"history_dim": 47, "feature_dim": -8},
      {"history_dim": 47, "hidden": 0},
      {"history_dim": 47, "hidden": (64, 0)},
      {"history_dim": 47, "beta": -0.5},
      {"history_dim": 47, "logvar_min": 4.0, "logvar_max": 4.0},
    )
    for kwargs in invalid_cases:
      with self.assertRaises(ValueError, msg=str(kwargs)):
        TerrainVAE(**kwargs)

  def test_history_shape_validation(self):
    model = TerrainVAE(history_dim=47, history_len=4, latent_dim=16, feature_dim=16)
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 3, 47))  # wrong frame count
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 4, 46))  # wrong per-frame dim
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 4, 47, 1))  # extra dim
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 3 * 47))  # wrong flat length


class HeightAttentionEncoderTests(unittest.TestCase):
  def _model(self, **overrides):
    kwargs = dict(
      feature_dim=48,
      latent_dim=16,
      hidden=32,
      num_samples=8,
      num_heads=4,
      image_hw=(8, 8),
    )
    kwargs.update(overrides)
    return HeightAttentionEncoder(**kwargs)

  def test_flat_and_grid_inputs_have_same_output(self):
    model = self._model().eval()
    flat = _scan(batch=3, side=8)  # (3, 64)
    grid = flat.reshape(3, 1, 8, 8)

    out_flat = model(flat)
    out_grid = model(grid)
    self.assertEqual(tuple(out_flat.shape), (3, 48))
    self.assertEqual(tuple(out_grid.shape), (3, 48))
    torch.testing.assert_close(out_flat, out_grid)

    # (B, H, W) without the explicit channel axis works too.
    torch.testing.assert_close(model(flat.reshape(3, 8, 8)), out_flat)

    # eval() is deterministic.
    torch.testing.assert_close(model(flat), out_flat)

  def test_flat_input_without_image_hw_infers_square(self):
    model = HeightAttentionEncoder(
      feature_dim=16, latent_dim=8, hidden=16, num_samples=4, num_heads=2
    ).eval()
    self.assertEqual(tuple(model(_scan(batch=2, side=6)).shape), (2, 16))

  def test_non_square_scan_with_image_hw(self):
    model = HeightAttentionEncoder(
      feature_dim=24,
      latent_dim=8,
      hidden=16,
      num_samples=6,
      num_heads=2,
      image_hw=(8, 10),
    ).eval()
    scan = torch.randn(2, 8 * 10)
    self.assertEqual(tuple(model(scan).shape), (2, 24))
    self.assertEqual(tuple(model(scan.reshape(2, 1, 8, 10)).shape), (2, 24))
    self.assertEqual(tuple(model.sample_locations(scan).shape), (2, 6, 2))

  def test_gradients_flow_to_parameters(self):
    model = self._model().train()
    scan = _scan(batch=3, side=8)
    model(scan).sum().backward()

    grad_mass = 0.0
    for name, parameter in model.named_parameters():
      self.assertIsNotNone(parameter.grad, name)
      self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
      grad_mass += float(parameter.grad.abs().sum())
    self.assertGreater(grad_mass, 0.0)

  def test_sampling_locations_are_bounded_and_differentiable(self):
    model = self._model().eval()
    scan = _scan(batch=3, side=8)
    locations = model.sample_locations(scan)
    self.assertEqual(tuple(locations.shape), (3, 8, 2))
    self.assertLessEqual(float(locations.detach().abs().max()), 1.0)

    pixel_locations = model.sample_locations(scan, pixel=True).detach()
    self.assertEqual(tuple(pixel_locations.shape), (3, 8, 2))
    self.assertGreaterEqual(float(pixel_locations[..., 0].min()), 0.0)
    self.assertLessEqual(float(pixel_locations[..., 0].max()), 7.0)
    self.assertGreaterEqual(float(pixel_locations[..., 1].min()), 0.0)
    self.assertLessEqual(float(pixel_locations[..., 1].max()), 7.0)

    model.train()
    scan.requires_grad_(True)
    model(scan).sum().backward()
    self.assertIsNotNone(scan.grad)
    self.assertTrue(bool(torch.isfinite(scan.grad).all()))

  def test_state_conditioned_sampling(self):
    model = HeightAttentionEncoder(
      feature_dim=16,
      latent_dim=8,
      hidden=16,
      num_samples=4,
      num_heads=2,
      image_hw=(6, 6),
      state_dim=12,
    ).eval()
    scan = _scan(batch=3, side=6)
    state_a = torch.randn(3, 12, generator=torch.Generator().manual_seed(1))
    state_b = torch.randn(3, 12, generator=torch.Generator().manual_seed(2))

    out_a = model(scan, state_a)
    out_b = model(scan, state_b)
    self.assertEqual(tuple(out_a.shape), (3, 16))
    self.assertFalse(torch.allclose(out_a, out_b))

    with self.assertRaises(ValueError):
      model(scan)  # missing state
    with self.assertRaises(ValueError):
      model(scan, torch.zeros(3, 11))  # wrong state dim
    with self.assertRaises(ValueError):
      model(scan, torch.zeros(4, 12))  # wrong batch

  def test_state_rejected_when_not_configured(self):
    model = HeightAttentionEncoder(
      feature_dim=8,
      latent_dim=4,
      hidden=8,
      num_samples=2,
      num_heads=2,
      image_hw=(4, 4),
    )
    with self.assertRaises(ValueError):
      model(_scan(batch=2, side=4), torch.zeros(2, 12))

  def test_invalid_dimensions_raise(self):
    base = dict(
      feature_dim=16,
      latent_dim=8,
      hidden=16,
      num_samples=4,
      num_heads=2,
      image_hw=(8, 8),
    )
    invalid_cases = (
      {"feature_dim": 0},
      {"feature_dim": -8},
      {"latent_dim": 0},
      {"latent_dim": -1},
      {"hidden": 0},
      {"num_samples": 0},
      {"num_heads": 0},
      {"latent_dim": 10, "num_heads": 4},  # not divisible
      {"state_dim": 0},
      {"offset_scale": -0.1},
      {"image_hw": (0, 8)},
    )
    for overrides in invalid_cases:
      kwargs = {**base, **overrides}
      with self.assertRaises(ValueError, msg=str(overrides)):
        HeightAttentionEncoder(**kwargs)

  def test_scan_shape_validation(self):
    model = self._model()
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 50))  # not a perfect square, no image_hw
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 63))  # 63 cells vs image_hw 8 * 8 = 64
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 2, 8, 8))  # two channels
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 1, 8, 9))  # spatial shape disagrees with image_hw
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 8))  # too few / mismatched cells
    with self.assertRaises(ValueError):
      model(torch.zeros(2, 1, 2, 3, 4))  # too many dims


if __name__ == "__main__":
  unittest.main()
