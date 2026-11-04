"""CPU tests for the MGDP perception port (no simulator required)."""

from __future__ import annotations

import tempfile
import unittest
from statistics import mean
from pathlib import Path
from types import SimpleNamespace

import torch

from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg
from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature

RELEASED_WM = Path(
  r"E:\足式机器人\MGDP - Mastering a Generalized Depth Perception Model for Quadruped Locomotion"
  r"\代码\arclab-hku__MGDP\legged_gym\models\MGDP\stage2\resume\stage1_nn\wm_best.pt"
)


def _batch(n=4, frames=2, hw=(16, 16), height_hw=(17, 11), seed=0):
  g = torch.Generator().manual_seed(seed)
  depth = torch.rand(n, frames, *hw, generator=g)
  height = torch.rand(n, 1, *height_hw, generator=g) * 2 - 1
  return depth, height


class _FakeSensor:
  def __init__(self, distances):
    self.data = SimpleNamespace(distances=distances)


class _FakeEnv:
  def __init__(self, batch, num_rays, device="cpu"):
    self.scene = {"depth_rays": _FakeSensor(torch.rand(batch, num_rays) * 4 + 0.2)}
    self.device = device
    self.episode_length_buf = torch.zeros(batch, dtype=torch.long)


class MGPDPerceptionTests(unittest.TestCase):
  def test_cfg_dict_roundtrip(self):
    cfg = MGPDPerceptionCfg(depth_hw=(24, 24), feature_dim=64)
    restored = MGPDPerceptionCfg.from_dict(cfg.to_dict())
    self.assertEqual(restored.depth_hw, (24, 24))
    self.assertEqual(restored.depth_pixels, 24 * 24)

  def test_feature_shapes(self):
    cfg = MGPDPerceptionCfg()
    model = MGPDPerception(cfg).eval()
    depth, _ = _batch(3)
    feature, hidden = model.feature(depth, None)
    self.assertEqual(tuple(feature.shape), (3, cfg.feature_dim))
    self.assertEqual(tuple(hidden.shape), (1, 3, cfg.gru_hidden))
    feature2, hidden2 = model.feature(depth, hidden)
    self.assertEqual(tuple(feature2.shape), (3, cfg.feature_dim))
    self.assertFalse(torch.allclose(feature, feature2))

  def test_losses_backward_and_decrease(self):
    # Isolate initialization AND noisy-depth augmentation from suite order,
    # without changing the RNG state seen by the other tests.
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(0)
      model = MGPDPerception(MGPDPerceptionCfg()).train()
      depth, height = _batch(16)
      # Match the pretraining defaults; the old 3e-3 was 30x the training LR
      # and could destabilize the recurrent loss's moving encoder target.
      opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
      before = model.memory.weight_ih_l0.detach().clone()
      totals = []
      for _ in range(40):
        loss, metrics = model.losses(depth, height)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.assertTrue(torch.isfinite(norm))
        opt.step()
        totals.append(metrics["total"])
      self.assertFalse(torch.equal(before, model.memory.weight_ih_l0))
      # Compare windows, not two individual noisy training steps. Require a
      # meaningful improvement so a frozen/broken optimizer cannot pass.
      self.assertLess(mean(totals[-5:]), 0.8 * mean(totals[:5]))

  def test_checkpoint_roundtrip(self):
    cfg = MGPDPerceptionCfg()
    model = MGPDPerception(cfg).eval()
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "mgdp.pt"
      model.save(path, extra={"note": "test"})
      loaded = MGPDPerception.from_checkpoint(path)
      depth, _ = _batch(2)
      a, _ = model.feature(depth, None)
      b, _ = loaded.feature(depth, None)
      torch.testing.assert_close(a, b)

  @unittest.skipUnless(RELEASED_WM.is_file(), "released MGDP checkpoint not present")
  def test_released_weights_load(self):
    cfg = MGPDPerceptionCfg()
    model = MGPDPerception(cfg)
    loaded = model.load_released_weights(RELEASED_WM)
    self.assertIn("image_encoder", loaded)
    self.assertIn("memory_encoder", loaded)
    depth, _ = _batch(2)
    feature, _ = model.eval().feature(depth, None)
    self.assertEqual(tuple(feature.shape), (2, cfg.feature_dim))

  def test_observation_term_shapes_and_validation(self):
    cfg = MGPDPerceptionCfg()
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = Path(tmp) / "mgdp.pt"
      MGPDPerception(cfg).save(ckpt)
      env = _FakeEnv(batch=5, num_rays=cfg.depth_pixels)
      feature = mgdp_depth_feature(env, "depth_rays", cfg, str(ckpt))
      self.assertEqual(tuple(feature.shape), (5, cfg.feature_dim))
      self.assertTrue(torch.isfinite(feature).all())
      # A miss (-1) must be handled like the far clip, not a negative depth.
      env.scene["depth_rays"].data.distances[0, :] = -1.0
      feature = mgdp_depth_feature(env, "depth_rays", cfg, str(ckpt))
      self.assertTrue(torch.isfinite(feature).all())

  def test_observation_term_rejects_wrong_ray_count(self):
    cfg = MGPDPerceptionCfg()
    with tempfile.TemporaryDirectory() as tmp:
      ckpt = Path(tmp) / "mgdp.pt"
      MGPDPerception(cfg).save(ckpt)
      env = _FakeEnv(batch=2, num_rays=100)
      with self.assertRaisesRegex(ValueError, "rays"):
        mgdp_depth_feature(env, "depth_rays", cfg, str(ckpt))


if __name__ == "__main__":
  unittest.main()
