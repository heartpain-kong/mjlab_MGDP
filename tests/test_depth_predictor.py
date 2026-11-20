"""CPU tests for the WMP-style depth predictor.

No simulator, MuJoCo or environment is built: the predictor is pure PyTorch and
is tested on synthetic depth / height-scan / proprioception tensors.

The module is loaded from disk instead of importing
``src.tasks.velocity.rl.depth_predictor`` through the package, because the
``rl`` package ``__init__`` pulls in rsl_rl/mjlab (and with it the simulator
stack) even though the predictor itself only needs torch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_MODULE_PATH = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "tasks"
  / "velocity"
  / "rl"
  / "depth_predictor.py"
)
_spec = importlib.util.spec_from_file_location("mjlab_depth_predictor", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(_module)

DepthPredictor = _module.DepthPredictor
depth_predictor_loss = _module.depth_predictor_loss
edge_loss = _module.edge_loss
train_depth_predictor = _module.train_depth_predictor


def _predictor(**overrides):
  torch.manual_seed(0)
  kwargs = dict(height_dim=187, prop_dim=47, out_hw=(24, 32))
  kwargs.update(overrides)
  return DepthPredictor(**kwargs)


def _inputs(batch=2, seed=0):
  generator = torch.Generator().manual_seed(seed)
  return (
    torch.randn(batch, 187, generator=generator),
    torch.randn(batch, 47, generator=generator),
  )


def _target(batch=2, seed=0):
  generator = torch.Generator().manual_seed(seed)
  return torch.rand(batch, 1, 24, 32, generator=generator)


# --------------------------------------------------------------------------- #
# Interface: shape, range, parameters
# --------------------------------------------------------------------------- #
def test_forward_shape_and_range():
  model = _predictor().eval()
  height, prop = _inputs(batch=2)
  with torch.no_grad():
    depth = model(height, prop)
  assert depth.shape == (2, 1, 24, 32)
  assert bool(torch.isfinite(depth).all())
  assert float(depth.min()) >= 0.0
  assert float(depth.max()) <= 1.0


def test_forward_accepts_grid_height():
  model = _predictor().eval()
  height, prop = _inputs(batch=3)
  with torch.no_grad():
    flat = model(height, prop)
    grid = model(height.reshape(3, 1, 17, 11), prop)
  torch.testing.assert_close(flat, grid)


def test_parameter_count():
  model = _predictor()
  expected = sum(parameter.numel() for parameter in model.parameters())
  assert model.parameter_count() == expected
  assert model.parameter_count() > 0


def test_forward_rejects_wrong_dimensions():
  model = _predictor()
  height, prop = _inputs()
  try:
    model(torch.zeros(2, 186), prop)
  except ValueError:
    pass
  else:  # pragma: no cover - failure path
    raise AssertionError("wrong height_dim must raise ValueError")
  try:
    model(height, torch.zeros(2, 46))
  except ValueError:
    pass
  else:  # pragma: no cover - failure path
    raise AssertionError("wrong prop_dim must raise ValueError")


# --------------------------------------------------------------------------- #
# Loss: MSE + Sobel edge, differentiable
# --------------------------------------------------------------------------- #
def test_loss_is_mse_plus_weighted_edge():
  generator = torch.Generator().manual_seed(1)
  pred = torch.rand(2, 1, 24, 32, generator=generator)
  target = torch.rand(2, 1, 24, 32, generator=generator)
  total, mse, edge = depth_predictor_loss(pred, target, edge_weight=0.1)
  torch.testing.assert_close(total, mse + 0.1 * edge)
  torch.testing.assert_close(mse, torch.nn.functional.mse_loss(pred, target))


def test_loss_backpropagates():
  model = _predictor().train()
  height, prop = _inputs(batch=2)
  target = _target(batch=2)
  total, mse, edge = model.losses(target, height, prop)
  for value in (total, mse, edge):
    assert bool(torch.isfinite(value)), value
  total.backward()

  grad_mass = 0.0
  for name, parameter in model.named_parameters():
    assert parameter.grad is not None, name
    assert bool(torch.isfinite(parameter.grad).all()), name
    grad_mass += float(parameter.grad.abs().sum())
  assert grad_mass > 0.0


# --------------------------------------------------------------------------- #
# Edge loss: zero on flat terrain, positive on a step
# --------------------------------------------------------------------------- #
def test_edge_loss_is_zero_on_flat_terrain():
  flat = torch.full((2, 1, 24, 32), 0.5)
  # Identical constant images: the Sobel gradients cancel exactly.
  assert float(edge_loss(flat, flat)) == 0.0
  # Two different constant heights still have no edge energy (constant depth
  # means a flat floor regardless of the actual distance).
  assert float(edge_loss(torch.full_like(flat, 0.2), flat)) < 1e-6


def test_edge_loss_is_positive_on_step():
  target = torch.zeros(1, 1, 24, 32)
  target[..., 16:] = 0.4  # a 20 cm step half-way through the image
  flat_pred = torch.full_like(target, 0.2)
  assert float(edge_loss(flat_pred, target)) > 0.01
  # Predicting the step exactly gives zero edge loss again.
  assert float(edge_loss(target, target)) == 0.0


# --------------------------------------------------------------------------- #
# Checkpoint round-trip
# --------------------------------------------------------------------------- #
def test_save_then_load_roundtrip(tmp_path):
  model = _predictor().eval()
  path = tmp_path / "nested" / "depth_predictor.pt"
  assert path.exists() is False
  model.save(path)
  assert path.is_file()

  loaded = _predictor().eval()
  loaded.load(path)
  for (name_a, value_a), (name_b, value_b) in zip(
    model.state_dict().items(), loaded.state_dict().items()
  ):
    assert name_a == name_b
    torch.testing.assert_close(value_a, value_b)

  height, prop = _inputs(batch=4)
  with torch.no_grad():
    torch.testing.assert_close(model(height, prop), loaded(height, prop))


def test_from_checkpoint_rebuilds_architecture(tmp_path):
  model = DepthPredictor(
    height_dim=187,
    prop_dim=47,
    hidden=(32, 16),
    base_channels=8,
    out_hw=(24, 32),
  )
  path = tmp_path / "small.pt"
  model.save(path, extra={"note": "unit-test"})
  loaded = DepthPredictor.from_checkpoint(path)
  assert loaded.parameter_count() == model.parameter_count()
  assert loaded.architecture() == model.architecture()
  height, prop = _inputs(batch=2)
  with torch.no_grad():
    torch.testing.assert_close(model(height, prop), loaded(height, prop))


def test_load_rejects_mismatched_architecture(tmp_path):
  path = tmp_path / "depth_predictor.pt"
  _predictor().save(path)
  other = _predictor(out_hw=(16, 16))
  try:
    other.load(path)
  except ValueError:
    pass
  else:  # pragma: no cover - failure path
    raise AssertionError("loading a different architecture must raise ValueError")


def test_load_rejects_foreign_checkpoint(tmp_path):
  path = tmp_path / "foreign.pt"
  torch.save({"format": "something_else", "state_dict": {}}, path)
  try:
    _predictor().load(path)
  except ValueError:
    pass
  else:  # pragma: no cover - failure path
    raise AssertionError("loading a foreign checkpoint must raise ValueError")


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def test_train_depth_predictor_history():
  torch.manual_seed(0)
  samples = 24
  depth = _target(batch=samples, seed=3)
  height, prop = _inputs(batch=samples, seed=4)
  model = _predictor()
  history = train_depth_predictor(
    depth,
    height,
    prop,
    epochs=2,
    batch_size=8,
    edge_weight=0.1,
    log_every=1,
    device="cpu",
    seed=0,
    model=model,
    verbose=False,
  )
  assert len(history) == 2
  assert len(history.loss) == len(history.mse) == len(history.edge) == 2
  assert history.as_dict().keys() == {"loss", "mse", "edge"}
  for value in history.loss + history.mse + history.edge:
    assert value == value and value not in (float("inf"), float("-inf"))
  assert history.model is model
  assert history.final_mse == history.mse[-1]
