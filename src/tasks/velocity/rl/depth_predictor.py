"""WMP-style depth predictor for mjlab: height map + proprioception -> depth image.

Method reference: WMP (World Model-based Perception for Visual Legged Locomotion,
ICRA 2025), ``rsl_rl/modules/depth_predictor.py``: an MLP encoder over
``[forward heightmap, proprioception]`` -> 256 -> 128 -> ``4 x 4 x C`` tokens,
followed by a transposed-convolution decoder that upsamples the tokens into a
depth image. WMP trains this predictor on the environments that *do* have a
camera so the remaining environments can synthesise the same perception input.

Why this module exists
----------------------
MGDP's perception encoder (``src/tasks/velocity/mdp/mgdp_perception.py``) consumes
a normalised depth image, which needs a depth camera or a ray sensor. An
environment (or a real robot) without one can instead run this predictor:
``height_scan`` + the 47-dim proprioception frame -> depth image. The predicted
image can either

  * replace the camera input of the frozen MGDP encoder (student / camera-free
    rollout), or
  * act as an auxiliary regularisation target while pretraining the depth
    encoder (``scripts/pretrain_mgdp.py`` already supervises it with real depth).

Conventions (must match the MGDP task in
``src/tasks/velocity/config/lgdog1/mgdp_env_cfg.py``)
----------------------------------------------------
  * depth:            24 x 32 (``DEPTH_HW``), normalised ``metres / DEPTH_MAX``
                      with ``DEPTH_MAX = 2.0 m`` and the encoder's
                      ``depth_min = 0.1 m`` clamp, so values live in [0.05, 1].
  * height:           187 dims (17 x 11 grid, 1.6 x 1.0 m, 0.1 m resolution),
                      normalised as ``height_scan / 5.0`` clamped to [-1, 1]
                      (5.0 m = ``terrain_scan.max_distance``, i.e. the same scale
                      the velocity task applies to its 187-dim height_scan term).
  * proprioception:   47 dims, in actor-observation order
                      ang_vel(3) + projected_gravity(3) + command(3) + phase(2)
                      + joint_pos(12) + joint_vel(12) + last_action(12).

The module is deliberately pure PyTorch: it has no mjlab / MuJoCo imports so the
unit tests (``tests/test_depth_predictor.py``) and offline training run on CPU.
Data collection lives in ``scripts/train_depth_predictor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Mirrors of the MGDP depth contract (kept local so this file stays torch-only).
DEPTH_HW = (24, 32)
HEIGHT_DIM = 187
PROP_DIM = 47
BASE_HW = (4, 4)
HEIGHT_SCAN_MAX_DISTANCE = 5.0
CHECKPOINT_FORMAT = "mjlab_depth_predictor_v1"


def _positive_int(value: int, name: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value < 1:
    raise ValueError(f"{name} must be a positive int, got {value!r}")
  return int(value)


def _as_tuple(values, name: str, length: int = 2) -> tuple[int, ...]:
  if isinstance(values, int) and not isinstance(values, bool):
    values = (values,)
  try:
    result = tuple(int(v) for v in values)
  except TypeError as exc:  # not iterable
    raise ValueError(f"{name} must be an int or an int tuple, got {values!r}") from exc
  if len(result) != length:
    raise ValueError(f"{name} must have {length} entries, got {values!r}")
  for value in result:
    _positive_int(value, name)
  return result


def _channel_layer_norm(channels: int, eps: float = 1e-3) -> nn.Module:
  """WMP's ``ImgChLayerNorm``: LayerNorm over the channel axis of a (B,C,H,W)."""

  class _ChannelLayerNorm(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

  return _ChannelLayerNorm()


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Sobel gradients of a (B, C, H, W) tensor.

  ``padding_mode='replicate'`` matters for the edge loss: with zero padding a
  constant image would produce non-zero gradients along its border, while a flat
  floor must have zero edge energy. Replicate padding keeps flat regions exactly
  flat, so a constant depth image yields exactly zero Sobel gradients.
  """
  if x.ndim != 4:
    raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(x.shape)}")
  channels = x.shape[1]
  kernel_x = x.new_tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
  ).view(1, 1, 3, 3)
  kernel_y = x.new_tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
  ).view(1, 1, 3, 3)
  kernel_x = kernel_x.repeat(channels, 1, 1, 1)
  kernel_y = kernel_y.repeat(channels, 1, 1, 1)
  # ``F.conv2d`` has no ``padding_mode`` argument, so replicate-pad by hand.
  padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
  gx = F.conv2d(padded, kernel_x, groups=channels)
  gy = F.conv2d(padded, kernel_y, groups=channels)
  return gx, gy


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Sobel edge loss: L1 distance between predicted and target image gradients."""
  if pred.shape != target.shape:
    raise ValueError(
      f"edge loss expects matching shapes, got {tuple(pred.shape)} and "
      f"{tuple(target.shape)}"
    )
  gx_pred, gy_pred = _sobel_gradients(pred)
  gx_target, gy_target = _sobel_gradients(target)
  return F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)


def depth_predictor_loss(
  pred: torch.Tensor,
  target: torch.Tensor,
  edge_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """MSE + ``edge_weight`` * Sobel edge loss; returns (total, mse, edge)."""
  if edge_weight < 0.0:
    raise ValueError(f"edge_weight must be non-negative, got {edge_weight!r}")
  mse = F.mse_loss(pred, target)
  edge = edge_loss(pred, target)
  return mse + edge_weight * edge, mse, edge


class _DepthDecoder(nn.Module):
  """Transposed-convolution decoder from the ``4 x 4 x C`` embedding to depth.

  WMP starts at a ``minres x minres`` grid and halves the channel width at each
  ``ConvTranspose2d(kernel_size=4, stride=2, padding=1)`` stage, ending on a
  1-channel image. The stage count is derived from ``out_hw`` so a different
  depth resolution only changes the constructor arguments.

  ``out_hw = (24, 32)`` is not a power-of-two multiple of the ``4 x 4`` base
  grid (24 / 4 = 6), so the last upsampled grid (32 x 32) is bilinearly resized
  to the requested size. The single interpolation is applied to *logits*, before
  the sigmoid, so it cannot push the output outside [0, 1].
  """

  def __init__(
    self,
    base_channels: int,
    base_hw: tuple[int, int] = BASE_HW,
    out_hw: tuple[int, int] = DEPTH_HW,
  ):
    super().__init__()
    self.out_hw = tuple(out_hw)
    stages = 0
    height, width = base_hw
    while height < out_hw[0] or width < out_hw[1]:
      height, width = height * 2, width * 2
      stages += 1

    if stages == 0:
      # Target is no larger than the token grid: a 1x1 conv is enough.
      self.blocks: nn.Module = nn.Sequential(nn.Conv2d(base_channels, 1, 1))
      return

    channels = [base_channels]
    for _ in range(stages - 1):
      channels.append(max(1, channels[-1] // 2))

    blocks: list[nn.Module] = []
    for index, in_channels in enumerate(channels):
      last = index == len(channels) - 1
      out_channels = 1 if last else channels[index + 1]
      blocks.append(
        nn.ConvTranspose2d(
          in_channels,
          out_channels,
          4,
          stride=2,
          padding=1,
          bias=last,
        )
      )
      if not last:
        blocks.append(_channel_layer_norm(out_channels))
        blocks.append(nn.ELU())
    self.blocks = nn.Sequential(*blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.blocks(x)
    if tuple(x.shape[-2:]) != self.out_hw:
      x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
    return x


class DepthPredictor(nn.Module):
  """MLP encoder + transposed-conv decoder predicting a normalised depth image.

  Example
  -------
  >>> predictor = DepthPredictor()
  >>> depth = predictor(height_scan, proprio)   # (B, 1, 24, 32), values in [0, 1]
  """

  def __init__(
    self,
    height_dim: int = HEIGHT_DIM,
    prop_dim: int = PROP_DIM,
    hidden: tuple[int, ...] | int = (256, 128),
    base_channels: int = 64,
    out_hw: tuple[int, int] = DEPTH_HW,
  ):
    super().__init__()
    self.height_dim = _positive_int(height_dim, "height_dim")
    self.prop_dim = _positive_int(prop_dim, "prop_dim")
    if isinstance(hidden, int) and not isinstance(hidden, bool):
      hidden = (hidden,)
    try:
      hidden = tuple(int(width) for width in hidden)
    except TypeError as exc:
      raise ValueError(f"hidden must be an int or int tuple, got {hidden!r}") from exc
    if not hidden:
      raise ValueError("hidden must contain at least one layer width")
    for width in hidden:
      _positive_int(width, "hidden")
    self.hidden = hidden
    self.base_channels = _positive_int(base_channels, "base_channels")
    self.out_hw = _as_tuple(out_hw, "out_hw")
    self.base_hw = BASE_HW

    encoder: list[nn.Module] = [
      nn.Linear(self.height_dim + self.prop_dim, hidden[0]),
      nn.ELU(),
    ]
    for index in range(1, len(hidden)):
      encoder.extend([nn.Linear(hidden[index - 1], hidden[index]), nn.ELU()])
    embedding_dim = self.base_hw[0] * self.base_hw[1] * self.base_channels
    encoder.append(nn.Linear(hidden[-1], embedding_dim))
    self.encoder = nn.Sequential(*encoder)
    self.decoder = _DepthDecoder(self.base_channels, self.base_hw, self.out_hw)

  # ------------------------------------------------------------------ #
  # Inference
  # ------------------------------------------------------------------ #
  def _flatten(self, value: torch.Tensor, expected: int, name: str) -> torch.Tensor:
    if value.ndim == 0:
      raise ValueError(f"{name} must not be a scalar")
    if value.ndim > 2:
      value = value.reshape(value.shape[0], -1)
    elif value.ndim == 1:
      value = value.unsqueeze(0)
    if value.shape[-1] != expected:
      raise ValueError(
        f"{name} has {value.shape[-1]} features, expected {expected}"
      )
    return value

  def forward(self, height: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    """height: (B, 187) ([B, 17, 11] / [B, 1, 17, 11] accepted); prop: (B, 47).

    Returns a (B, 1, 24, 32) depth image with values in [0, 1].
    """
    height = self._flatten(height, self.height_dim, "height")
    prop = self._flatten(prop, self.prop_dim, "prop")
    if height.shape[0] != prop.shape[0]:
      raise ValueError(
        f"height/prop batch mismatch: {height.shape[0]} vs {prop.shape[0]}"
      )
    features = self.encoder(torch.cat([height, prop], dim=-1))
    features = features.reshape(features.shape[0], self.base_channels, *self.base_hw)
    return torch.sigmoid(self.decoder(features))

  def losses(
    self,
    depth: torch.Tensor,
    height: torch.Tensor,
    prop: torch.Tensor,
    edge_weight: float = 0.1,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict and score: returns (total, mse, edge) for ``depth`` targets."""
    return depth_predictor_loss(self.forward(height, prop), depth, edge_weight)

  # ------------------------------------------------------------------ #
  # Introspection
  # ------------------------------------------------------------------ #
  def parameter_count(self) -> int:
    """Number of scalar parameters (the number WMP reports for the predictor)."""
    return sum(parameter.numel() for parameter in self.parameters())

  @property
  def embedding_dim(self) -> int:
    return self.base_hw[0] * self.base_hw[1] * self.base_channels

  def architecture(self) -> dict:
    return {
      "height_dim": self.height_dim,
      "prop_dim": self.prop_dim,
      "hidden": list(self.hidden),
      "base_channels": self.base_channels,
      "base_hw": list(self.base_hw),
      "out_hw": list(self.out_hw),
      "embedding_dim": self.embedding_dim,
      "parameter_count": self.parameter_count(),
    }

  # ------------------------------------------------------------------ #
  # Checkpoint helpers
  # ------------------------------------------------------------------ #
  def save(self, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    payload = {
      "format": CHECKPOINT_FORMAT,
      "architecture": self.architecture(),
      "state_dict": {
        name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
      },
    }
    if extra:
      payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path

  def _check_architecture(self, payload: dict, path: str | Path) -> None:
    stored = payload.get("architecture") or {}
    expected = self.architecture()
    mismatched = {}
    for key in ("height_dim", "prop_dim", "hidden", "base_channels", "out_hw"):
      stored_value = stored.get(key)
      if stored_value is None:
        continue
      stored_value = list(stored_value) if isinstance(stored_value, (list, tuple)) else stored_value
      expected_value = expected[key]
      expected_value = list(expected_value) if isinstance(expected_value, (list, tuple)) else expected_value
      if stored_value != expected_value:
        mismatched[key] = (stored_value, expected_value)
    if mismatched:
      raise ValueError(
        f"{path} was saved for a different architecture: {mismatched}. "
        "Use DepthPredictor.from_checkpoint(path) to rebuild the saved model."
      )

  def load(self, path: str | Path, map_location="cpu", strict: bool = True) -> "DepthPredictor":
    """Load weights into this instance in place (architecture must match)."""
    payload = _read_checkpoint(path, map_location)
    self._check_architecture(payload, path)
    self.load_state_dict(payload["state_dict"], strict=strict)
    return self

  @classmethod
  def from_checkpoint(cls, path: str | Path, map_location="cpu") -> "DepthPredictor":
    """Rebuild a predictor (architecture + weights) from a ``save()`` file."""
    payload = _read_checkpoint(path, map_location)
    architecture = payload.get("architecture") or {}
    model = cls(
      height_dim=architecture.get("height_dim", HEIGHT_DIM),
      prop_dim=architecture.get("prop_dim", PROP_DIM),
      hidden=architecture.get("hidden", (256, 128)),
      base_channels=architecture.get("base_channels", 64),
      out_hw=architecture.get("out_hw", DEPTH_HW),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _read_checkpoint(path: str | Path, map_location="cpu") -> dict:
  payload = torch.load(str(path), map_location=map_location, weights_only=False)
  if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
    raise ValueError(
      f"{path} is not an mjlab depth-predictor checkpoint "
      f"(format={payload.get('format') if isinstance(payload, dict) else None!r})"
    )
  if "state_dict" not in payload:
    raise ValueError(f"{path} has no state_dict")
  return payload


@dataclass
class DepthPredictorHistory:
  """Per-epoch training history returned by :func:`train_depth_predictor`."""

  loss: list[float] = field(default_factory=list)
  mse: list[float] = field(default_factory=list)
  edge: list[float] = field(default_factory=list)
  model: "DepthPredictor | None" = None

  def __len__(self) -> int:
    return len(self.loss)

  @property
  def epochs(self) -> int:
    return len(self.loss)

  @property
  def final_mse(self) -> float:
    return self.mse[-1] if self.mse else float("nan")

  def as_dict(self) -> dict[str, list[float]]:
    return {
      "loss": list(self.loss),
      "mse": list(self.mse),
      "edge": list(self.edge),
    }


def _flatten_rows(value: torch.Tensor, name: str) -> torch.Tensor:
  """(N, ...) input rows -> (N, features); a single 1-D row becomes a batch of 1."""
  tensor = torch.as_tensor(value)
  if tensor.ndim == 0:
    raise ValueError(f"{name} must not be a scalar")
  if tensor.ndim == 1:
    tensor = tensor.unsqueeze(0)
  return tensor.reshape(tensor.shape[0], -1).float()


def _depth_targets(depth_stack: torch.Tensor, out_hw=None) -> torch.Tensor:
  """Depth targets -> (N, 1, H, W); validates the (N, H, W) / (N, 1, H, W) input."""
  depth = torch.as_tensor(depth_stack)
  if depth.ndim == 3:
    depth = depth.unsqueeze(1)
  if depth.ndim != 4 or depth.shape[1] != 1:
    raise ValueError(
      f"depth_stack must be (N, 1, H, W) or (N, H, W), got {tuple(depth.shape)}"
    )
  if depth.shape[0] < 1:
    raise ValueError("depth_stack is empty")
  if out_hw is not None and tuple(depth.shape[-2:]) != tuple(out_hw):
    raise ValueError(
      f"depth_stack has shape {tuple(depth.shape[-2:])} but the predictor "
      f"emits {tuple(out_hw)}"
    )
  return depth.float()


def train_depth_predictor(
  depth_stack: torch.Tensor,
  height: torch.Tensor,
  prop: torch.Tensor,
  *,
  epochs: int = 50,
  lr: float = 3e-4,
  batch_size: int = 64,
  edge_weight: float = 0.1,
  log_every: int = 10,
  device: str | torch.device | None = None,
  seed: int | None = None,
  model: DepthPredictor | None = None,
  verbose: bool = True,
) -> DepthPredictorHistory:
  """Train a :class:`DepthPredictor` on collected (height, prop) -> depth pairs.

  Data conventions (see :func:`scripts.train_depth_predictor.collect_depth_data`):
    * ``depth_stack``: (N, 1, 24, 32) or (N, 24, 32) target depth in [0, 1]
      (metres / ``DEPTH_MAX``),
    * ``height``: (N, 187) normalised ``height_scan / 5.0`` in [-1, 1]
      (a (N, 17, 11) / (N, 1, 17, 11) grid is flattened for you),
    * ``prop``: (N, 47) proprioception, actor-observation order.

  Loss: ``MSE + edge_weight * Sobel edge L1``. Returns one loss/mse/edge value
  per epoch plus the trained model (``history.model``).
  """
  if not isinstance(epochs, int) or epochs < 1:
    raise ValueError(f"epochs must be a positive int, got {epochs!r}")
  if lr <= 0.0:
    raise ValueError(f"lr must be positive, got {lr!r}")
  if batch_size < 1:
    raise ValueError(f"batch_size must be positive, got {batch_size!r}")
  if edge_weight < 0.0:
    raise ValueError(f"edge_weight must be non-negative, got {edge_weight!r}")
  if seed is not None:
    torch.manual_seed(seed)

  resolved_device = torch.device(
    device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
  )
  depth = _depth_targets(depth_stack)
  height = _flatten_rows(height, "height")
  prop = _flatten_rows(prop, "prop")
  if model is None:
    model = DepthPredictor(
      height_dim=height.shape[-1],
      prop_dim=prop.shape[-1],
      out_hw=tuple(depth.shape[-2:]),
    )
  else:
    depth = _depth_targets(depth_stack, model.out_hw)
  model = model.to(resolved_device)

  if height.shape[0] != depth.shape[0] or prop.shape[0] != depth.shape[0]:
    raise ValueError(
      "depth/height/prop batch mismatch: "
      f"{depth.shape[0]}/{height.shape[0]}/{prop.shape[0]}"
    )
  if height.shape[-1] != model.height_dim:
    raise ValueError(
      f"height has {height.shape[-1]} features but the predictor expects "
      f"{model.height_dim}"
    )
  if prop.shape[-1] != model.prop_dim:
    raise ValueError(
      f"prop has {prop.shape[-1]} features but the predictor expects "
      f"{model.prop_dim}"
    )
  samples = depth.shape[0]
  generator = (
    torch.Generator().manual_seed(seed) if seed is not None else None
  )

  history = DepthPredictorHistory(model=model)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()
  for epoch in range(1, epochs + 1):
    permutation = torch.randperm(samples, generator=generator)
    sums = {"loss": 0.0, "mse": 0.0, "edge": 0.0}
    weight = 0
    for start in range(0, samples, batch_size):
      indices = permutation[start : start + batch_size]
      target = depth[indices].to(resolved_device)
      total, mse, edge = depth_predictor_loss(
        model(height[indices].to(resolved_device), prop[indices].to(resolved_device)),
        target,
        edge_weight,
      )
      optimizer.zero_grad(set_to_none=True)
      total.backward()
      optimizer.step()
      count = int(indices.numel())
      weight += count
      sums["loss"] += float(total.detach()) * count
      sums["mse"] += float(mse.detach()) * count
      sums["edge"] += float(edge.detach()) * count

    history.loss.append(sums["loss"] / weight)
    history.mse.append(sums["mse"] / weight)
    history.edge.append(sums["edge"] / weight)
    if verbose and log_every and (epoch % log_every == 0 or epoch == epochs):
      print(
        f"[predictor] epoch {epoch:4d}/{epochs}  "
        f"loss={history.loss[-1]:.6f}  mse={history.mse[-1]:.6f}  "
        f"edge={history.edge[-1]:.6f}",
        flush=True,
      )
  return history
