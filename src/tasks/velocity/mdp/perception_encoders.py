"""Optional perception encoders for the velocity task (pure PyTorch, no MuJoCo).

Two paper-inspired encoding front-ends live here:

* :class:`TerrainVAE` -- DreamWaQ (#6) style ``beta``-VAE.  A short
  proprioceptive history ``(B, H, D)`` is flattened and mapped by an MLP
  encoder to ``(mu, logvar)``; a reparameterised latent ``z`` is decoded back
  to the history.  The ``z`` / reconstruction objective forces the latent to
  encode implicit terrain context ("terrain imagination") without any height
  map.  In DreamWaQ the same encoder is shared with a velocity-estimation head
  (CENet) and trained with an asymmetric actor-critic PPO setup.
* :class:`HeightAttentionEncoder` -- simplified DELTA (#7).  A small conv stem
  turns a height/terrain scan into a feature map, a content- (and optionally
  state-) conditioned offset head predicts a *fixed* number of sampling
  locations, ``grid_sample`` extracts deformable local terrain tokens, and
  multi-head attention pools them into one ``feature_dim`` vector.  Because the
  token count is fixed, encoding cost is independent of map resolution.

Both modules are deliberately dependency-free (``torch`` only), CPU friendly,
and validate their tunable dimensions at construction time so configuration
mistakes fail loudly instead of silently training a degenerate encoder.

Typical usage::

  vae = TerrainVAE(history_dim=47, history_len=4, latent_dim=16)
  z, mu, logvar, recon = vae(history)          # history: (B, 4, 47)
  total, recon_loss, kl_loss = vae.loss(history)

  scan_encoder = HeightAttentionEncoder(feature_dim=64, latent_dim=32)
  terrain_feature = scan_encoder(scan)         # scan: (B, N) or (B, 1, H, W)

These building blocks are *not* wired into ``mdp/observations.py`` yet; see the
module notes in the task report for the intended observation-term integration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HeightAttentionEncoder", "TerrainVAE"]

_LOGVAR_MIN_DEFAULT = -10.0
_LOGVAR_MAX_DEFAULT = 4.0


def _as_widths(hidden: int | Sequence[int], name: str = "hidden") -> tuple[int, ...]:
  """Normalise a ``hidden`` spec (int or sequence of ints) and validate it."""
  if isinstance(hidden, int):
    widths: tuple[int, ...] = (hidden,)
  elif isinstance(hidden, Sequence):
    widths = tuple(int(w) for w in hidden)
  else:
    raise TypeError(
      f"{name} must be an int or a sequence of ints, got {type(hidden).__name__}"
    )
  if len(widths) == 0:
    raise ValueError(f"{name} must contain at least one layer width")
  if any(w <= 0 for w in widths):
    raise ValueError(f"{name} widths must be positive, got {widths}")
  return widths


class TerrainVAE(nn.Module):
  """DreamWaQ-style ``beta``-VAE over a proprioceptive history.

  The encoder is a shared MLP trunk; ``mu``/``logvar`` heads and the decoder
  are the auto-encoding half of CENet.  ``feature_dim`` is the width of the
  shared encoder feature (before the Gaussian heads), ``latent_dim`` the size
  of ``z``, and ``hidden`` the MLP widths (int or sequence of ints).

  Args:
    history_dim: Per-frame observation dimension ``D`` (e.g. 47 or 42).
    history_len: Number of stacked frames ``H`` (default 1).
    latent_dim: Latent ``z`` dimension (DreamWaQ uses 16 or 32).
    feature_dim: Shared encoder feature width feeding the ``mu``/``logvar``
      heads.
    hidden: Encoder/decoder MLP widths; an int or a sequence of ints.
    beta: Weight of the KL term in :meth:`loss` (``beta``-VAE; 1.0 = plain VAE).
    logvar_min: Lower clamp for the predicted log-variance (numerical safety).
    logvar_max: Upper clamp for the predicted log-variance (numerical safety).

  Shapes:
    * ``forward(history) -> (z, mu, logvar, recon)`` with
      ``history: (B, H, D)`` and ``z, mu, logvar: (B, latent_dim)``,
      ``recon: (B, H, D)``.  A flattened ``(B, H * D)`` input is also accepted.
    * ``loss(history) -> (total, recon_loss, kl_loss)``; all three are scalars
      and ``total = recon_loss + beta * kl_loss``.
  """

  def __init__(
    self,
    history_dim: int,
    history_len: int = 1,
    latent_dim: int = 16,
    feature_dim: int = 64,
    hidden: int | Sequence[int] = 128,
    beta: float = 1.0,
    logvar_min: float = _LOGVAR_MIN_DEFAULT,
    logvar_max: float = _LOGVAR_MAX_DEFAULT,
  ) -> None:
    super().__init__()
    if history_dim <= 0:
      raise ValueError(f"history_dim must be positive, got {history_dim}")
    if history_len <= 0:
      raise ValueError(f"history_len must be positive, got {history_len}")
    if latent_dim <= 0:
      raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if feature_dim <= 0:
      raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if beta < 0.0:
      raise ValueError(f"beta must be non-negative, got {beta}")
    if not logvar_min < logvar_max:
      raise ValueError(
        f"logvar_min ({logvar_min}) must be smaller than logvar_max ({logvar_max})"
      )

    self.history_dim = int(history_dim)
    self.history_len = int(history_len)
    self.latent_dim = int(latent_dim)
    self.feature_dim = int(feature_dim)
    self.beta = float(beta)
    self.logvar_min = float(logvar_min)
    self.logvar_max = float(logvar_max)

    widths = _as_widths(hidden)
    flat_dim = self.history_len * self.history_dim

    encoder_layers: list[nn.Module] = []
    prev = flat_dim
    for width in widths:
      encoder_layers += [nn.Linear(prev, width), nn.ELU()]
      prev = width
    encoder_layers += [nn.Linear(prev, self.feature_dim), nn.ELU()]
    self.encoder = nn.Sequential(*encoder_layers)
    self.mu_head = nn.Linear(self.feature_dim, self.latent_dim)
    self.logvar_head = nn.Linear(self.feature_dim, self.latent_dim)

    decoder_layers: list[nn.Module] = []
    prev = self.latent_dim
    for width in widths:
      decoder_layers += [nn.Linear(prev, width), nn.ELU()]
      prev = width
    decoder_layers.append(nn.Linear(prev, flat_dim))
    self.decoder = nn.Sequential(*decoder_layers)

  # ------------------------------------------------------------------ helpers
  def _flatten_history(self, history: torch.Tensor) -> torch.Tensor:
    """Validate ``(B, H, D)`` / ``(B, H * D)`` history and flatten to 2D."""
    if history.ndim == 2:
      expected = self.history_len * self.history_dim
      if history.shape[1] != expected:
        raise ValueError(
          f"flat history must have {expected} features "
          f"(history_len={self.history_len} * history_dim={self.history_dim}), "
          f"got {history.shape[1]}"
        )
      return history
    if history.ndim == 3:
      if history.shape[1:] != (self.history_len, self.history_dim):
        raise ValueError(
          f"history must have shape (B, {self.history_len}, {self.history_dim}), "
          f"got {tuple(history.shape)}"
        )
      return history.reshape(history.shape[0], -1)
    raise ValueError(
      f"history must be (B, H, D) or (B, H * D), got shape {tuple(history.shape)}"
    )

  def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mu, logvar)``; ``logvar`` is clamped for numerical safety."""
    features = self.encoder(self._flatten_history(history))
    mu = self.mu_head(features)
    logvar = self.logvar_head(features).clamp(
      min=self.logvar_min, max=self.logvar_max
    )
    return mu, logvar

  @staticmethod
  def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample ``z = mu + eps * exp(0.5 * logvar)`` (the VAE reparameterisation)."""
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decode ``(B, latent_dim)`` to a reconstructed history ``(B, H, D)``."""
    flat = self.decoder(z)
    return flat.reshape(z.shape[0], self.history_len, self.history_dim)

  def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|history) || N(0, I)), averaged over the batch; always ``>= 0``."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
    return kl.mean().clamp_min(0.0)

  # -------------------------------------------------------------------- public
  def forward(
    self,
    history: torch.Tensor,
    sample: bool | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode then decode ``history``.

    Args:
      history: ``(B, H, D)`` or flattened ``(B, H * D)`` proprioceptive history.
      sample: Whether to reparameterise ``z``.  ``None`` (default) samples in
        training mode and uses ``mu`` in ``eval()`` mode, so evaluation is
        deterministic and deployable.

    Returns:
      ``(z, mu, logvar, recon)`` with ``z/mu/logvar: (B, latent_dim)`` and
      ``recon: (B, H, D)``.
    """
    if sample is None:
      sample = self.training
    mu, logvar = self.encode(history)
    z = self.reparameterize(mu, logvar) if sample else mu
    return z, mu, logvar, self.decode(z)

  def loss(
    self,
    history: torch.Tensor,
    beta: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, recon_loss, kl_loss)`` for a batch of histories.

    ``recon_loss`` is the mean squared error against the input history,
    ``kl_loss`` the mean KL against the standard Gaussian prior, and
    ``total = recon_loss + beta * kl_loss``.  ``beta`` overrides the value set
    at construction time.
    """
    beta_value = self.beta if beta is None else float(beta)
    if beta_value < 0.0:
      raise ValueError(f"beta must be non-negative, got {beta_value}")
    flat = self._flatten_history(history)
    _, mu, logvar, recon = self.forward(history)
    recon_loss = F.mse_loss(recon.reshape(flat.shape[0], -1), flat)
    kl_loss = self.kl_divergence(mu, logvar)
    total = recon_loss + beta_value * kl_loss
    return total, recon_loss, kl_loss


class HeightAttentionEncoder(nn.Module):
  """Simplified DELTA deformable-attention encoder for height/terrain scans.

  Pipeline: conv stem -> feature map -> learned content/state-conditioned 2D
  sampling locations -> ``grid_sample`` deformable tokens (fixed count) ->
  multi-head attention pooling -> ``feature_dim`` vector.

  The sampling locations are predicted from the globally pooled feature map
  (optionally concatenated with a state vector such as base velocity and
  command), i.e. *where to look* is learned end-to-end without any footstep or
  attention supervision.  :meth:`sample_locations` exposes them for
  visualisation.

  Args:
    feature_dim: Output feature width.
    latent_dim: Width of the deformable terrain tokens (also the attention
      embedding dim); must be divisible by ``num_heads``.
    hidden: Conv stem / offset-head width (channels for the first conv).
    num_samples: Fixed number of deformable sampling locations (token count).
      Encoding cost is independent of the map resolution for fixed
      ``num_samples``/patch size.
    num_heads: Number of attention heads used to pool the tokens.
    image_hw: Optional ``(H, W)`` map shape.  Needed for non-square flat scans;
      square flat scans infer it.  If given it must match 4D inputs too.
    state_dim: Optional proprioceptive state/command width.  When set,
      :meth:`forward` requires a ``state`` tensor so that the sampling offsets
      are state-conditioned (DELTA's "state-dependent terrain evidence").
    offset_scale: Maximum absolute displacement of a sampling location in
      normalised coordinates; ``0`` disables deformation.

  Shapes:
    ``forward(x, state=None) -> (B, feature_dim)`` where ``x`` is ``(B, N)``,
    ``(B, H, W)`` or ``(B, 1, H, W)`` and ``state`` is ``(B, state_dim)`` when
    ``state_dim`` is configured.
  """

  def __init__(
    self,
    feature_dim: int = 64,
    latent_dim: int = 32,
    hidden: int = 64,
    num_samples: int = 16,
    num_heads: int = 4,
    image_hw: tuple[int, int] | None = None,
    state_dim: int | None = None,
    offset_scale: float = 0.5,
  ) -> None:
    super().__init__()
    if feature_dim <= 0:
      raise ValueError(f"feature_dim must be positive, got {feature_dim}")
    if latent_dim <= 0:
      raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if hidden <= 0:
      raise ValueError(f"hidden must be positive, got {hidden}")
    if num_samples <= 0:
      raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_heads <= 0:
      raise ValueError(f"num_heads must be positive, got {num_heads}")
    if latent_dim % num_heads != 0:
      raise ValueError(
        f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
      )
    if state_dim is not None and state_dim <= 0:
      raise ValueError(f"state_dim must be positive when set, got {state_dim}")
    if offset_scale < 0.0:
      raise ValueError(f"offset_scale must be non-negative, got {offset_scale}")
    if image_hw is not None:
      if len(image_hw) != 2:
        raise ValueError(f"image_hw must be a (H, W) tuple, got {image_hw!r}")
      if any(int(v) <= 0 for v in image_hw):
        raise ValueError(f"image_hw entries must be positive, got {image_hw!r}")

    self.feature_dim = int(feature_dim)
    self.latent_dim = int(latent_dim)
    self.hidden = int(hidden)
    self.num_samples = int(num_samples)
    self.num_heads = int(num_heads)
    self.image_hw = (
      (int(image_hw[0]), int(image_hw[1])) if image_hw is not None else None
    )
    self.state_dim = int(state_dim) if state_dim is not None else None
    self.offset_scale = float(offset_scale)

    self.stem = nn.Sequential(
      nn.Conv2d(1, self.hidden, kernel_size=3, padding=1),
      nn.ELU(),
      nn.Conv2d(self.hidden, self.latent_dim, kernel_size=3, padding=1),
      nn.ELU(),
    )
    self.offset_head = nn.Sequential(
      nn.Linear(self.latent_dim + (self.state_dim or 0), self.hidden),
      nn.ELU(),
      nn.Linear(self.hidden, self.num_samples * 2),
    )
    self.register_buffer("base_grid", self._make_base_grid(self.num_samples))
    self.norm = nn.LayerNorm(self.latent_dim)
    self.attention = nn.MultiheadAttention(
      self.latent_dim, self.num_heads, batch_first=True
    )
    self.pool_query = nn.Parameter(torch.randn(1, 1, self.latent_dim) * 0.02)
    self.head = nn.Sequential(
      nn.Linear(self.latent_dim, self.hidden),
      nn.ELU(),
      nn.Linear(self.hidden, self.feature_dim),
    )

  @staticmethod
  def _make_base_grid(num_samples: int) -> torch.Tensor:
    """Golden-angle spiral of fixed base sampling positions in ``[-1, 1]^2``."""
    index = torch.arange(num_samples, dtype=torch.float32)
    radius = ((index + 0.5) / num_samples).sqrt() * 0.9
    angle = index * 2.399963229728653
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)

  # ------------------------------------------------------------------ helpers
  def _as_feature_map(self, x: torch.Tensor) -> torch.Tensor:
    """Validate the scan and return a ``(B, 1, H, W)`` float tensor."""
    if x.ndim == 2:
      batch, num_cells = x.shape
      if num_cells <= 0:
        raise ValueError(f"scan must have at least one cell, got {tuple(x.shape)}")
      if self.image_hw is not None:
        img_h, img_w = self.image_hw
        if img_h * img_w != num_cells:
          raise ValueError(
            f"scan has {num_cells} cells but image_hw={self.image_hw} needs "
            f"{img_h * img_w}"
          )
      else:
        side = math.isqrt(num_cells)
        if side * side != num_cells:
          raise ValueError(
            f"flat scan length {num_cells} is not a perfect square; pass "
            "image_hw=(H, W) to reshape non-square height maps"
          )
        img_h = img_w = side
      return x.reshape(batch, 1, img_h, img_w)
    if x.ndim == 3:
      return self._as_feature_map(x.unsqueeze(1))
    if x.ndim == 4:
      if x.shape[1] != 1:
        raise ValueError(
          f"scan must have exactly one channel, got shape {tuple(x.shape)}"
        )
      if x.shape[2] <= 0 or x.shape[3] <= 0:
        raise ValueError(f"scan spatial dims must be positive, got {tuple(x.shape)}")
      if self.image_hw is not None:
        if (x.shape[2], x.shape[3]) != self.image_hw:
          raise ValueError(
            f"scan has spatial shape {(x.shape[2], x.shape[3])} but "
            f"image_hw={self.image_hw}"
          )
      return x
    raise ValueError(
      "scan must be (B, N), (B, H, W) or (B, 1, H, W), got shape "
      f"{tuple(x.shape)}"
    )

  def _deformable_grid(
    self,
    x: torch.Tensor,
    state: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grid, feature_map)``; ``grid`` is ``(B, num_samples, 2)``."""
    feature_map = self.stem(self._as_feature_map(x))
    batch = feature_map.shape[0]
    context = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
    if self.state_dim is not None:
      if state is None:
        raise ValueError(
          f"this encoder was built with state_dim={self.state_dim}; forward "
          "requires a state tensor"
        )
      if state.ndim != 2 or state.shape[1] != self.state_dim:
        raise ValueError(
          f"state must have shape (B, {self.state_dim}), got "
          f"{tuple(state.shape)}"
        )
      if state.shape[0] != batch:
        raise ValueError(
          f"state batch ({state.shape[0]}) does not match scan batch ({batch})"
        )
      state = state.to(device=context.device, dtype=context.dtype)
      context = torch.cat([context, state], dim=-1)
    elif state is not None:
      raise ValueError("state was provided but this encoder has no state_dim")
    offsets = torch.tanh(self.offset_head(context))
    offsets = offsets.reshape(batch, self.num_samples, 2) * self.offset_scale
    grid = (self.base_grid.unsqueeze(0) + offsets).clamp(-1.0, 1.0)
    return grid, feature_map

  # -------------------------------------------------------------------- public
  def sample_locations(
    self,
    x: torch.Tensor,
    state: torch.Tensor | None = None,
    pixel: bool = False,
  ) -> torch.Tensor:
    """Return the learned sampling locations ``(B, num_samples, 2)``.

    Coordinates are normalised ``(x, y)`` in ``[-1, 1]`` by default (the
    ``grid_sample`` convention); with ``pixel=True`` they are converted to
    pixel ``(x, y)`` coordinates of the input map.  Useful for the DELTA-style
    interpretability plots (where the encoder "looks").
    """
    grid, _ = self._deformable_grid(x, state)
    if not pixel:
      return grid
    img_h, img_w = self._as_feature_map(x).shape[2:]
    scale = grid.new_tensor([(img_w - 1) * 0.5, (img_h - 1) * 0.5])
    return (grid + 1.0) * scale

  def forward(
    self,
    x: torch.Tensor,
    state: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Encode a height/terrain scan into a ``(B, feature_dim)`` feature vector."""
    grid, feature_map = self._deformable_grid(x, state)
    sampled = F.grid_sample(
      feature_map,
      grid.unsqueeze(2),  # (B, num_samples, 1, 2)
      mode="bilinear",
      padding_mode="border",
      align_corners=True,
    )  # (B, latent_dim, num_samples, 1)
    tokens = sampled.squeeze(-1).transpose(1, 2)  # (B, num_samples, latent_dim)
    tokens = self.norm(tokens)
    query = self.pool_query.expand(feature_map.shape[0], -1, -1)
    pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
    return self.head(pooled.squeeze(1))
