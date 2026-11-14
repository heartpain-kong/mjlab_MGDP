"""MGDP perception module for mjlab (U-Net contrastive depth encoder + GRU).

Faithful port of the released MGDP perception model
(``arclab-hku/MGDP``, Advanced Science 2026) so the published ``wm_best.pt``
weights can be loaded directly when the depth convention matches.

What is kept from MGDP:
  * dual-modality encoders (depth image + height map) with U-Net skip paths
  * depth reconstruction against the *clean* frame (explicit denoising)
  * height reconstruction (auxiliary supervision)
  * InfoNCE alignment between the depth token and the height token
  * GRU that turns the per-frame token into a temporal feature for the policy

What differs from the original repo (mjlab integration):
  * depth comes from ``RayCastSensor`` pinhole distances (metres), not an
    Isaac Gym / Warp render
  * the module owns its own two-frame buffer and GRU state, so the mjlab
    observation manager only sees a ``[B, 64]`` feature
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    """(conv - norm - SiLU) x2, matching the released MGDP blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.fuse(torch.cat([x, skip], dim=1))


class ImageEncoder(nn.Module):
    """Depth encoder: (B, 2, H, W) -> (B, out_channels)."""

    def __init__(self, input_channels=2, hidden=(32, 64, 96), output_channels=64, pool=4):
        super().__init__()
        self.pool = pool
        self.conv1 = DoubleConv(input_channels, hidden[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv(hidden[0], hidden[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv(hidden[1], hidden[2])
        self.bottleneck = DoubleConv(hidden[2], hidden[2])
        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        proj = hidden[2] * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden[2] * pool * pool, proj),
            nn.LayerNorm(proj),
            nn.SiLU(inplace=True),
            nn.Linear(proj, output_channels),
        )
        self.skip_connections: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.skip_connections.clear()
        b = x.size(0)
        x1 = self.conv1(x)
        self.skip_connections.append(x1)
        x2 = self.conv2(self.pool1(x1))
        self.skip_connections.append(x2)
        x3 = self.bottleneck(self.conv3(self.pool2(x2)))
        self.skip_connections.append(x3)
        pooled = self.global_pool(x3).reshape(b, -1)
        return self.fc(pooled)


class ImageDecoder(nn.Module):
    """Depth decoder: (B, token) -> (B, 2, h, w) in [0, 1]."""

    def __init__(self, input_dims=64, hidden=(96, 64, 32), pool=4, out_channels=2):
        super().__init__()
        self.pool = pool
        self.hidden = hidden
        proj = hidden[0] * pool * pool
        self.fc = nn.Sequential(
            nn.Linear(input_dims, proj),
            nn.LayerNorm(proj),
            nn.SiLU(inplace=True),
        )
        self.input_refine = DoubleConv(hidden[0], hidden[0])
        self.up1 = UpFuseBlock(hidden[0], hidden[0], hidden[0])
        self.up2 = UpFuseBlock(hidden[0], hidden[1], hidden[1])
        self.up3 = UpFuseBlock(hidden[1], hidden[2], hidden[2])
        self.refine = DoubleConv(hidden[2], hidden[2])
        self.out_conv = nn.Conv2d(hidden[2], out_channels, 1)

    def forward(self, x, skips, target_size):
        b = x.size(0)
        x = self.fc(x).reshape(b, self.hidden[0], self.pool, self.pool)
        x = self.input_refine(x)
        x = self.up1(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[0])
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        return torch.sigmoid(self.out_conv(self.refine(x)))


class MapEncoder(nn.Module):
    """Height-map encoder: (B, 1, 17, 11) -> (B, 32)."""

    def __init__(self, input_channels=1, hidden=(16, 32, 64), output_channels=32, pool=2):
        super().__init__()
        self.pool = pool
        self.conv1 = DoubleConv(input_channels, hidden[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv(hidden[0], hidden[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv(hidden[1], hidden[2])
        self.bottleneck = DoubleConv(hidden[2], hidden[2])
        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        proj = hidden[2] * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden[2] * pool * pool, proj),
            nn.LayerNorm(proj),
            nn.SiLU(inplace=True),
            nn.Linear(proj, output_channels),
        )
        self.skip_connections: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.skip_connections.clear()
        b = x.size(0)
        x1 = self.conv1(x)
        self.skip_connections.append(x1)
        x2 = self.conv2(self.pool1(x1))
        self.skip_connections.append(x2)
        x3 = self.bottleneck(self.conv3(self.pool2(x2)))
        self.skip_connections.append(x3)
        return self.fc(self.global_pool(x3).reshape(b, -1))


class MapDecoder(nn.Module):
    """Height decoder: (B, token) -> (B, 1, 17, 11) in [-1, 1]."""

    def __init__(self, input_dims=32, hidden=(64, 32, 16), pool=2):
        super().__init__()
        self.pool = pool
        self.hidden = hidden
        proj = hidden[0] * pool * pool
        self.fc = nn.Sequential(
            nn.Linear(input_dims, proj),
            nn.LayerNorm(proj),
            nn.SiLU(inplace=True),
        )
        self.input_refine = DoubleConv(hidden[0], hidden[0])
        self.up1 = UpFuseBlock(hidden[0], hidden[0], hidden[0])
        self.up2 = UpFuseBlock(hidden[0], hidden[1], hidden[1])
        self.up3 = UpFuseBlock(hidden[1], hidden[2], hidden[2])
        self.refine = DoubleConv(hidden[2], hidden[2])
        self.out_conv = nn.Conv2d(hidden[2], 1, 1)

    def forward(self, x, skips, target_size):
        b = x.size(0)
        x = self.fc(x).reshape(b, self.hidden[0], self.pool, self.pool)
        x = self.input_refine(x)
        x = self.up1(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[0])
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        return torch.tanh(self.out_conv(self.refine(x)))


@dataclass
class MGPDPerceptionCfg:
    """Configuration shared by pretraining and the rollout observation term."""

    depth_hw: tuple[int, int] = (16, 16)
    height_hw: tuple[int, int] = (17, 11)
    depth_max: float = 5.0
    depth_min: float = 0.1
    noise_sigma: float = 0.03
    dropout: float = 0.1
    temperature: float = 0.1
    depth_mse_weight: float = 1.2
    edge_weight: float = 0.3
    height_mse_weight: float = 1.0
    contrastive_weight: float = 0.3
    temporal_weight: float = 1.0
    frames: int = 2
    feature_dim: int = 64
    gru_hidden: int = 64

    @property
    def depth_pixels(self) -> int:
        return self.depth_hw[0] * self.depth_hw[1]

    @property
    def height_pixels(self) -> int:
        return self.height_hw[0] * self.height_hw[1]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["depth_hw"] = tuple(self.depth_hw)
        data["height_hw"] = tuple(self.height_hw)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MGPDPerceptionCfg":
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in allowed}
        for key in ("depth_hw", "height_hw"):
            if key in clean:
                clean[key] = tuple(clean[key])
        return cls(**clean)


def _sobel_edges(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients for (B, C, H, W) tensors (MGDP edge loss)."""
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
    return gx, gy


class MGPDPerception(nn.Module):
    """Encoder/decoder/GRU bundle with MGDP's training losses and inference API."""

    def __init__(self, cfg: MGPDPerceptionCfg):
        super().__init__()
        self.cfg = cfg
        self.sensing = None
        self.camera = None
        self.depth_encoder = ImageEncoder(
            input_channels=cfg.frames, hidden=(32, 64, 96),
            output_channels=cfg.feature_dim, pool=4,
        )
        self.depth_decoder = ImageDecoder(
            input_dims=cfg.feature_dim, hidden=(96, 64, 32), pool=4,
            out_channels=cfg.frames,
        )
        self.height_encoder = MapEncoder(
            input_channels=1, hidden=(16, 32, 64), output_channels=32, pool=2,
        )
        self.height_decoder = MapDecoder(input_dims=32, hidden=(64, 32, 16), pool=2)
        self.memory = nn.GRU(cfg.feature_dim, cfg.gru_hidden, num_layers=1, batch_first=True)
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return self.depth_encoder(depth)

    def feature(self, depth: torch.Tensor, hidden: torch.Tensor | None):
        """depth: (B, frames, h, w) normalized; returns ((B, 64), new_hidden)."""
        token = self.depth_encoder(depth)
        # Reconstruction needs skip maps during pretraining; frozen feature
        # inference does not. Do not retain a full-env U-Net activation batch.
        self.depth_encoder.skip_connections.clear()
        out, hidden = self.memory(token.unsqueeze(1), hidden)
        return out.squeeze(1), hidden

    def feature_batched(self, depth, hidden, batch_size=256,
                        execution="eager", precision="fp32"):
        """Bound frozen inference activation memory without mixing GRU states.

        GroupNorm/LayerNorm act within each sample, so environment batches can
        be split independently. Zero selects the original whole-batch path.
        """
        if batch_size < 0:
            raise ValueError("inference batch size must be nonnegative")
        from .mgdp_inference import FrozenDepthRuntime
        runtime = getattr(self, "_frozen_depth_runtime", None)
        settings = (execution, precision)
        if runtime is None or (runtime.execution, runtime.precision) != settings:
            runtime = FrozenDepthRuntime(self, execution, precision)
            self._frozen_depth_runtime = runtime
        chunk = batch_size or len(depth)
        tokens = [runtime.encode(depth[start:start + chunk])
                  for start in range(0, len(depth), chunk)]
        token = tokens[0] if len(tokens) == 1 else torch.cat(tokens, dim=0)
        # Environments are independent in the batch dimension: run GRU once.
        out, hidden = self.memory(token.unsqueeze(1), hidden)
        return out.squeeze(1), hidden

    @torch.no_grad()
    def act_feature(self, depth: torch.Tensor, hidden: torch.Tensor | None):
        self.eval()
        return self.feature(depth, hidden)

    # ------------------------------------------------------------------ #
    # Training losses
    # ------------------------------------------------------------------ #
    def _corrupt(self, depth: torch.Tensor) -> torch.Tensor:
        noisy = depth
        if self.training and self.cfg.noise_sigma > 0:
            noisy = noisy + torch.randn_like(noisy) * self.cfg.noise_sigma
        if self.training and self.cfg.dropout > 0:
            mask = torch.rand_like(noisy) > self.cfg.dropout
            noisy = noisy * mask
        return noisy.clamp(0.0, 1.0)

    def losses(self, depth_clean: torch.Tensor, height: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the MGDP perception loss.

        depth_clean: (B,T,frames,h,w) or (B,frames,h,w), in [0,1]
        height: (B,T,1,H,W) or (B,1,H,W), in [-1,1].
        T must contain consecutive steps from a single episode.
        """
        # Sequences are [B,T,frames,H,W]; legacy single-pair callers use T=1.
        if depth_clean.ndim == 5:
            batch, steps = depth_clean.shape[:2]
            depth_clean = depth_clean.flatten(0, 1)
            height = height.flatten(0, 1)
        else:
            batch, steps = depth_clean.shape[0], 1
        depth_noisy = self._corrupt(depth_clean)
        depth_token = self.depth_encoder(depth_noisy)
        depth_recon = self.depth_decoder(
            depth_token, self.depth_encoder.skip_connections, depth_clean.shape[-2:]
        )
        depth_mse = F.mse_loss(depth_recon, depth_clean)
        gx_p, gy_p = _sobel_edges(depth_recon)
        gx_t, gy_t = _sobel_edges(depth_clean)
        edge = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

        height_token = self.height_encoder(height)
        height_recon = self.height_decoder(
            height_token, self.height_encoder.skip_connections, height.shape[-2:]
        )
        height_mse = F.mse_loss(height_recon, height)

        shared = min(depth_token.size(1), height_token.size(1))
        d_norm = F.normalize(depth_token[:, :shared], p=2, dim=1, eps=1e-8)
        h_norm = F.normalize(height_token[:, :shared], p=2, dim=1, eps=1e-8)
        logits = d_norm @ h_norm.t() / self.cfg.temperature
        labels = torch.arange(depth_token.size(0), device=depth_token.device)
        contrastive = F.cross_entropy(logits, labels)

        # Train the actual recurrent feature used during policy rollout. Predict
        # the clean depth token at each step, with gradients through the whole
        # sequence (including recurrent weights); detach the target branch.
        with torch.no_grad():
            clean_token = self.depth_encoder(depth_clean)
        temporal, _ = self.memory(depth_token.reshape(batch, steps, -1))
        if temporal.shape[-1] != clean_token.shape[-1]:
            raise ValueError("GRU hidden width must match depth feature width")
        temporal_loss = F.mse_loss(temporal.flatten(0, 1), clean_token)

        total = (
            self.cfg.depth_mse_weight * depth_mse
            + self.cfg.edge_weight * edge
            + self.cfg.height_mse_weight * height_mse
            + self.cfg.contrastive_weight * contrastive
            + self.cfg.temporal_weight * temporal_loss
        )
        return total, {
            "total": float(total.detach()),
            "depth_mse": float(depth_mse.detach()),
            "edge": float(edge.detach()),
            "height_mse": float(height_mse.detach()),
            "contrastive": float(contrastive.detach()),
            "temporal": float(temporal_loss.detach()),
        }

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {
            "format": "mgdp_mjlab_v1",
            "cfg": self.cfg.to_dict(),
            "depth_encoder": self.depth_encoder.state_dict(),
            "depth_decoder": self.depth_decoder.state_dict(),
            "height_encoder": self.height_encoder.state_dict(),
            "height_decoder": self.height_decoder.state_dict(),
            "memory": self.memory.state_dict(),
        }
        if extra:
            payload.update(extra)
        if "sensing" not in payload and self.sensing is not None:
            payload["sensing"] = self.sensing
        if 'camera' not in payload and self.camera is not None:
            payload['camera'] = self.camera
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location="cpu") -> "MGPDPerception":
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "mgdp_mjlab_v1":
            raise ValueError(
                f"{path} is not an mjlab MGDP checkpoint (format={payload.get('format')!r})"
            )
        model = cls(MGPDPerceptionCfg.from_dict(payload["cfg"]))
        model.sensing = payload.get("sensing")
        model.camera = payload.get('camera')
        model.depth_encoder.load_state_dict(payload["depth_encoder"])
        model.depth_decoder.load_state_dict(payload["depth_decoder"])
        model.height_encoder.load_state_dict(payload["height_encoder"])
        model.height_decoder.load_state_dict(payload["height_decoder"])
        model.memory.load_state_dict(payload["memory"])
        model.eval()
        return model

    def load_released_weights(self, wm_path: str | Path) -> list[str]:
        """Optional init from the original MGDP ``wm_best.pt`` (depth convention!).

        Returns the list of modules that were loaded. The original checkpoint
        uses Isaac Gym depth in metres normalised by 5 m, which matches the
        RayCastSensor convention here.
        """
        wm = torch.load(str(wm_path), map_location="cpu", weights_only=False)
        mapping = {
            "image_encoder": self.depth_encoder,
            "image_decoder": self.depth_decoder,
            "map_encoder": self.height_encoder,
            "map_decoder": self.height_decoder,
            "memory_encoder": self.memory,
        }
        loaded = []
        for key, module in mapping.items():
            if key not in wm:
                continue
            state = wm[key]
            if key == "memory_encoder":
                state = {k.removeprefix("rnn."): v for k, v in state.items()}
            module.load_state_dict(state)
            loaded.append(key)
        return loaded
