"""Observation term: MGDP depth feature for mjlab velocity tasks.

The term reads a pinhole ``RayCastSensor`` (a depth camera made of rays),
normalises distances to [0, 1], keeps MGDP's two-frame stack, and runs the
frozen perception module (U-Net + GRU) to produce a ``[B, 64]`` feature.

The module is created lazily on first call and cached on the environment, so
adding the term only requires an env config change - no runner surgery.
"""

from __future__ import annotations

import warnings

import torch

from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


_STATE_ATTR = "_mgdp_perception_state"


def _as_cfg(cfg) -> MGPDPerceptionCfg:
    if cfg is None:
        return MGPDPerceptionCfg()
    if isinstance(cfg, MGPDPerceptionCfg):
        return cfg
    if isinstance(cfg, dict):
        return MGPDPerceptionCfg.from_dict(cfg)
    raise TypeError(f"unsupported MGDP cfg type: {type(cfg)!r}")


def get_perception(env, cfg: MGPDPerceptionCfg, checkpoint: str = "") -> MGPDPerception:
    """Return the env-owned perception module, creating/loading it once."""
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        if checkpoint:
            module = MGPDPerception.from_checkpoint(checkpoint, map_location=env.device)
        else:
            raise RuntimeError(
                "MGDP perception checkpoint is not set. Run "
                "`python scripts/pretrain_mgdp.py` first, or pass "
                "`--env.observations.perception.terms.depth_feature.params.checkpoint <path>`."
            )
        module = module.to(env.device).eval()
        from src.mgdp_checkpoint import sensing_contract
        filters = sensing_contract(getattr(env, "cfg", None))
        if module.sensing is not None and filters and module.sensing != filters:
            raise ValueError("MGDP感知模型的深度/高度扫描过滤设置与当前环境不同；"
                             "请使用配套感知权重或在当前设置下重新预训练")
        if module.sensing is None and filters and all(v == [0] for v in filters.values()):
            warnings.warn("旧MGDP感知权重没有记录传感器约定；当前新训练只扫描地形。"
                          "建议重新预训练感知模型后创建新策略训练。", stacklevel=2)
        state = {"module": module, "hidden": None, "prev": None}
        setattr(env, _STATE_ATTR, state)
    return state["module"]


def _depth_image(env, sensor_name: str, cfg: MGPDPerceptionCfg) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    distances = sensor.data.distances  # [B, N], metres, -1 for miss
    num = distances.shape[1]
    h, w = cfg.depth_hw
    if num != h * w:
        raise ValueError(
            f"depth sensor '{sensor_name}' has {num} rays but MGDP expects "
            f"{h}x{w}={h * w}; update PinholeCameraPatternCfg or cfg.depth_hw."
        )
    depth = distances.reshape(distances.shape[0], 1, h, w)
    miss = depth < 0
    depth = torch.where(miss, torch.full_like(depth, cfg.depth_max), depth)
    depth = depth.clamp(cfg.depth_min, cfg.depth_max) / cfg.depth_max
    return depth


def mgdp_depth_feature(
    env,
    sensor_name: str = "depth_rays",
    cfg=None,
    checkpoint: str = "",
    inference_batch_size: int = 256,
    inference_execution: str = "eager",
    inference_precision: str = "fp32",
) -> torch.Tensor:
    """Return the temporal MGDP feature ``[B, 64]`` for the actor group."""
    cfg = _as_cfg(cfg)
    module = get_perception(env, cfg, checkpoint)
    state = getattr(env, _STATE_ATTR)

    depth = _depth_image(env, sensor_name, cfg)
    batch = depth.shape[0]
    reset = (env.episode_length_buf == 0).to(depth.device)

    prev = state["prev"]
    if prev is None or prev.shape[0] != batch:
        prev = depth
    else:
        # Do not stack a frame from the previous episode across a reset.
        prev = torch.where(reset.view(-1, 1, 1, 1), depth, prev)
    stack = torch.cat([prev, depth], dim=1)

    hidden = state["hidden"]
    if hidden is not None:
        hidden = hidden * (~reset).view(1, -1, 1)

    with torch.no_grad():
        if inference_execution == "eager" and inference_precision == "fp32":
            feature, hidden = module.feature_batched(stack, hidden, inference_batch_size)
        else:
            feature, hidden = module.feature_batched(
                stack, hidden, inference_batch_size, inference_execution, inference_precision)

    state["prev"] = depth.detach()
    state["hidden"] = hidden.detach()
    return feature.to(dtype=torch.float32)


def reset_perception_state(env) -> None:
    """Clear cached GRU state (call on episode reset / play)."""
    state = getattr(env, _STATE_ATTR, None)
    if state is not None:
        state["hidden"] = None
        state["prev"] = None
