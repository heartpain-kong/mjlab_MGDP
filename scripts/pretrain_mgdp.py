"""MGDP stage 1 in mjlab: pretrain the contrastive depth perception model.

The script plays random actions in the LgDog1 rough-terrain environment,
collects (depth, height) pairs from the raycast sensors, and trains the
MGDP perception module with its original losses:

    depth reconstruction (clean target) + Sobel edge
  + height reconstruction
  + InfoNCE between depth and height tokens

The resulting checkpoint is what ``LgDog1-Rough-MGDP`` loads as a frozen
observation encoder.

Examples
--------
    # GPU server, default 256 envs / 3000 steps / 3000 iterations
    python scripts/pretrain_mgdp.py

    # smaller smoke test
    python scripts/pretrain_mgdp.py --num-envs 64 --steps 300 --iterations 200

    # initialise from the released MGDP checkpoint (optional)
    python scripts/pretrain_mgdp.py --init-from /path/to/wm_best.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import height_scan

from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import (
    DEPTH_HW,
    DEPTH_MAX,
    DEFAULT_CHECKPOINT,
    lgdog1_mgdp_env_cfg,
)
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--sample-envs", type=int, default=32,
                   help="how many envs are recorded per step")
    p.add_argument("--steps", type=int, default=3000, help="policy steps to collect")
    p.add_argument("--action-hold", type=int, default=25,
                   help="hold a random action for N steps (smooth motion, fewer resets)")
    p.add_argument("--terrain-rows", type=int, default=12)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=DEFAULT_CHECKPOINT)
    p.add_argument("--init-from", default="", help="released MGDP wm_best.pt (optional)")
    return p.parse_args()


def build_env(args) -> ManagerBasedRlEnv:
    cfg = lgdog1_mgdp_env_cfg(with_perception=False)
    cfg.scene.num_envs = args.num_envs
    terrain = cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        tg = terrain.terrain_generator
        tg.curriculum = False
        tg.num_rows = args.terrain_rows
        tg.difficulty_range = (0.4, 1.0)
    return ManagerBasedRlEnv(cfg=cfg, device=args.device)


@torch.no_grad()
def collect(args, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll random actions and record consecutive (depth pair, height) samples."""
    h, w = DEPTH_HW
    k = min(args.sample_envs, args.num_envs)
    stacks: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []

    env.reset()
    actions = torch.zeros((env.num_envs, 12), device=env.device)
    prev_depth = None
    prev_valid = torch.zeros(k, dtype=torch.bool)
    for step in range(args.steps):
        if step % max(1, args.action_hold) == 0:
            actions = (torch.rand((env.num_envs, 12), device=env.device) - 0.5) * 0.3
        env.step(actions)

        depth = env.scene["depth_rays"].data.distances[:k].reshape(k, 1, h, w)
        miss = depth < 0
        depth = torch.where(miss, torch.full_like(depth, DEPTH_MAX), depth)
        depth = (depth.clamp(0.0, DEPTH_MAX) / DEPTH_MAX).float().cpu()

        height = height_scan(env, "terrain_scan")[:k]
        height = (height / 5.0).clamp(-1.0, 1.0).float().cpu()

        eplen = env.episode_length_buf[:k].cpu()
        valid = eplen > 0
        if prev_depth is not None:
            take = valid & prev_valid
            if take.any():
                idx = take.nonzero(as_tuple=False).squeeze(-1)
                stacks.append(torch.cat([prev_depth[idx], depth[idx]], dim=1))
                heights.append(height[idx])
        prev_depth = depth
        prev_valid = valid

        if step % 200 == 0:
            print(f"[collect] {step}/{args.steps}  pairs={sum(s.shape[0] for s in stacks)}")

    if not stacks:
        raise RuntimeError("no valid samples collected; increase --steps")
    return torch.cat(stacks), torch.cat(heights)


def evaluate(model, stacks, heights_flat, cfg) -> dict:
    model.eval()
    with torch.no_grad():
        _, metrics = model.losses(stacks, heights_flat.view(-1, 1, *cfg.height_hw))
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[env] device={args.device}  num_envs={args.num_envs}")
    env = build_env(args)

    t0 = time.time()
    stacks, heights = collect(args, env)
    print(f"[collect] {args.steps} steps in {time.time() - t0:.1f}s  "
          f"stacks={tuple(stacks.shape)} height={tuple(heights.shape)}")

    print(f"[data] pairs={stacks.shape[0]}  stack={tuple(stacks.shape)}  "
          f"height={tuple(heights.shape)}")

    cfg = MGPDPerceptionCfg(depth_hw=DEPTH_HW, height_hw=(17, 11), depth_max=DEPTH_MAX)
    model = MGPDPerception(cfg).to(args.device)
    if args.init_from:
        loaded = model.load_released_weights(args.init_from)
        print(f"[init] loaded from {args.init_from}: {loaded}")
    model.train()

    perm = torch.randperm(stacks.shape[0])
    n_val = max(1, int(stacks.shape[0] * args.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    val_stacks = stacks[val_idx].to(args.device)
    val_heights = heights[val_idx].to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = args.batch_size
    for it in range(1, args.iterations + 1):
        idx = train_idx[torch.randint(0, train_idx.numel(), (batch,))]
        s = stacks[idx].to(args.device)
        hh = heights[idx].to(args.device).view(-1, 1, *cfg.height_hw)

        loss, metrics = model.losses(s, hh)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % args.log_every == 0 or it == 1:
            val = evaluate(model, val_stacks, val_heights, cfg)
            print(
                f"[train] {it:5d}  total={metrics['total']:.4f}  "
                f"depth={metrics['depth_mse']:.4f}  edge={metrics['edge']:.4f}  "
                f"height={metrics['height_mse']:.4f}  nce={metrics['contrastive']:.4f}  "
                f"| val total={val['total']:.4f} depth={val['depth_mse']:.4f}"
            )

    out = Path(args.out)
    model.save(out, extra={
        "iterations": args.iterations,
        "pairs": int(stacks.shape[0]),
        "val": evaluate(model, val_stacks, val_heights, cfg),
        "source": "mjlab-random-rollout",
    })
    print(f"[save] {out}")

    stats = evaluate(model, val_stacks, val_heights, cfg)
    print(f"[val ] {stats}")
    env.close()


if __name__ == "__main__":
    main()


