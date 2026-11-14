"""MGDP stage 1 in mjlab: pretrain the contrastive depth perception model.

The script plays random actions in the selected LgDog1 terrain environment,
collects (depth, height) pairs from the raycast sensors, and trains the
MGDP perception module with its original losses:

    depth reconstruction (clean target) + Sobel edge
  + height reconstruction
  + InfoNCE between depth and height tokens
  + recurrent feature prediction on episode-contiguous sequences

The selected Rough/Race MGDP task loads the resulting checkpoint as its
frozen observation encoder.

Examples
--------
    # GPU server, default 256 envs / 3000 steps / 3000 iterations
    python scripts/pretrain_mgdp.py

    # smaller smoke test
    python scripts/pretrain_mgdp.py --num-envs 64 --steps 300 --iterations 200

    # compact race collection map; retain all ten obstacle types
    python -m scripts.pretrain_mgdp --task LgDog1-Race-MGDP

    # initialise from the released MGDP checkpoint (optional)
    python scripts/pretrain_mgdp.py --init-from /path/to/wm_best.pt
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import os
import random
import time
from pathlib import Path

if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    print("[init] importing PyTorch and simulation libraries...", flush=True)

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import height_scan
from src.config.mgdp_pretrain import pretrain_resources
from src.sim_memory import compact_model_allocations

from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import (
    DEPTH_HW,
    DEPTH_MAX,
    DEFAULT_CHECKPOINT,
    DEFAULT_RACE_CHECKPOINT,
    lgdog1_mgdp_env_cfg,
)
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=("LgDog1-Rough-MGDP","LgDog1-Race-MGDP"),default="LgDog1-Rough-MGDP",
                   help="感知采集地图；比赛任务应采集2027障碍，包含限高和轮胎")
    p.add_argument("--num-envs", type=int, default=None,
                   help="parallel envs: race defaults to 64, rough to 256")
    p.add_argument("--sequence-length", type=int, default=8, help="consecutive steps for GRU training, minimum 2")
    p.add_argument("--sample-envs", type=int, default=32,
                   help="how many envs are recorded per step")
    p.add_argument("--steps", type=int, default=3000, help="policy steps to collect")
    p.add_argument("--action-hold", type=int, default=25,
                   help="hold a random action for N steps (smooth motion, fewer resets)")
    p.add_argument("--terrain-rows", type=int, default=None)
    p.add_argument("--terrain-cols", type=int, default=None)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=None,
                   help="training/validation sequence batch: race defaults to 64, rough to 256")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--optimize-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collision-mode", choices=("auto", "sap"), default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--init-from", default="", help="released MGDP wm_best.pt (optional)")
    args = p.parse_args()
    resources = pretrain_resources(args.task)
    for name in ("num_envs", "terrain_rows", "terrain_cols", "batch_size"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(resources, name))
    args.out = args.out or (DEFAULT_RACE_CHECKPOINT if args.task=="LgDog1-Race-MGDP" else DEFAULT_CHECKPOINT)
    if args.sequence_length < 2 or args.steps < args.sequence_length + 2:
        p.error("sequence-length must be >=2 and steps must exceed sequence-length+1")
    if not 0 < args.val_fraction < 1:
        p.error("val-fraction must be in (0,1)")
    if min(args.iterations, args.num_envs, args.sample_envs, args.batch_size,
           args.terrain_rows, args.log_every) < 1:
        p.error("iterations, num-envs, sample-envs, batch-size, terrain-rows and log-every must be positive")
    if args.terrain_cols is not None and args.terrain_cols < 1:
        p.error("terrain-cols must be positive")
    if args.task == "LgDog1-Race-MGDP" and args.terrain_cols < resources.terrain_cols:
        p.error("race terrain-cols must be >=10 to retain every obstacle type")
    return args


def build_env(args) -> ManagerBasedRlEnv:
    cfg = lgdog1_mgdp_env_cfg(with_perception=False)
    race = getattr(args,"task","LgDog1-Rough-MGDP")=="LgDog1-Race-MGDP"
    if race:
        from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import lgdog1_passage_terrain_cfg
        from src.tasks.velocity.config.lgdog1.domain_rand_cfg import configure_race_himloco_randomization
        configure_race_himloco_randomization(cfg)
        cfg.scene.terrain.terrain_generator = lgdog1_passage_terrain_cfg()
        # Face the obstacle from its entry, without requiring a trained walking
        # controller to reach it during random-action perception collection.
        cfg.events["reset_base"].params["pose_range"] = {
            "x":(0.,.2),"y":(-.1,.1),"z":(.02,.06),"yaw":(-.2,.2)}
        cfg.curriculum = {}
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    terrain = cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        tg = terrain.terrain_generator
        # In race collection, stratify columns by obstacle and rows by difficulty.
        # No RL curriculum is active: all difficulty rows are sampled directly.
        tg.curriculum = race
        tg.num_rows = args.terrain_rows
        columns = getattr(args, "terrain_cols", None) or pretrain_resources(
            getattr(args, "task", "LgDog1-Rough-MGDP")).terrain_cols
        if columns is not None:
            tg.num_cols = columns
        if race:
            terrain.max_init_terrain_level = None
        tg.difficulty_range = (0.0, 1.0) if race else (0.4, 1.0)
        print(f"[init] terrain={tg.num_rows}x{tg.num_cols}, envs={args.num_envs}; "
              "building scene, compiling MuJoCo and initializing physics...", flush=True)
    with compact_model_allocations(getattr(args, "optimize_vram", True), getattr(args, "collision_mode", "auto")):
        return ManagerBasedRlEnv(cfg=cfg, device=args.device)


@torch.no_grad()
def collect(args, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll random actions and record consecutive (depth pair, height) samples."""
    h, w = DEPTH_HW
    depth_min = MGPDPerceptionCfg().depth_min
    k = min(args.sample_envs, args.num_envs)
    # Terrain types are grouped by environment index. A prefix would record
    # only the first few obstacle columns, especially when num_envs is large.
    sample_ids = torch.linspace(0, env.num_envs - 1, k, device=env.device).round().long()
    stacks: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []

    print(f"[collect] resetting environment; recording {k} evenly distributed envs...", flush=True)
    env.reset()
    actions = torch.zeros((env.num_envs, 12), device=env.device)
    prev_depth = None
    previous_episode = None
    pending_depth, pending_height = [], []
    pending_continuity = []
    for step in range(args.steps):
        if step % max(1, args.action_hold) == 0:
            actions = (torch.rand((env.num_envs, 12), device=env.device) - 0.5) * 0.3
        env.step(actions)

        depth = env.scene["depth_rays"].data.distances[sample_ids].reshape(k, 1, h, w)
        miss = depth < 0
        depth = torch.where(miss, torch.full_like(depth, DEPTH_MAX), depth)
        depth = (depth.clamp(depth_min, DEPTH_MAX) / DEPTH_MAX).float().cpu()

        height = height_scan(env, "terrain_scan")[sample_ids]
        height = (height / 5.0).clamp(-1.0, 1.0).float().cpu()

        eplen = env.episode_length_buf[sample_ids].cpu()
        if prev_depth is not None:
            continuous = (eplen > 0) & (eplen == previous_episode + 1)
            pending_depth.append(torch.cat([prev_depth, depth], dim=1))
            pending_height.append(height)
            pending_continuity.append(continuous)
            if len(pending_depth) == args.sequence_length:
                take = torch.stack(pending_continuity).all(dim=0)
                if take.any():
                    stacks.append(torch.stack(pending_depth, dim=1)[take])
                    heights.append(torch.stack(pending_height, dim=1)[take])
                pending_depth, pending_height, pending_continuity = [], [], []
        prev_depth = depth
        previous_episode = eplen.clone()

        if step % 200 == 0:
            print(f"[collect] {step}/{args.steps}  sequences={sum(s.shape[0] for s in stacks)}", flush=True)

    if not stacks:
        raise RuntimeError("no valid samples collected; increase --steps")
    return torch.cat(stacks), torch.cat(heights)


def evaluate(model, stacks, heights_flat, cfg, batch_size=64) -> dict:
    """Bound GPU memory, including the quadratic InfoNCE logits matrix.

    Metrics are sequence-weighted means of minibatch losses; contrastive loss
    uses negatives inside each minibatch, not across the whole validation set.
    """
    if len(stacks) == 0 or batch_size < 1:
        raise ValueError("Validation requires samples and a positive batch size")
    was_training = model.training
    model.eval()
    totals = {}
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for start in range(0, len(stacks), batch_size):
                s = stacks[start:start + batch_size].to(device)
                h = heights_flat[start:start + batch_size].to(device)
                _, metrics = model.losses(s, h.reshape(*h.shape[:-1], 1, *cfg.height_hw))
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.) + float(value) * len(s)
        return {name: value / len(stacks) for name, value in totals.items()}
    finally:
        model.train(was_training)


def collect_and_release(args):
    """Keep only CPU samples/metadata after the simulation phase."""
    from src.tasks.velocity.config.lgdog1.domain_rand_cfg import himloco_randomization_metadata
    t0 = time.time()
    env = build_env(args)
    try:
        print(f"[env] ready in {time.time() - t0:.1f}s", flush=True)
        generator = env.cfg.scene.terrain.terrain_generator
        metadata = {
            "terrain_layout_version": getattr(next(iter(generator.sub_terrains.values())), "layout_version", None),
            "terrain_grid": (generator.num_rows, generator.num_cols),
            "domain_randomization": himloco_randomization_metadata(env.cfg),
        }
        from src.mgdp_checkpoint import sensing_contract
        metadata["sensing"] = sensing_contract(env.cfg)
        t0 = time.time()
        stacks, heights = collect(args, env)
        print(f"[collect] {args.steps} steps in {time.time() - t0:.1f}s  "
              f"stacks={tuple(stacks.shape)} height={tuple(heights.shape)}", flush=True)
    finally:
        env.close()
        del env
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
    print("[release] simulation objects released; starting perception training", flush=True)
    return stacks, heights, metadata


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[env] device={args.device}  num_envs={args.num_envs}  batch_size={args.batch_size}", flush=True)
    stacks, heights, metadata = collect_and_release(args)

    print(f"[data] sequences={stacks.shape[0]}  stack={tuple(stacks.shape)}  "
          f"height={tuple(heights.shape)}")

    cfg = MGPDPerceptionCfg(depth_hw=DEPTH_HW, height_hw=(17, 11), depth_max=DEPTH_MAX)
    model = MGPDPerception(cfg).to(args.device)
    if args.init_from:
        loaded = model.load_released_weights(args.init_from)
        print(f"[init] loaded from {args.init_from}: {loaded}")
    model.train()

    perm = torch.randperm(stacks.shape[0])
    if stacks.shape[0] < 2:
        raise RuntimeError("Need at least two valid sequences; increase --steps or --sample-envs")
    n_val = min(stacks.shape[0] - 1, max(1, int(stacks.shape[0] * args.val_fraction)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Keep the dataset on CPU and transfer only one minibatch at a time.
    val_stacks = stacks[val_idx]
    val_heights = heights[val_idx]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = args.batch_size
    for it in range(1, args.iterations + 1):
        idx = train_idx[torch.randint(0, train_idx.numel(), (batch,))]
        s = stacks[idx].to(args.device)
        hh = heights[idx].to(args.device).reshape(-1, args.sequence_length, 1, *cfg.height_hw)

        loss, metrics = model.losses(s, hh)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if it % args.log_every == 0 or it == 1:
            val = evaluate(model, val_stacks, val_heights, cfg, batch)
            print(
                f"[train] {it:5d}  total={metrics['total']:.4f}  "
                f"depth={metrics['depth_mse']:.4f}  edge={metrics['edge']:.4f}  "
                f"height={metrics['height_mse']:.4f}  nce={metrics['contrastive']:.4f}  temporal={metrics['temporal']:.4f}  "
                f"| val total={val['total']:.4f} depth={val['depth_mse']:.4f}", flush=True
            )

    out = Path(args.out)
    stats = evaluate(model, val_stacks, val_heights, cfg, batch)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    model.save(temporary, extra={
        "task": args.task,
        "seed": args.seed,
        **metadata,
        "iterations": args.iterations,
        "sequences": int(stacks.shape[0]),
        "sequence_length": args.sequence_length,
        "pairs": int(stacks.shape[0] * args.sequence_length),
        "sample_envs": min(args.sample_envs, args.num_envs),
        "sampling": "uniform_env_index",
        "gru_trained": True,
        "val": stats,
        "validation_batch_size": batch,
        "source": "mjlab-random-rollout",
    })
    temporary.replace(out)
    print(f"[save] {out.resolve()}", flush=True)
    print(f"[val ] {stats}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("[error] CUDA out of memory. Reduce --num-envs/--terrain-rows/--batch-size "
              "and close other GPU jobs (including Play). No new completed checkpoint was saved.", flush=True)
        raise


