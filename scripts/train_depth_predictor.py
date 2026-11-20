"""Train the WMP-style depth predictor on random-rollout perception data.

Pipeline
--------
1. Build the same LgDog1 MGDP terrain environment ``scripts/pretrain_mgdp.py``
   uses (reused through ``pretrain_mgdp.build_env``), roll random actions and
   record matched triples ``(depth image, height scan, proprioception)``.
2. Train :class:`src.tasks.velocity.rl.depth_predictor.DepthPredictor` to map
   ``(height, prop) -> depth`` with ``MSE + 0.1 * Sobel edge`` loss.
3. Save the checkpoint for camera-free perception (predict the depth the frozen
   MGDP encoder would have seen) or as an auxiliary depth-reconstruction signal.

Why not call ``pretrain_mgdp.collect`` directly
-----------------------------------------------
That collector emits *two-frame depth stacks + height maps* for the contrastive
GRU perception model and never records proprioception; it also builds
``MGPDPerceptionCfg()`` with its legacy 16x16 default depth grid. The reusable
parts are environment construction (``pretrain_mgdp.build_env``), the existing
ray-distance -> depth-image conversion (``mgdp_observations._depth_image``) and
mjlab's ``height_scan``; the minimal loop below records exactly the triples this
predictor consumes. Data stays in CPU memory, and the simulation is released
before training (same pattern as ``pretrain_mgdp.collect_and_release``).

Examples
--------
    # default rough terrain, 64 envs, 2000 steps, 32 sampled envs per step
    python scripts/train_depth_predictor.py --collect-steps 2000 --num-envs 64

    # race obstacles (2027 course) and a custom checkpoint path
    python scripts/train_depth_predictor.py --task LgDog1-Race-MGDP \\
        --num-envs 64 --collect-steps 3000 --out logs/mgdp/depth_predictor_race.pt

    # smoke test
    python scripts/train_depth_predictor.py --num-envs 4 --collect-steps 5 \\
        --sample-envs 4 --epochs 1 --batch-size 4
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.config.mgdp_pretrain import pretrain_resources  # noqa: E402

TASKS = (
  "LgDog1-Rough-MGDP",
  "LgDog1-Race-MGDP",
  "LgDog1-Race-MGDP-WallTraining",
)
DEFAULT_OUT = "logs/mgdp/depth_predictor.pt"
# ``terrain_scan.max_distance`` in src/tasks/velocity/velocity_env_cfg.py; read
# from the env config at runtime, this is only the fallback.
DEFAULT_HEIGHT_SCAN_MAX = 5.0
HEIGHT_HW = (17, 11)  # 17 x 11 grid, 1.6 x 1.0 m at 0.1 m resolution -> 187 rays
PROP_ORDER = (
  "base_ang_vel(3)",
  "projected_gravity(3)",
  "command(3)",
  "phase(2)",
  "joint_pos(12)",
  "joint_vel(12)",
  "last_action(12)",
)
PROP_DIM = 3 + 3 + 3 + 2 + 12 + 12 + 12


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  p.add_argument("--task", choices=TASKS, default="LgDog1-Rough-MGDP",
                 help="perception-collection terrain; race uses the 2027 obstacles")
  p.add_argument("--num-envs", type=int, default=64, help="parallel environments")
  p.add_argument("--sample-envs", type=int, default=32,
                 help="evenly spaced envs recorded per step (terrain diversity)")
  p.add_argument("--collect-steps", type=int, default=2000,
                 help="policy steps to roll random actions for")
  p.add_argument("--action-hold", type=int, default=25,
                 help="hold a random action for N steps (smoother motion, fewer resets)")
  p.add_argument("--terrain-rows", type=int, default=None)
  p.add_argument("--terrain-cols", type=int, default=None)
  p.add_argument("--epochs", type=int, default=50)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--edge-weight", type=float, default=0.1)
  p.add_argument("--log-every", type=int, default=10, help="print every N epochs")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default=None,
                 help="defaults to cuda when available, else cpu")
  p.add_argument("--optimize-vram", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--collision-mode", choices=("auto", "sap"), default="auto")
  p.add_argument("--out", default=DEFAULT_OUT, help="checkpoint path")
  args = p.parse_args()

  resources = pretrain_resources(args.task)
  if args.terrain_rows is None:
    args.terrain_rows = resources.terrain_rows
  if args.terrain_cols is None and resources.terrain_cols is not None:
    args.terrain_cols = resources.terrain_cols
  args.sample_envs = min(args.sample_envs, args.num_envs)

  if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
  if args.num_envs < 1 or args.sample_envs < 1:
    p.error("--num-envs and --sample-envs must be positive")
  if args.collect_steps < 1:
    p.error("--collect-steps must be positive")
  if args.action_hold < 1:
    p.error("--action-hold must be positive")
  if args.epochs < 1 or args.batch_size < 1:
    p.error("--epochs and --batch-size must be positive")
  if args.lr <= 0.0:
    p.error("--lr must be positive")
  if args.edge_weight < 0.0 or args.log_every < 0:
    p.error("--edge-weight and --log-every must be non-negative")
  if args.terrain_rows < 1 or (args.terrain_cols is not None and args.terrain_cols < 1):
    p.error("--terrain-rows/--terrain-cols must be positive")
  if args.task == "LgDog1-Race-MGDP-WallTraining" and args.terrain_rows != 8:
    p.error("WallTraining requires exactly 8 terrain rows")
  if args.task == "LgDog1-Race-MGDP" and (args.terrain_cols or 0) < 10:
    p.error("race terrain-cols must be >=10 to retain every obstacle type")
  return args


def _pretrain_module():
  """Import ``scripts/pretrain_mgdp.py`` as a module (reuse its env builder)."""
  try:
    import pretrain_mgdp  # when running ``python scripts/train_depth_predictor.py``
    return pretrain_mgdp
  except ImportError:
    import importlib.util

    path = _REPO_ROOT / "scripts" / "pretrain_mgdp.py"
    spec = importlib.util.spec_from_file_location("pretrain_mgdp", path)
    if spec is None or spec.loader is None:
      raise ImportError(f"cannot import pretrain_mgdp from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pretrain_mgdp"] = module
    spec.loader.exec_module(module)
    return module


def build_env(args: argparse.Namespace):
  """Build the MGDP collection env through ``pretrain_mgdp.build_env``."""
  pretrain = _pretrain_module()
  # ``build_env`` reads a few optional attributes via getattr; a plain namespace
  # keeps this script independent of pretrain_mgdp's CLI parser.
  env_args = SimpleNamespace(
    task=args.task,
    num_envs=args.num_envs,
    seed=args.seed,
    device=args.device,
    optimize_vram=args.optimize_vram,
    collision_mode=args.collision_mode,
    terrain_rows=args.terrain_rows,
    terrain_cols=args.terrain_cols,
  )
  return pretrain.build_env(env_args)


def proprioception(env) -> torch.Tensor:
  """The 47-dim actor proprioception frame, in actor-observation order.

  Mirrors the MGDP task's actor terms without observation noise:
  ang_vel(3) + projected_gravity(3) + command(3) + phase(2) + joint_pos(12)
  + joint_vel(12) + last_action(12). This is the exact vector a policy or a
  camera-free rollout can hand to the predictor.
  """
  import src.tasks.velocity.mdp as mdp

  return torch.cat(
    [
      mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel"),
      mdp.projected_gravity(env),
      mdp.generated_commands(env, command_name="twist"),
      mdp.phase(env, period=0.6, command_name="twist"),
      mdp.joint_pos_rel(env),
      mdp.joint_vel_rel(env),
      mdp.last_action(env),
    ],
    dim=-1,
  )


def _height_scan_max_distance(env) -> float:
  """Height-scan normaliser: the same ``1 / max_distance`` scale the task uses."""
  scene = getattr(env.cfg, "scene", None)
  for sensor in getattr(scene, "sensors", None) or ():
    if getattr(sensor, "name", None) == "terrain_scan":
      return float(sensor.max_distance)
  return DEFAULT_HEIGHT_SCAN_MAX


@torch.no_grad()
def collect_depth_data(
  args: argparse.Namespace, env
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
  """Roll random actions and record (depth, height, prop) triples on CPU.

  Returns ``depth`` (S, 1, 24, 32) in [0, 1] (metres / ``DEPTH_MAX``),
  ``height`` (S, 187) in [-1, 1] (``height_scan / max_distance``), ``prop``
  (S, 47) as built by :func:`proprioception`, plus the normalisation metadata
  that is stored in the checkpoint.
  """
  from mjlab.envs.mdp import height_scan

  from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import DEPTH_HW, DEPTH_MAX
  from src.tasks.velocity.mdp.mgdp_observations import _depth_image
  from src.tasks.velocity.mdp.mgdp_perception import MGPDPerceptionCfg

  # Same depth contract as ``lgdog1_mgdp_env_cfg``'s ``with_perception`` path:
  # 24x32 rays, metres / 2.0 m, clamped to the encoder's [0.1, 2.0] m range.
  perception_cfg = MGPDPerceptionCfg(
    depth_hw=DEPTH_HW,
    height_hw=HEIGHT_HW,
    depth_max=DEPTH_MAX,
  )
  sample_envs = min(args.sample_envs, env.num_envs)
  # Terrain types are grouped by env index; evenly spaced ids see every column
  # instead of only the first few (same choice as pretrain_mgdp.collect).
  sample_ids = torch.linspace(
    0, env.num_envs - 1, sample_envs, device=env.device
  ).round().long()
  scan_max = _height_scan_max_distance(env)
  action_dim = env.action_manager.total_action_dim
  print(
    f"[collect] {args.collect_steps} steps x {sample_envs} envs = "
    f"{args.collect_steps * sample_envs} triples; depth={DEPTH_HW}, "
    f"height_scan/{scan_max:.1f} m, prop={PROP_DIM} dims",
    flush=True,
  )

  depths: list[torch.Tensor] = []
  heights: list[torch.Tensor] = []
  props: list[torch.Tensor] = []
  env.reset()
  actions = torch.zeros((env.num_envs, action_dim), device=env.device)
  for step in range(args.collect_steps):
    if step % args.action_hold == 0:
      actions = (torch.rand((env.num_envs, action_dim), device=env.device) - 0.5) * 0.3
    env.step(actions)

    depths.append(_depth_image(env, "depth_rays", perception_cfg)[sample_ids].float().cpu())
    height = height_scan(env, "terrain_scan")[sample_ids]
    heights.append((height / scan_max).clamp(-1.0, 1.0).float().cpu())
    props.append(proprioception(env)[sample_ids].float().cpu())

    if step % 200 == 0 or step == args.collect_steps - 1:
      print(f"[collect] {step + 1}/{args.collect_steps} steps", flush=True)

  if not depths:
    raise RuntimeError("no samples collected; increase --collect-steps")
  metadata = {
    "depth_hw": list(DEPTH_HW),
    "depth_max_m": float(DEPTH_MAX),
    "depth_min_m": float(perception_cfg.depth_min),
    "depth_unit": "metres / depth_max",
    "height_hw": list(HEIGHT_HW),
    "height_scan_max_distance_m": scan_max,
    "height_unit": "height_scan / terrain_scan.max_distance, clamp [-1, 1]",
    "prop_order": list(PROP_ORDER),
  }
  return torch.cat(depths), torch.cat(heights), torch.cat(props), metadata


def main() -> None:
  args = parse_args()
  random.seed(args.seed)
  torch.manual_seed(args.seed)

  from src.tasks.velocity.rl.depth_predictor import (
    DepthPredictor,
    train_depth_predictor,
  )

  print(
    f"[env] task={args.task} device={args.device} num_envs={args.num_envs} "
    f"terrain={args.terrain_rows}x{args.terrain_cols}",
    flush=True,
  )
  started = time.time()
  env = build_env(args)
  try:
    print(f"[env] ready in {time.time() - started:.1f}s", flush=True)
    t0 = time.time()
    depth, height, prop, metadata = collect_depth_data(args, env)
    print(
      f"[data] collected in {time.time() - t0:.1f}s  depth={tuple(depth.shape)} "
      f"height={tuple(height.shape)} prop={tuple(prop.shape)}",
      flush=True,
    )
  finally:
    env.close()
    del env
    gc.collect()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
  print("[release] simulation released; training the depth predictor", flush=True)

  print(
    f"[data] depth range [{float(depth.min()):.3f}, {float(depth.max()):.3f}] "
    f"mean={float(depth.mean()):.4f}; height range "
    f"[{float(height.min()):.3f}, {float(height.max()):.3f}]",
    flush=True,
  )
  if prop.shape[-1] != PROP_DIM:
    print(
      f"[warn] proprioception is {prop.shape[-1]}-dim, not the {PROP_DIM}-dim "
      "actor frame; the predictor adapts, but a deployment stack must pass the "
      "same vector",
      flush=True,
    )

  model = DepthPredictor(
    height_dim=height.shape[-1],
    prop_dim=prop.shape[-1],
    out_hw=tuple(depth.shape[-2:]),
  )
  print(
    f"[model] DepthPredictor(height_dim={model.height_dim}, "
    f"prop_dim={model.prop_dim}, out_hw={model.out_hw})  "
    f"parameters={model.parameter_count():,}",
    flush=True,
  )

  history = train_depth_predictor(
    depth,
    height,
    prop,
    epochs=args.epochs,
    lr=args.lr,
    batch_size=args.batch_size,
    edge_weight=args.edge_weight,
    log_every=args.log_every,
    device=args.device,
    seed=args.seed,
    model=model,
  )
  print(
    f"[result] epochs={len(history)}  final_mse={history.final_mse:.6f}  "
    f"final_loss={history.loss[-1]:.6f}  final_edge={history.edge[-1]:.6f}",
    flush=True,
  )

  out = model.save(
    args.out,
    extra={
      "task": args.task,
      "seed": args.seed,
      "device": str(args.device),
      "source": "mjlab-random-rollout",
      "collect_steps": args.collect_steps,
      "num_envs": args.num_envs,
      "sample_envs": args.sample_envs,
      "action_hold": args.action_hold,
      "samples": int(depth.shape[0]),
      "data": metadata,
      "epochs": args.epochs,
      "lr": args.lr,
      "batch_size": args.batch_size,
      "edge_weight": args.edge_weight,
      "final_mse": history.final_mse,
      "history": history.as_dict(),
    },
  )
  print(f"[save] {Path(out).resolve()}", flush=True)


if __name__ == "__main__":
  print("[init] importing PyTorch and simulation libraries...", flush=True)
  main()
