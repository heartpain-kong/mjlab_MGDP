"""Check that the LgDog runtime is installed and task registration works."""

from __future__ import annotations

import importlib.metadata as metadata
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


REQUIRED_DISTS = (
  "mjlab",
  "mujoco",
  "mujoco-warp",
  "warp-lang",
  "rsl-rl-lib",
  "torch",
  "onnx",
  "onnxruntime",
)


def main() -> int:
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--require-cuda',action='store_true',help='Require real PyTorch and Warp CUDA allocations')
  args=parser.parse_args()
  missing = []
  versions = {}
  for dist in REQUIRED_DISTS:
    try:
      versions[dist] = metadata.version(dist)
    except metadata.PackageNotFoundError:
      missing.append(dist)
  if missing:
    print(f"FAIL: missing distributions: {', '.join(missing)}", file=sys.stderr)
    return 1

  import torch

  import mjlab  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  tasks = list_tasks()
  required_tasks = {
    "LgDog1-Rough-Blind-HIM",
    "LgDog1-Rough-Blind-HIM-Standard",
  }
  missing_tasks = sorted(required_tasks.difference(tasks))
  if missing_tasks:
    print(f"FAIL: LgDog tasks were not registered: {', '.join(missing_tasks)}", file=sys.stderr)
    return 1

  from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
  env_cfg=load_env_cfg('LgDog1-Rough-Blind-HIM-Standard')
  agent_cfg=load_rl_cfg('LgDog1-Rough-Blind-HIM-Standard')
  if env_cfg.observations['actor'].history_length!=3 or agent_cfg.actor.history_length!=3:
    print('FAIL: expected the exported three-frame HIM configuration',file=sys.stderr)
    return 1
  if args.require_cuda:
    if not torch.cuda.is_available():
      print('FAIL: CUDA is unavailable in this Python environment; check GPU/driver/PyTorch wheel',file=sys.stderr)
      return 1
    try:
      import warp as wp
      torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
      wp.init();device=wp.get_device('cuda:0')
      probe=wp.zeros(1,dtype=wp.float32,device=device);wp.synchronize_device(device)
      print(f'PASS: PyTorch + Warp CUDA allocations, {device}, {torch.cuda.get_device_name(0)}')
      del probe
    except Exception as error:
      print(f'FAIL: CUDA/Warp allocation failed: {error}',file=sys.stderr)
      return 1

  print("PASS: LgDog runtime is ready")
  for dist, version in versions.items():
    print(f"  {dist}=={version}")
  print(f"  python={sys.executable}")
  print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
  print(f"  registered_tasks={len(tasks)}")
  print('  HIM history=3, frame=47, input=141; foot_clearance='+str(env_cfg.rewards['foot_clearance'].weight))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
