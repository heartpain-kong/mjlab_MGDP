"""Check that the LgDog runtime is installed and task registration works."""

from __future__ import annotations

import importlib.metadata as metadata
import sys


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

  print("PASS: LgDog runtime is ready")
  for dist, version in versions.items():
    print(f"  {dist}=={version}")
  print(f"  python={sys.executable}")
  print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
  print(f"  registered_tasks={len(tasks)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
