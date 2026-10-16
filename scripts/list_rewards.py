"""Show final reward weights and source locations, without creating a simulator."""

import argparse
import inspect
from pathlib import Path
import pprint
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_location(func):
  path = inspect.getsourcefile(func)
  if path is None:
    return "<source unavailable>"
  try:
    line = inspect.getsourcelines(func)[1]
  except (OSError, TypeError):
    return str(path)
  try:
    path = Path(path).resolve().relative_to(ROOT)
  except ValueError:
    pass
  return f"{path}:{line}"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", nargs="?", default="LgDog1-Rough-Blind-HIM")
  parser.add_argument("--term", help="Show full parameters and source for one reward")
  parser.add_argument("--play", action="store_true", help="Inspect play instead of training config")
  args = parser.parse_args()

  # Registry configs are enough: no ManagerBasedRlEnv, GPU or checkpoint required.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  if args.task not in list_tasks():
    parser.error(f"Unknown task {args.task!r}; use scripts/list_envs.py to list task IDs")
  cfg = load_env_cfg(args.task, play=args.play)
  if args.term and args.term not in cfg.rewards:
    parser.error(f"Unknown reward {args.term!r}; choose: {', '.join(cfg.rewards)}")
  print(f"Task: {args.task} ({'play' if args.play else 'train'})")
  print(f"Control dt: {cfg.sim.mujoco.timestep * cfg.decimation:g} s")
  print("Per-step contribution = raw reward * weight * control dt; weight 0 = disabled")
  if args.task.startswith("LgDog1-"):
    print("Edit: src/tasks/velocity/config/lgdog1/reward_cfg.py")
  print()
  print(f"{'Reward':28} {'Weight':>12}  Function")
  print("-" * 85)
  for name, term in cfg.rewards.items():
    if args.term and name != args.term:
      continue
    print(f"{name:28} {term.weight:12g}  {term.func.__module__}.{term.func.__qualname__}")
    print(f"  source: {source_location(term.func)}")
    if args.term:
      print("  params:")
      print(pprint.pformat(term.params, width=100, sort_dicts=False))


if __name__ == "__main__":
  main()
