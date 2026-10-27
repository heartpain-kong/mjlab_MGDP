"""One entry for the local browser workbench and common LgDog commands."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.dashboard.catalog import TASKS
from src.dashboard.commands import build_command, discover_models, display_command

ROOT = Path(__file__).resolve().parent


def main():
  for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
      stream.reconfigure(encoding="utf-8")
  parser = argparse.ArgumentParser(description="LgDog 工作台：python lgdog.py（打开浏览器）")
  parser.add_argument("--python", default=sys.executable, help="实际运行训练/回放的 Python，默认当前解释器")
  sub = parser.add_subparsers(dest="mode")
  ui = sub.add_parser("ui", help="启动本地浏览器控制面板（默认操作）")
  ui.add_argument("--port", type=int, default=8765)
  ui.add_argument("--no-browser", action="store_true")
  kline = sub.add_parser("kline", help="启动曲线终端：TensorBoard 曲线按窗口合成 K 线")
  kline.add_argument("--port", type=int, default=8766)
  kline.add_argument("--logdir", action="append", default=None,
                     help="event 目录，可给多次；默认 logs/rsl_rl")
  kline.add_argument("--no-browser", action="store_true")
  web = sub.add_parser("web", help="启动 mjlab 总览：环境/GPU/进程/轮次一页看完，并带起工作台和曲线终端")
  web.add_argument("--port", type=int, default=8770)
  web.add_argument("--host", default="127.0.0.1", help="默认只绑 127.0.0.1；0.0.0.0 允许局域网访问")
  web.add_argument("--logdir", action="append", default=None, help="额外的 event 目录，可给多次")
  web.add_argument("--no-autostart", action="store_true", help="不自动带起工作台和曲线终端")
  web.add_argument("--no-browser", action="store_true")
  sub.add_parser("tasks", help="查看短任务名，无需加载训练依赖")
  models = sub.add_parser("models", help="列出任务的模型")
  models.add_argument("task", nargs="?", default="standard")
  for mode in ("train", "play", "sim", "rewards", "curves"):
    child = sub.add_parser(mode)
    child.add_argument("task", nargs="?", default="standard", help="standard/him_flat/him/terrain/gentle/blind/flat/rough")
    child.add_argument("--dry-run", action="store_true", help="仅显示命令，不执行")
    if mode in {"train", "play"}:
      child.add_argument("--envs", type=int, default=1024 if mode == "train" else 4)
      child.add_argument("--checkpoint", default="")
    if mode == "train":
      child.add_argument("--terrain-curriculum", choices=("manual", "uniform", "lp_acrl"), default="manual")
      child.add_argument("--lp-temperature", type=float, default=2.0)
      child.add_argument("--lp-uniform-mix", type=float, default=0.1)
      child.add_argument("--lp-min-episodes", type=int, default=32)
      child.add_argument("--lp-update-interval-s", type=float, default=100.0)
      child.add_argument("--iterations", type=int, default=16000)
      child.add_argument("--seed", type=int, default=42)
      child.add_argument("--run-name", default="")
      child.add_argument("--terrain-num-rows", type=int, default=None,
                         help="地形课程行数；标准 HIM 默认 10")
      child.add_argument("--terrain-difficulty-max", type=float, default=None,
                         help="最大地形难度倍率，默认 1.0")
      child.add_argument("--velocity-min-abs", type=float, default=None,
                         help="最低速度绝对值；训练命令下限会自动取负值")
      child.add_argument("--velocity-max-abs", type=float, default=None,
                         help="最高速度绝对值；训练命令上限会自动取正值")
      child.add_argument("--resume", action="store_true")
    if mode == "play":
      child.add_argument("--viewer", choices=("viser", "native"), default="viser")
    if mode == "sim":
      child.add_argument("--onnx", default="")
      child.add_argument("--map", default="flat")
      child.add_argument("--map-file", default="", help="外部地形 XML，优先于 --map")
      child.add_argument("--vx", type=float, default=0.4)
      child.add_argument("--vy", type=float, default=0.0)
      child.add_argument("--wz", type=float, default=0.0)
      child.add_argument("--spawn-x", type=float, default=0.0)
      child.add_argument("--spawn-y", type=float, default=0.0)
      child.add_argument("--spawn-yaw", type=float, default=0.0)
      child.add_argument("--duration", type=float, default=20)
      child.add_argument("--speed-x", type=float, default=1.0)
      child.add_argument("--speed-y", type=float, default=0.5)
      child.add_argument("--speed-yaw", type=float, default=1.0)
      child.add_argument("--action-clip", type=float, default=1.0)
      child.add_argument("--heading-hold", action="store_true", default=True)
      child.add_argument("--no-heading-hold", dest="heading_hold", action="store_false")
      child.add_argument("--check-obs", action="store_true")
      child.add_argument("--headless", action="store_true")
      child.add_argument("--keyboard", action="store_true")
    if mode == "rewards":
      child.add_argument("--term", default="")
    if mode == "curves":
      child.add_argument("--port", type=int, default=6006)
  args = parser.parse_args()
  python = shutil.which(args.python) or str(Path(args.python).expanduser().resolve())
  if not Path(python).is_file():
    parser.error(f"Python 不存在：{python}")
  try:
    if args.mode in (None, "ui"):
      from src.dashboard.server import serve
      serve(ROOT, python, port=getattr(args, "port", 8765), open_browser=not getattr(args, "no_browser", False))
      return 0
    if args.mode == "kline":
      from src.kline.server import serve as serve_kline
      serve_kline(ROOT, args.logdir, port=args.port, open_browser=not args.no_browser)
      return 0
    if args.mode == "web":
      from src.portal.server import serve as serve_portal
      return serve_portal(ROOT, python, args.logdir, port=args.port, host=args.host,
                          autostart=not args.no_autostart, open_browser=not args.no_browser) or 0
    if args.mode == "tasks":
      for alias, task in TASKS.items():
        print(f"{alias:10} {task['id']}\n           {task['description']}")
      return 0
    if args.mode == "models":
      print(json.dumps(discover_models(ROOT, args.task), ensure_ascii=False, indent=2))
      return 0
    options = vars(args).copy()
    if args.mode == "sim":
      options["show_viewer"] = not args.headless
    argv = build_command(ROOT, python, options)
    print(display_command(argv), flush=True)
    if args.dry_run:
      return 0
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call(argv, cwd=ROOT, env=env, shell=False)
  except (ValueError, OSError) as error:
    parser.error(str(error))


if __name__ == "__main__":
  raise SystemExit(main())
