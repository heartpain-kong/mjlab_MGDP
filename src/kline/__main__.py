"""``python -m src.kline`` —— 起一个本地行情终端。

    python -m src.kline                       # 默认读 logs/rsl_rl，端口 8766
    python -m src.kline --port 9000
    python -m src.kline --logdir D:/archive/logs/rsl_rl --logdir logs/rsl_rl

不加 --no-browser 时会自动打开浏览器。服务只监听 127.0.0.1。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .server import create_server, serve

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
  for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
      stream.reconfigure(encoding="utf-8")
  parser = argparse.ArgumentParser(description="LgDog 曲线终端：TensorBoard 转 K 线")
  parser.add_argument("--logdir", action="append", default=None,
                      help="event 文件所在的目录，可以给多次；默认 logs/rsl_rl")
  parser.add_argument("--port", type=int, default=8766, help="端口，默认 8766")
  parser.add_argument("--cache-dir", default=None, help="缓存目录，默认 logs/kline_cache")
  parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
  parser.add_argument("--once", action="store_true", help="只打印状态后退出（自检用）")
  args = parser.parse_args(argv)
  logdirs = [Path(item) for item in args.logdir] if args.logdir else None
  if args.once:
    server = create_server(ROOT, logdirs, port=0, cache_dir=args.cache_dir)
    try:
      print(f"数据源: {[str(path) for path in server.store.roots]}")
      print(f"轮次: {len(server.store.runs)}")
      for run in server.store.ordered():
        print(f"  {run.label:38} {run.experiment:40} {run.size / 1024 / 1024:7.1f} MB")
    finally:
      server.server_close()
    return 0
  serve(ROOT, logdirs, port=args.port, cache_dir=args.cache_dir, open_browser=not args.no_browser)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
