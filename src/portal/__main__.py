"""``python -m src.portal`` —— 起一个 mjlab 总览页。

    python -m src.portal                  # http://127.0.0.1:8770，自动带起工作台 + 曲线终端
    python -m src.portal --no-autostart   # 只起总览，服务按钮自己点
    python -m src.portal --host 0.0.0.0   # 让局域网里的手机/平板也能看（有 token 保护写操作）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .server import create_server, discover_logdirs, serve

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
  for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
      stream.reconfigure(encoding="utf-8")
  parser = argparse.ArgumentParser(description="mjlab 总览：一个页面看本机状态")
  parser.add_argument("--port", type=int, default=8770)
  parser.add_argument("--host", default="127.0.0.1", help="默认只绑 127.0.0.1；0.0.0.0 表示允许局域网访问")
  parser.add_argument("--logdir", action="append", default=None, help="额外的 event 目录（可给多次）")
  parser.add_argument("--python", default=sys.executable, help="训练/子服务用的 Python")
  parser.add_argument("--no-autostart", action="store_true", help="不自动带起工作台和曲线终端")
  parser.add_argument("--no-browser", action="store_true")
  parser.add_argument("--once", action="store_true", help="只打印数据源和状态后退出（自检用）")
  args = parser.parse_args(argv)
  if args.once:
    server = create_server(ROOT, args.python, args.logdir, port=0, autostart=False)
    try:
      print(f"数据源: {[str(path) for path in server.services.logdirs]}")
      print(f"服务: {[(item['name'], item['status'], item['port']) for item in server.services.list()]}")
    finally:
      server.server_close()
    return 0
  return serve(ROOT, args.python, args.logdir, port=args.port, host=args.host,
               autostart=not args.no_autostart, open_browser=not args.no_browser) or 0


if __name__ == "__main__":
  raise SystemExit(main())
