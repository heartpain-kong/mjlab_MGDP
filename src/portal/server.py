"""mjlab 总览：一个页面看本机状态，并把工作台 / 曲线终端 / TensorBoard 串起来。

端口默认 8770，只监听 127.0.0.1。三个子服务默认自动带起（``--no-autostart`` 可关）。
页面本身只读；启动/停止子服务需要页面里的 token（跨站请求会被拦掉）。
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from src.dashboard.history import read_history

from . import system
from .services import Services

WEB = Path(__file__).parent / "web"
ASSETS = {
  "/": ("index.html", "text/html; charset=utf-8"),
  "/index.html": ("index.html", "text/html; charset=utf-8"),
  "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
  "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
  "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
OVERVIEW_TAGS = (
  "Train/mean_reward",
  "Curriculum/terrain_levels/level_mean",
  "Curriculum/terrain_levels/tracking_ratio",
  "Curriculum/terrain_levels/reset_batch_failure_fraction",
  "Episode_Metrics/velocity_error_xy_mps",
  "Policy/mean_std",
  "Perf/total_fps",
)


def discover_logdirs(root, extra=None):
  """默认数据源：工程的 logs/rsl_rl + 附近的「数据」归档目录（有才算）。

  归档目录不一定紧挨着工程（这份工程还套了一层 ``unitree_rl_mjlab_lgdog/``），
  所以往上找两级，认得 ``数据`` / ``data`` 两种叫法。
  """
  root = Path(root).resolve()
  candidates = [root / "logs" / "rsl_rl"]
  for base in (root.parent, root.parent.parent):
    candidates.extend(base / name for name in ("数据", "data"))
  for path in extra or []:
    candidates.append(Path(path).expanduser())
  seen, result = set(), []
  for path in candidates:
    try:
      resolved = path.resolve()
    except OSError:
      continue
    if resolved in seen or not resolved.is_dir():
      continue
    seen.add(resolved)
    result.append(resolved)
  return result


def create_server(root, python, logdirs=None, port=8770, host="127.0.0.1", autostart=True, ports=None):
  root = Path(root).resolve()
  logdirs = discover_logdirs(root, logdirs)
  services = Services(root, python, logdirs,
                      ports=ports or {"dashboard": 8765, "kline": 8766, "tensorboard": 6006})
  token = secrets.token_urlsafe(32)
  started = time.time()
  cache = {}
  lock = threading.Lock()

  def cached(key, ttl, factory):
    now = time.time()
    with lock:
      item = cache.get(key)
      if item and now - item[0] < ttl:
        return item[1]
    value = factory()
    with lock:
      cache[key] = (time.time(), value)
    return value

  store = None

  def run_store():
    nonlocal store
    if store is None:
      from src.kline.store import Store
      store = Store(logdirs, root / "logs" / "kline_cache")
      store.scan()
    elif time.time() - store.last_scan > 20:
      store.scan()
    return store

  def runs_payload():
    def build():
      try:
        store = run_store()
      except Exception as error:  # 归档目录不存在也不该让整页挂
        return {"source": "none", "error": f"{type(error).__name__}: {error}", "items": []}
      items = []
      for run in store.ordered()[:40]:
        try:
          depth = len(run.path.relative_to(run.root).parts)
        except ValueError:
          depth = 2
        items.append({
          "id": run.id, "label": run.label, "stamp": run.stamp,
          # 归档目录是扁的（数据/V14），这时 experiment 就是根目录名，显示出来没意义
          "experiment": run.experiment if depth > 1 else "",
          "size": run.size, "mtime": run.mtime, "running": run.running,
          "state": run.state, "path": str(run.path), "values": {},
        })
      source = "files"
      # 曲线终端在跑的话，直接拿它的 overview（已解析、带指标），顺便保证两边 id 一致
      overview = services.describe("kline")
      if overview["status"] in {"running", "external"}:
        try:
          url = f"http://127.0.0.1:{overview['port']}/api/overview?tags=" + ",".join(OVERVIEW_TAGS)
          with urlopen(url, timeout=2.5) as response:
            data = json.load(response)
          by_id = {row["id"]: row for row in data.get("runs", [])}
          matched = 0
          for item in items:
            row = by_id.get(item["id"])
            if row:
              item["values"] = row.get("values") or {}
              item["state"] = row.get("state") or item["state"]
              item["running"] = bool(row.get("running", item["running"]))
              matched += 1
          if matched:
            source = "kline"
        except Exception:
          pass
      return {"source": source, "items": items, "logdirs": [str(path) for path in logdirs]}

    return cached("runs", 8.0, build)

  def state_payload():
    disks = cached("resources", 5.0, lambda: system.resources(root))
    return {
      "token": token,
      "root": str(root),
      "python": str(python),
      "host": host,
      "port": port,
      "started": started,
      "now": time.time(),
      "autostart": autostart,
      "logdirs": [str(path) for path in logdirs],
      "environment": cached("environment", 60.0, lambda: system.environment(root, python)),
      "resources": disks,
      "gpu": system.gpu(),
      "gpu_processes": cached("gpu_processes", 5.0, system.gpu_processes),
      "processes": cached("processes", 3.0, lambda: system.processes(root, exclude=(os.getpid(), os.getppid()))),
      "services": services.list(),
      "runs": runs_payload(),
      "logs": cached("logs", 8.0, lambda: system.recent_logs(root)),
      "history": cached("history", 15.0, lambda: read_history(root, limit=12)),
    }

  def allowed_log_path(value):
    """只允许读工程 logs/ 下的文件，防止把任意路径当接口用。"""
    base = (root / "logs").resolve()
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else (root / candidate)).resolve()
    if base not in resolved.parents and resolved != base:
      raise ValueError("只能读取工程 logs/ 下的日志")
    if not resolved.is_file():
      raise FileNotFoundError("日志文件不存在")
    return resolved

  class Handler(BaseHTTPRequestHandler):
    server_version = "MjlabPortal/1.0"

    def log_message(self, *args):
      pass

    def valid_host(self):
      if host not in {"127.0.0.1", "localhost"}:
        return True  # 绑到局域网时由 token 兜底
      try:
        address = urlparse("http://" + self.headers.get("Host", ""))
        return address.hostname in {"127.0.0.1", "localhost"} and address.port == self.server.server_port
      except ValueError:
        return False

    def send(self, value, status=200, content_type="application/json; charset=utf-8"):
      body = json.dumps(value, ensure_ascii=False).encode("utf-8") if isinstance(value, (dict, list)) else value
      if isinstance(body, str):
        body = body.encode("utf-8")
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.send_header("X-Content-Type-Options", "nosniff")
      self.send_header("Content-Security-Policy", CSP)
      self.end_headers()
      self.wfile.write(body)

    def fail(self, message, status=400):
      return self.send({"error": str(message)}, status)

    def do_GET(self):
      if not self.valid_host():
        return self.fail("只接受本机地址访问", 403)
      parsed = urlparse(self.path)
      query = parse_qs(parsed.query)
      try:
        if parsed.path in ASSETS:
          filename, mime = ASSETS[parsed.path]
          return self.send((WEB / filename).read_bytes(), content_type=mime)
        if parsed.path == "/api/state":
          return self.send(state_payload())
        if parsed.path == "/api/log":
          name = (query.get("name") or [""])[0]
          if name:
            return self.send(services.tail(name, int((query.get("lines") or ["80"])[0])))
          path = (query.get("path") or [""])[0]
          if not path:
            return self.fail("缺少 name 或 path")
          target = allowed_log_path(path)
          lines = int((query.get("lines") or ["120"])[0])
          return self.send({"path": str(target), "text": "\n".join(system.tail_lines(target, lines)),
                            "size": target.stat().st_size, "exists": True})
        return self.fail("页面不存在", 404)
      except (KeyError, ValueError) as error:
        return self.fail(error, 400)
      except FileNotFoundError as error:
        return self.fail(error, 404)
      except OSError as error:
        return self.fail(f"{type(error).__name__}: {error}", 500)

    def do_POST(self):
      supplied = self.headers.get("X-Mjlab-Token", "")
      origin = self.headers.get("Origin")
      expected = f"http://{self.headers.get('Host', '')}"
      if not self.valid_host() or not hmac.compare_digest(supplied, token) or (origin and origin != expected):
        return self.fail("请刷新页面后再操作", 403)
      try:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 4096:
          raise ValueError("请求大小无效")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
          raise ValueError("请求必须为对象")
        if self.path == "/api/service":
          name = str(data.get("name", ""))
          action = str(data.get("action", ""))
          if action == "start":
            meta = services.start(name)
            return self.send({"service": services.wait_ready(name, timeout=10)})
          if action == "stop":
            return self.send({"service": services.stop(name)})
          raise ValueError("action 只能是 start 或 stop")
        return self.fail("接口不存在", 404)
      except (KeyError, ValueError, OSError) as error:
        return self.fail(str(error).strip("'\""), 400)

  server = ThreadingHTTPServer((host, port), Handler)
  server.daemon_threads = True
  server.services = services
  server.start_services = (lambda: _autostart(services)) if autostart else (lambda: None)
  return server


def _autostart(services):
  for name in ("dashboard", "kline"):
    try:
      meta = services.describe(name)
      if meta["status"] == "stopped":
        services.start(name)
        services.wait_ready(name, timeout=8)
    except Exception:
      continue


def serve(root, python, logdirs=None, port=8770, host="127.0.0.1", autostart=True, open_browser=True,ports=None):
  try:
    server = create_server(root, python, logdirs, port, host, autostart,ports=ports)
  except OSError as error:
    print(f"端口 {port} 起不来：{error}", flush=True)
    print(f"可能是已经有一个总览在跑，或者端口被别的程序占了；试一下 --port {port + 1}", flush=True)
    return 1
  url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{server.server_port}"
  print(f"mjlab 总览: {url}", flush=True)
  print(f"工程: {Path(root).resolve()}", flush=True)
  print(f"训练 Python: {python}", flush=True)
  for path in server.services.logdirs:
    print(f"  数据源: {path}", flush=True)
  if host not in {"127.0.0.1", "localhost"}:
    print("⚠ 已绑定到所有网卡：局域网内任何人都能打开这个页面（启动/停止服务仍需页面里的 token）。", flush=True)
  print("子服务："+" · ".join(f"{item['label']} :{item['port']}" for item in server.services.definitions())+"（Ctrl+C只停总览）",flush=True)
  threading.Thread(target=server.start_services, daemon=True).start()
  if open_browser:
    import webbrowser
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("已停止（子服务继续运行）", flush=True)
  finally:
    server.server_close()
