"""Loopback-only browser workbench. No external frontend/runtime dependencies."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlparse
import webbrowser

from .catalog import TASKS, resolve_task
from .commands import build_command, discover_models, display_command
from .jobs import JobManager
from .reward_editor import read_rewards, save_rewards

WEB = Path(__file__).parent / "web"
DOCUMENTS = {
  "start": ("使用说明", "doc/WORKBENCH.md"),
  "rewards": ("奖励调参", "doc/REWARDS.md"),
  "standard": ("标准地形 HIM", "doc/LGDOG_HIM_STANDARD.md"),
  "reward-code": ("奖励配置源码", "src/tasks/velocity/config/lgdog1/reward_cfg.py"),
}


def create_server(root, python, host="127.0.0.1", port=8765):
  if host not in {"127.0.0.1", "localhost"}:
    raise ValueError("本地面板只允许绑定 127.0.0.1 或 localhost")
  root = Path(root).resolve()
  jobs = JobManager(root)
  token = secrets.token_urlsafe(32)

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def valid_host(self):
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
      self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      if not self.valid_host():
        return self.send({"error": "只接受本机地址访问"}, 403)
      parsed = urlparse(self.path)
      query = parse_qs(parsed.query)
      try:
        if parsed.path in {"/", "/assets/app.js", "/assets/style.css"}:
          filename, mime = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
          }[parsed.path]
          return self.send((WEB / filename).read_bytes(), content_type=mime)
        if parsed.path == "/api/state":
          return self.send({"token": token, "root": str(root), "python": str(python),
                            "tasks": [{"alias": alias, **task} for alias, task in TASKS.items()],
                            "jobs": jobs.list(), "documents": [{"id": key, "label": value[0]} for key, value in DOCUMENTS.items()]})
        if parsed.path == "/api/models":
          return self.send({"models": discover_models(root, query.get("task", ["standard"])[0])})
        if parsed.path == "/api/rewards":
          alias, task = resolve_task(query.get("task", ["standard"])[0])
          return self.send({"task": alias, **read_rewards(root, task["profile"])})
        if parsed.path == "/api/log":
          return self.send({"text": jobs.tail(query.get("id", [""])[0])})
        if parsed.path == "/api/document":
          document = DOCUMENTS.get(query.get("id", [""])[0])
          if not document:
            raise ValueError("未知文档")
          return self.send({"title": document[0], "path": str(root / document[1]),
                            "text": (root / document[1]).read_text(encoding="utf-8-sig")})
        return self.send({"error": "页面不存在"}, 404)
      except (ValueError, OSError) as error:
        return self.send({"error": str(error)}, 400)

    def do_POST(self):
      # Same-origin token blocks cross-site process launches from unrelated tabs.
      supplied = self.headers.get("X-LgDog-Token", "")
      origin = self.headers.get("Origin")
      expected_origin = f"http://{self.headers.get('Host', '')}"
      if not self.valid_host() or not hmac.compare_digest(supplied, token) or (origin and origin != expected_origin):
        return self.send({"error": "请刷新本地面板后再操作"}, 403)
      try:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 16384:
          raise ValueError("请求大小无效")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
          raise ValueError("请求必须为对象")
        if self.path in {"/api/preview", "/api/run"}:
          argv = build_command(root, python, data)
          if self.path == "/api/preview":
            return self.send({"command": display_command(argv), "argv": argv})
          alias, _ = resolve_task(data.get("task", "standard"))
          return self.send({"job": jobs.launch(argv, data.get("mode", "train"), alias)})
        if self.path == "/api/rewards/save":
          alias, task = resolve_task(data.get("task", "standard"))
          result = save_rewards(root, task["profile"], data.get("changes"))
          return self.send({"task": alias, **result})
        if self.path == "/api/stop":
          return self.send({"job": jobs.stop(str(data.get("id", "")))})
        return self.send({"error": "操作不存在"}, 404)
      except (ValueError, OSError) as error:
        return self.send({"error": str(error)}, 400)

  server = ThreadingHTTPServer((host, port), Handler)
  server.daemon_threads = True
  server.jobs = jobs
  return server


def serve(root, python, host="127.0.0.1", port=8765, open_browser=True):
  server = create_server(root, python, host, port)
  url = f"http://127.0.0.1:{server.server_port}"
  print(f"LgDog 训练工作台: {url}\n工程: {root}\n训练 Python: {python}", flush=True)
  print("关闭浏览器不会停止任务。Ctrl+C 关闭面板服务；任务输出直接保存在日志中。", flush=True)
  if open_browser:
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    for job in server.jobs.list():
      if job["status"] in {"running", "stopping"}:
        print(f"仍在运行: pid={job['pid']} {job['mode']}，日志 {job['log']}")
    server.server_close()
