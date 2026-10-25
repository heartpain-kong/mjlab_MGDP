"""Loopback-only browser workbench. No external frontend/runtime dependencies."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

from .catalog import ALGORITHMS, SCENES, SIM2SIM_MAPS, TASKS, resolve_task
from .commands import build_command, discover_models, display_command, integer
from .run_naming import next_run_name
from . import history
from .jobs import JobManager
from .reward_editor import read_rewards, reward_detail, save_rewards

WEB = Path(__file__).parent / "web"
DOCUMENTS = {
  "start": ("使用说明", "doc/WORKBENCH.md"),
  "rewards": ("奖励调参", "doc/REWARDS.md"),
  "standard": ("标准地形 HIM", "doc/LGDOG_HIM_STANDARD.md"),
  "reward-code": ("奖励配置源码", "src/tasks/velocity/config/lgdog1/reward_cfg.py"),
}

ACTION_METRIC_TAGS = {
  "raw_action_abs_max": "Metrics/raw_action_abs_max",
  "raw_action_abs_mean": "Metrics/raw_action_abs_mean",
  "action_saturation_fraction": "Metrics/action_saturation_fraction",
  "policy_mean_std": "Policy/mean_std",
  "entropy": "Loss/entropy",
  "mean_reward": "Train/mean_reward",
  "terrain_level": "Curriculum/terrain_levels/level_mean",
}


def _nvidia_smi_path():
  candidates = [shutil.which("nvidia-smi")]
  if os.name == "nt":
    candidates.extend([
      r"C:\Windows\System32\nvidia-smi.exe",
      r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ])
  return next((path for path in candidates if path and Path(path).is_file()), None)


def gpu_info():
  """Return lightweight GPU telemetry without importing torch or CUDA libraries."""
  executable = _nvidia_smi_path()
  if not executable:
    return {"available": False, "gpus": [], "reason": "未找到 nvidia-smi"}
  query = [
    executable, "--query-gpu=index,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits",
  ]
  kwargs = {"capture_output": True, "text": True, "timeout": 2, "check": False}
  if os.name == "nt":
    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
  try:
    result = subprocess.run(query, **kwargs)
  except (OSError, subprocess.SubprocessError) as error:
    return {"available": False, "gpus": [], "reason": f"nvidia-smi 读取失败：{type(error).__name__}"}
  if result.returncode != 0:
    reason = (result.stderr or "nvidia-smi 返回错误").strip().splitlines()[0]
    return {"available": False, "gpus": [], "reason": reason[:160]}
  gpus = []
  for row in csv.reader(result.stdout.splitlines()):
    if len(row) < 8:
      continue
    def numeric(value, integer_value=False):
      value = value.strip()
      try:
        number = float(value)
      except ValueError:
        return None
      return int(number) if integer_value else number
    gpus.append({
      "index": numeric(row[0], True), "name": row[1].strip(),
      "memory_total_mib": numeric(row[2], True), "memory_used_mib": numeric(row[3], True),
      "memory_free_mib": numeric(row[4], True), "utilization_gpu": numeric(row[5], True),
      "temperature_c": numeric(row[6], True), "power_w": numeric(row[7]),
    })
  return {"available": bool(gpus), "gpus": gpus,
          "updated": time.time(), "source": executable,
          "reason": None if gpus else "nvidia-smi 没有返回 GPU 数据"}


def create_server(root, python, host="127.0.0.1", port=8765):
  if host not in {"127.0.0.1", "localhost"}:
    raise ValueError("本地面板只允许绑定 127.0.0.1 或 localhost")
  root = Path(root).resolve()
  jobs = JobManager(root, on_event=lambda event, record: history.record_run_event(root, event, record))
  token = secrets.token_urlsafe(32)
  metrics_cache = {}
  metrics_lock = threading.Lock()

  def parse_metrics(alias, task, event_file, stat, signature):
    """Load one event file off the request thread; large runs can be tens of MB."""
    try:
      from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
      accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
      accumulator.Reload()
      available_tags = set(accumulator.Tags().get("scalars", []))
      metrics = {}
      for name, tag in ACTION_METRIC_TAGS.items():
        if tag not in available_tags:
          continue
        points = accumulator.Scalars(tag)
        if points:
          point = points[-1]
          metrics[name] = {"value": float(point.value), "step": int(point.step), "tag": tag}
      raw_max = metrics.get("raw_action_abs_max", {}).get("value")
      saturation = metrics.get("action_saturation_fraction", {}).get("value")
      policy_std = metrics.get("policy_mean_std", {}).get("value")
      warnings = []
      if raw_max is not None and raw_max > 3.0:
        warnings.append("原始动作幅值偏大")
      if saturation is not None and saturation > 0.05:
        warnings.append("动作饱和率偏高")
      if policy_std is not None and policy_std > 3.0:
        warnings.append("策略标准差偏大")
      data = {"task": alias, "experiment": task["experiment"], "available": bool(metrics),
              "run": event_file.parent.name, "updated": stat.st_mtime, "metrics": metrics,
              "warning": "；".join(warnings) if warnings else None}
    except ImportError:
      data = {"task": alias, "experiment": task["experiment"], "available": False,
              "run": event_file.parent.name, "updated": stat.st_mtime,
              "metrics": {}, "warning": "当前 Python 环境未安装 TensorBoard"}
    except Exception as error:  # A partially written event file should not break the panel.
      data = {"task": alias, "experiment": task["experiment"], "available": False,
              "run": event_file.parent.name, "updated": stat.st_mtime,
              "metrics": {}, "warning": f"读取 TensorBoard 失败：{type(error).__name__}"}
    with metrics_lock:
      metrics_cache[alias] = {"signature": signature, "loading": False, "data": data}

  def latest_metrics(task_value):
    """Read the newest TensorBoard scalars without making the dashboard write anything."""
    alias, task = resolve_task(task_value)
    base = root / "logs" / "rsl_rl" / task["experiment"]
    # RSL-RL writes one event file directly inside each run.  Keep this one
    # level deep: recursively walking checkpoints/videos every 1.8 s makes the
    # browser panel appear frozen on a large training archive.
    event_files = sorted(
      (path for path in base.glob("*/events.out.tfevents*") if path.is_file()),
      key=lambda path: path.stat().st_mtime,
      reverse=True,
    ) if base.is_dir() else []
    if not event_files:
      return {"task": alias, "experiment": task["experiment"], "available": False,
              "run": None, "updated": None, "metrics": {}, "warning": None}
    event_file = event_files[0]
    stat = event_file.stat()
    signature = (str(event_file), stat.st_mtime_ns, stat.st_size)
    with metrics_lock:
      cached = metrics_cache.get(alias)
      if cached and cached["signature"] == signature:
        if cached.get("loading"):
          return {"task": alias, "experiment": task["experiment"], "available": False,
                  "run": event_file.parent.name, "updated": stat.st_mtime,
                  "metrics": {}, "warning": "正在读取最新 TensorBoard 记录…"}
        return cached["data"]
      metrics_cache[alias] = {"signature": signature, "loading": True}
    threading.Thread(target=parse_metrics, args=(alias, task, event_file, stat, signature), daemon=True).start()
    return {"task": alias, "experiment": task["experiment"], "available": False,
            "run": event_file.parent.name, "updated": stat.st_mtime,
            "metrics": {}, "warning": "正在读取最新 TensorBoard 记录…"}

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
                            "algorithms": [{"alias": alias, **item} for alias, item in ALGORITHMS.items()],
                            "scenes": [{"alias": alias, **item} for alias, item in SCENES.items()],
                            "sim2sim_maps": [{"alias": alias, **item} for alias, item in SIM2SIM_MAPS.items()],
                            "gpu": gpu_info(), "jobs": jobs.list(),
                            "documents": [{"id": key, "label": value[0]} for key, value in DOCUMENTS.items()]})
        if parsed.path == "/api/gpu":
          return self.send(gpu_info())
        if parsed.path == "/api/run-name":
          alias, task = resolve_task(query.get("task", ["standard"])[0])
          seed = integer(query.get("seed", [42])[0], "seed", 0, 2147483647)
          return self.send({"task": alias, "seed": seed,
                            "name": next_run_name(root / "logs/rsl_rl" / task["experiment"], seed)})
        if parsed.path == "/api/models":
          return self.send({"models": discover_models(root, query.get("task", ["standard"])[0])})
        if parsed.path == "/api/metrics":
          return self.send(latest_metrics(query.get("task", ["standard"])[0]))
        if parsed.path == "/api/rewards":
          alias, task = resolve_task(query.get("task", ["standard"])[0])
          return self.send({"task": alias, **read_rewards(root, task["profile"])})
        if parsed.path == "/api/reward-detail":
          alias, task = resolve_task(query.get("task", ["standard"])[0])
          return self.send({"task": alias, **reward_detail(root, task["profile"], query.get("name", [""])[0])})
        if parsed.path == "/api/log":
          return self.send({"text": jobs.tail(query.get("id", [""])[0])})
        if parsed.path == "/api/history":
          return self.send(history.read_history(root, query.get("limit", [history.DEFAULT_LIMIT])[0]))
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
          before = {item["name"]: {"weight": item["weight"], "params": item["params"]}
                    for item in read_rewards(root, task["profile"])["rewards"]}
          result = save_rewards(root, task["profile"], data.get("changes"))
          after = {item["name"]: {"weight": item["weight"], "params": item["params"]}
                   for item in result.get("rewards") or []}
          history.record_reward_edit(root, task=alias, profile=task["profile"], before=before,
                                     after=after, source=result.get("source"),
                                     backup=result.get("backup"))
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
