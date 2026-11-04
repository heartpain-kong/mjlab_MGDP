"""Loopback-only browser workbench. No external frontend/runtime dependencies."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

from .catalog import resolve_task
from . import routes
from .commands import build_command, discover_models, display_command, integer
from .run_naming import next_run_name
from . import history
from .jobs import JobManager
from .reward_editor import read_rewards, reward_detail, save_rewards
from .terrain_editor import read_terrain, save_terrain

WEB = Path(__file__).parent / "web"
STATIC_FILES = {
  "/": ("index.html", "text/html; charset=utf-8"),
  "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
  "/assets/js/format.js": ("js/format.js", "text/javascript; charset=utf-8"),
  "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
}

ACTION_METRIC_TAGS = {
  "raw_action_abs_max": "Metrics/raw_action_abs_max",
  "raw_action_abs_mean": "Metrics/raw_action_abs_mean",
  "action_saturation_fraction": "Metrics/action_saturation_fraction",
  "policy_mean_std": "Policy/mean_std",
  "entropy": "Loss/entropy",
  "mean_reward": "Train/mean_reward",
  "foot_clearance_height": "Metrics/swing_clearance_height_numerator",
  "foot_clearance_valid": "Metrics/swing_clearance_valid_fraction",
  "velocity_error_xy": "Metrics/twist/error_vel_xy",
  "velocity_error_yaw": "Metrics/twist/error_vel_yaw",
  "velocity_tracking_score": "Episode_Metrics/velocity_tracking_score",
  "velocity_tracking_active": "Episode_Metrics/velocity_tracking_active",
  "terrain_level": ("Curriculum/passage/level_mean", "Curriculum/terrain_levels/level_mean", "Curriculum/terrain_sampling/level_mean"),
  "passage_speed_max": "Curriculum/passage/speed_max",
  "terrain_promote_fraction": "Curriculum/terrain_levels/promote_fraction",
  "terrain_demote_fraction": "Curriculum/terrain_levels/demote_fraction",
  "terrain_tracking_ratio": ("Curriculum/terrain_levels/tracking_ratio", "Curriculum/terrain_sampling/tracking_ratio"),
  "terrain_velocity_error": "Curriculum/terrain_levels/velocity_error_xy_mps",
  "terrain_failure_fraction": ("Curriculum/terrain_levels/reset_batch_failure_fraction", "Curriculum/terrain_sampling/reset_batch_failure_fraction"),
  "terrain_velocity_gate_fraction": "Curriculum/terrain_levels/velocity_gate_pass_fraction",
  "terrain_progress_gate_fraction": "Curriculum/terrain_levels/progress_gate_pass_fraction",
  "terrain_ratio_gate_fraction": "Curriculum/terrain_levels/ratio_gate_pass_fraction",
  "terrain_distance_gate_fraction": "Curriculum/terrain_levels/distance_gate_pass_fraction",
  "terrain_exposure_gate_fraction": "Curriculum/terrain_levels/exposure_gate_pass_fraction",
  "command_linear_score": "Curriculum/command_vel/linear_score",
  "command_linear_score_ema": "Curriculum/command_vel/linear_score_ema",
  "command_angular_score": "Curriculum/command_vel/angular_score",
  "command_progress_ratio": "Curriculum/command_vel/progress_ratio",
  "command_progress_threshold": "Curriculum/command_vel/min_progress_ratio",
  "command_progress_gate": "Curriculum/command_vel/progress_gate_pass",
  "command_score_gate": "Curriculum/command_vel/score_gate_pass",
  "command_episode_fraction": "Curriculum/command_vel/episode_fraction",
  "command_failure_fraction": "Curriculum/command_vel/failure_fraction",
  "command_promotion_allowed": "Curriculum/command_vel/promotion_allowed",
  "command_vx_min": "Curriculum/command_vel/curriculum_vx_min",
  "command_vx_max": "Curriculum/command_vel/curriculum_vx_max",
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


def read_terrain_curriculum_status(run_directory):
  path = Path(run_directory) / "curriculum" / "terrain_sampling.json"
  try:
    if path.stat().st_size > 1024 * 1024:
      return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("version") != 1
        or data.get("mode") not in {"lp_acrl", "uniform"}
        or not isinstance(data.get("tasks"), list) or len(data["tasks"]) > 2000):
      return None
    for task in data["tasks"]:
      if not isinstance(task.get("name"), str) or not isinstance(task.get("terrain"), str):
        return None
      probability = float(task["probability"])
      if not math.isfinite(probability) or not 0 <= probability <= 1:
        return None
      for name in ("learning_progress", "mean_return", "failure_fraction", "progress_ratio"):
        if task.get(name) is not None and not math.isfinite(float(task[name])):
          return None
    return data
  except (OSError, ValueError, KeyError, TypeError, AttributeError):
    return None


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
      for name, candidates in ACTION_METRIC_TAGS.items():
        candidates = (candidates,) if isinstance(candidates, str) else candidates
        tag = next((tag for tag in candidates if tag in available_tags), None)
        if tag is None:
          continue
        points = accumulator.Scalars(tag)
        if points:
          point = points[-1]
          metrics[name] = {"value": float(point.value), "step": int(point.step), "tag": tag}
      passage = {}
      prefix = "Curriculum/passage/success_rate/"
      for tag in sorted(available_tags):
        if not tag.startswith(prefix):
          continue
        kind = tag.removeprefix(prefix)
        rate = accumulator.Scalars(tag)
        count_tag = "Curriculum/passage/samples/" + kind
        counts = accumulator.Scalars(count_tag) if count_tag in available_tags else []
        if rate and counts:
          passage[kind] = {"rate": float(rate[-1].value), "samples": float(counts[-1].value)}
      raw_max = metrics.get("raw_action_abs_max", {}).get("value")
      saturation = metrics.get("action_saturation_fraction", {}).get("value")
      policy_std = metrics.get("policy_mean_std", {}).get("value")
      # Raw output clipping is an advisory signal.  It is not motor or torque
      # saturation: V19 is stable in sim2sim while its normalized policy output
      # still crosses the local ±1 execution clip.  Keep a hard warning only
      # for a clearly divergent policy distribution.
      advisories = []
      warnings = []
      if raw_max is not None and raw_max > 3.0:
        advisories.append("策略原始输出超过 3（裁剪前观察项）")
      if saturation is not None and saturation > 0.05:
        advisories.append("策略裁剪比例超过 5%（不等于力矩饱和）")
      if policy_std is not None and policy_std > 3.0:
        warnings.append("策略标准差异常，可能发生分布发散")
      terrain_promote = metrics.get("terrain_promote_fraction", {}).get("value")
      terrain_demote = metrics.get("terrain_demote_fraction", {}).get("value")
      if terrain_promote is not None and terrain_demote is not None and terrain_demote > terrain_promote * 1.5:
        advisories.append("地形降级明显多于升级")
      status = "warning" if warnings else "observe" if advisories else "healthy"
      data = {"task": alias, "experiment": task["experiment"], "available": bool(metrics),
              "run": event_file.parent.name, "updated": stat.st_mtime, "metrics": metrics, "passage": passage,
              "status": status,
              "warning": "；".join(warnings) if warnings else None,
              "advisory": "；".join(advisories) if advisories else None,
              "note": "原始动作指标统计的是策略输出裁剪，不等于电机力矩饱和；请结合 sim2sim 和实际姿态判断。"}
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
    terrain_status = read_terrain_curriculum_status(event_file.parent)
    with metrics_lock:
      cached = metrics_cache.get(alias)
      if cached and cached["signature"] == signature:
        if cached.get("loading"):
          return {"task": alias, "experiment": task["experiment"], "available": False,
                  "run": event_file.parent.name, "updated": stat.st_mtime,
                  "metrics": {}, "warning": "正在读取最新 TensorBoard 记录…",
                  "terrain_curriculum": terrain_status}
        return {**cached["data"], "terrain_curriculum": terrain_status}
      metrics_cache[alias] = {"signature": signature, "loading": True}
    threading.Thread(target=parse_metrics, args=(alias, task, event_file, stat, signature), daemon=True).start()
    return {"task": alias, "experiment": task["experiment"], "available": False,
            "run": event_file.parent.name, "updated": stat.st_mtime,
            "metrics": {}, "warning": "正在读取最新 TensorBoard 记录…",
            "terrain_curriculum": terrain_status}

  ctx = routes.ApiContext(root=root, python=Path(python), token=token,
                          jobs=jobs, gpu=gpu_info, metrics=latest_metrics)

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
        static = STATIC_FILES.get(parsed.path)
        if static is not None:
          filename, mime = static
          return self.send((WEB / filename).read_bytes(), content_type=mime)
        status, payload = routes.dispatch_get(ctx, parsed.path, query)
        return self.send(payload, status)
      except (ValueError, OSError) as error:
        return self.send({"error": str(error)}, 400)

    def do_POST(self):
      # Same-origin token blocks cross-site process launches from unrelated tabs.
      supplied = self.headers.get("X-LgDog-Token", "")
      origin = self.headers.get("Origin")
      expected_origin = f"http://{self.headers.get('Host', '')}"
      if (not self.valid_host() or not hmac.compare_digest(supplied, token)
          or (origin and origin != expected_origin)):
        return self.send({"error": "请刷新本地面板后再操作"}, 403)
      try:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 16384:
          raise ValueError("请求大小无效")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
          raise ValueError("请求必须为对象")
        status, payload = routes.dispatch_post(ctx, self.path, data)
        return self.send(payload, status)
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


