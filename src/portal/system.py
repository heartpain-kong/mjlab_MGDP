"""本机状态采集：环境版本、GPU、CPU/内存/磁盘、相关进程、日志。

全部只读；缺什么就如实报什么（这台机器没有 nvidia-smi 就报「没有 NVIDIA 驱动」，
不会假装有 GPU）。除 psutil 是可选的以外只用标准库。
"""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PACKAGES = ("torch", "mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "tensorboard", "numpy", "psutil", "wandb")

# 命令行里出现这些片段，就认为这条进程和本工程有关
ROLE_HINTS = (
  ("scripts/train.py", "训练", "train"),
  ("scripts/play.py", "回放", "play"),
  ("scripts/sim2sim.py", "Sim2Sim", "sim"),
  ("src.kline", "曲线终端", "kline"),
  ("src.portal", "总览", "portal"),
  ("tensorboard", "TensorBoard", "tensorboard"),
  ("viser", "Viser 视图", "viser"),
)

_PROCESS_CACHE = {}
_GPU_CACHE = {"at": 0.0, "data": None}


def _role_of(cmdline):
  """认角色。``lgdog.py`` 要再看子命令：web 是总览，ui 是训练工作台，别混成一个。"""
  for hint, label, key in ROLE_HINTS:
    if hint in cmdline:
      return label, key
  if "lgdog.py" in cmdline:
    if " web " in cmdline + " ":
      return "总览", "portal"
    if " ui " in cmdline + " ":
      return "训练工作台", "dashboard"
    return "lgdog 命令", "cli"
  return None


def _version(name):
  try:
    return metadata.version(name)
  except Exception:
    return None


def environment(root, python):
  versions = {name: _version(name) for name in PACKAGES}
  torch_version = versions.get("torch") or ""
  return {
    "python": sys.version.split()[0],
    "executable": str(python),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "hostname": socket.gethostname(),
    "cpu_count": os.cpu_count(),
    "packages": versions,
    "torch_cpu_only": "+cpu" in torch_version.lower(),
    "cwd": str(root),
  }


def resources(root):
  data = {"cpu_count": os.cpu_count(), "load": None, "memory": None, "disk": None}
  try:
    data["load"] = list(os.getloadavg())  # Linux / macOS；Windows 没有
  except (AttributeError, OSError):
    data["load"] = None
  try:
    usage = shutil.disk_usage(str(root))
    data["disk"] = {
      "total": usage.total, "used": usage.total - usage.free, "free": usage.free,
      "percent": round((usage.total - usage.free) / usage.total * 100, 1) if usage.total else None,
    }
  except OSError:
    pass
  try:
    import psutil
    memory = psutil.virtual_memory()
    data["memory"] = {"total": memory.total, "used": memory.total - memory.available,
                      "percent": memory.percent}
    data["cpu_percent"] = psutil.cpu_percent(interval=None)
  except Exception:
    data["memory"] = None
  return data


def _nvidia_smi():
  candidates = [shutil.which("nvidia-smi")]
  if os.name == "nt":
    candidates.extend([
      r"C:\Windows\System32\nvidia-smi.exe",
      r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ])
  return next((path for path in candidates if path and Path(path).is_file()), None)


def gpu(max_age=3.0):
  """GPU 遥测。优先 nvidia-smi；没有 NVIDIA 驱动时退回 WMI 只报型号。"""
  if _GPU_CACHE["data"] is not None and time.time() - _GPU_CACHE["at"] < max_age:
    return _GPU_CACHE["data"]
  data = _gpu_snapshot()
  _GPU_CACHE.update(at=time.time(), data=data)
  return data


def _gpu_snapshot():
  executable = _nvidia_smi()
  if not executable:
    return {
      "available": False, "gpus": [], "source": "wmi" if os.name == "nt" else None,
      "reason": "未找到 nvidia-smi：这台机器没有 NVIDIA 驱动，跑不了 CUDA 训练",
      "adapters": _wmi_adapters() if os.name == "nt" else [],
    }
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
    return {"available": False, "gpus": [], "reason": f"nvidia-smi 读取失败：{type(error).__name__}", "adapters": []}
  if result.returncode != 0:
    reason = (result.stderr or "nvidia-smi 返回错误").strip().splitlines()[0]
    return {"available": False, "gpus": [], "reason": reason[:160], "adapters": []}
  gpus = []
  for row in csv.reader(result.stdout.splitlines()):
    if len(row) < 8:
      continue
    def number(value, integer=False):
      try:
        parsed = float(value.strip())
      except ValueError:
        return None
      return int(parsed) if integer else parsed
    gpus.append({
      "index": number(row[0], True), "name": row[1].strip(),
      "memory_total_mib": number(row[2], True), "memory_used_mib": number(row[3], True),
      "memory_free_mib": number(row[4], True), "utilization_gpu": number(row[5], True),
      "temperature_c": number(row[6], True), "power_w": number(row[7]),
    })
  return {"available": bool(gpus), "gpus": gpus, "source": executable,
          "reason": None if gpus else "nvidia-smi 没有返回 GPU 数据", "adapters": []}


def _wmi_adapters():
  """没有 nvidia-smi 时至少把显卡型号读出来（Windows 专用，失败就返回空）。"""
  try:
    result = subprocess.run(
      ["powershell", "-NoProfile", "-NonInteractive", "-Command",
       "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"],
      capture_output=True, text=True, timeout=4, check=False,
      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    text = (result.stdout or "").strip()
    if not text:
      return []
    import json
    parsed = json.loads(text)
    rows = parsed if isinstance(parsed, list) else [parsed]
    return [{"name": row.get("Name"), "driver": row.get("DriverVersion")} for row in rows if row.get("Name")]
  except Exception:
    return []


def processes(root, exclude=()):
  """和本工程有关的进程：训练 / 回放 / Sim2Sim / 面板 / TensorBoard。

  ``exclude`` 用来把总览服务自己（以及 Windows 上那个同命令行的 python 壳）排掉。
  """
  try:
    import psutil
  except Exception:
    return {"available": False, "items": [], "reason": "当前 Python 没有 psutil，读不到进程列表"}
  exclude = set(exclude or ())
  root_text = str(root).lower()
  now = time.time()
  items = []
  for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "memory_info", "cpu_times"]):
    if proc.info["pid"] in exclude:
      continue
    try:
      argv = proc.info.get("cmdline") or []
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      continue
    if not argv:
      continue
    # 只认真正的 Python 进程：任务包装器（node / pwsh）的命令行里也会有 lgdog.py，
    # 但它们的可执行文件不是 python，不能算「训练进程」。
    executable = Path(str(argv[0])).name.lower()
    if "python" not in (proc.info.get("name") or "").lower() and "python" not in executable:
      continue
    cmdline = " ".join(argv)
    role = _role_of(cmdline)
    if role is None:
      # 只按命令行认角色；宁可漏掉，也不把编辑器 / 终端之类混进「训练进程」。
      continue
    try:
      memory = proc.info.get("memory_info")
      cpu_times = proc.info.get("cpu_times")
      total_cpu = (cpu_times.user + cpu_times.system) if cpu_times else 0.0
      previous = _PROCESS_CACHE.get(proc.info["pid"])
      _PROCESS_CACHE[proc.info["pid"]] = (total_cpu, now)
      cpu_percent = None
      if previous and now - previous[1] > 0.2:
        cpu_percent = round((total_cpu - previous[0]) / (now - previous[1]) * 100, 1)
      items.append({
        "pid": proc.info["pid"],
        "ppid": proc.ppid(),
        "name": proc.info.get("name"),
        "role": role[0],
        "role_key": role[1],
        "cpu_percent": cpu_percent,
        "memory_mb": round(memory.rss / 1024 / 1024, 1) if memory else None,
        "uptime": round(now - (proc.info.get("create_time") or now), 0),
        "command": cmdline[:400],
        "args": " ".join(argv[1:])[:400],
        "own": root_text in cmdline.lower(),
      })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      continue
  # Windows 某些 venv 的 python.exe 是个「壳」：壳和真正的解释器只差 argv[0]，
  # 两行都列出来只会让人以为跑了两份。参数一样就把壳隐掉。
  by_pid = {item["pid"]: item for item in items}
  for item in items:
    parent = by_pid.get(item.get("ppid"))
    if parent and parent["args"] == item["args"]:
      parent["shim"] = True
  items = [item for item in items if not item.get("shim")]
  for item in items:
    item.pop("ppid", None)
    item.pop("args", None)
  items.sort(key=lambda item: item["uptime"], reverse=True)
  return {"available": True, "items": items, "reason": None}


def gpu_processes():
  """nvidia-smi 里正在用显存的进程（有的话）。"""
  executable = _nvidia_smi()
  if not executable:
    return []
  kwargs = {"capture_output": True, "text": True, "timeout": 2, "check": False}
  if os.name == "nt":
    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
  try:
    result = subprocess.run(
      [executable, "--query-compute-apps=pid,process_name,used_memory",
       "--format=csv,noheader,nounits"], **kwargs)
  except (OSError, subprocess.SubprocessError):
    return []
  rows = []
  for row in csv.reader((result.stdout or "").splitlines()):
    if len(row) < 3:
      continue
    try:
      rows.append({"pid": int(row[0]), "name": row[1].strip(), "memory_mib": int(float(row[2]))})
    except ValueError:
      continue
  return rows


def recent_logs(root, limit=6):
  """logs/ 下最近的 .log（训练标准输出、面板日志都在这），带末尾几行。"""
  base = Path(root) / "logs"
  if not base.is_dir():
    return []
  candidates = []
  for path in base.rglob("*.log"):
    try:
      stat = path.stat()
    except OSError:
      continue
    candidates.append((stat.st_mtime, stat.st_size, path))
  candidates.sort(reverse=True)
  return [{
    "path": str(path), "name": str(path.relative_to(root)),
    "size": size, "mtime": mtime,
    "tail": tail_lines(path, 12),
  } for mtime, size, path in candidates[:limit]]


def tail_lines(path, count=60, max_bytes=96 * 1024):
  """读文件尾部，不把大日志整个读进内存。"""
  path = Path(path)
  try:
    size = path.stat().st_size
    with path.open("rb") as handle:
      handle.seek(max(0, size - max_bytes))
      data = handle.read()
  except OSError:
    return []
  text = data.decode("utf-8", errors="replace")
  lines = text.splitlines()
  if len(data) >= max_bytes and lines:
    lines = lines[1:]  # 第一行可能被截断
  return lines[-count:]
