"""把三个本地服务（工作台 / 曲线终端 / TensorBoard）当子进程管起来。

设计原则和训练工作台那边一致：
  * 只认**自己启动的**子进程，别处起的同端口服务只显示「已在运行」，不给停止按钮；
  * 子进程的 stdout/stderr 直接落到 ``logs/web/<服务>.log``，页面按需读尾巴；
  * 父进程退出不杀子进程（和 ``lgdog.py ui`` 的行为一致），下次启动按端口识别。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SERVICE_DEFS = (
  {
    "name": "dashboard",
    "label": "训练工作台",
    "port": 8765,
    "path": "/",
    "description": "选任务、起训练 / 回放 / Sim2Sim、改奖励、看真实输出和审计记录",
  },
  {
    "name": "kline",
    "label": "曲线终端",
    "port": 8766,
    "path": "/",
    "description": "TensorBoard 曲线按训练步合成 K 线：分时、均线、波动柱、盘口",
  },
  {
    "name": "tensorboard",
    "label": "TensorBoard",
    "port": 6006,
    "path": "/",
    "description": "官方曲线页，按 tag 分组的原始视图，用来和 K 线对拍",
  },
)


def port_open(port, host="127.0.0.1", timeout=0.25):
  try:
    with socket.create_connection((host, port), timeout=timeout):
      return True
  except OSError:
    return False


SIGNATURE = {
  "dashboard": "lgdog.py",
  "kline": "src.kline",
  "tensorboard": "tensorboard",
}


def _listener_pid(port):
  """谁在监听这个端口（psutil 可用时）。"""
  try:
    import psutil
    for connection in psutil.net_connections(kind="inet"):
      if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == port:
        return connection.pid
  except Exception:
    return None
  return None


def _looks_like(pid, name):
  """别误杀：只动命令行里带本服务特征串的进程。"""
  try:
    import psutil
    cmdline = " ".join(psutil.Process(pid).cmdline())
    return SIGNATURE.get(name, name) in cmdline
  except Exception:
    return False


class Services:
  def __init__(self, root, python, logdirs, ports=None):
    self.root = Path(root)
    self.python = str(python)
    self.logdirs = [Path(path) for path in logdirs]
    self.log_dir = self.root / "logs" / "web"
    self.ports = dict(ports or {})
    self.children = {}

  # ---- 查询 ---------------------------------------------------------------
  def definitions(self):
    return [dict(item, port=self.ports.get(item["name"], item["port"])) for item in SERVICE_DEFS]

  def describe(self, name):
    definition = next((item for item in self.definitions() if item["name"] == name), None)
    if definition is None:
      raise KeyError(f"没有这个服务：{name}")
    port = definition["port"]
    child = self.children.get(name)
    meta = {
      "name": name,
      "label": definition["label"],
      "port": port,
      "url": f"http://127.0.0.1:{port}{definition['path']}",
      "description": definition["description"],
      "log": str(self.log_dir / f"{name}.log"),
      "managed": False,
      "pid": None,
      "status": "stopped",
      "since": None,
      "returncode": None,
    }
    if child is not None:
      code = child.get("process").poll()
      if code is None:
        meta.update(managed=True, pid=child.get("process").pid, since=child.get("since"),
                    status="running" if port_open(port) else "starting")
        return meta
      meta.update(returncode=code, status="failed" if code != 0 else "stopped")
      return meta
    if port_open(port):
      meta.update(status="external")
    return meta

  def list(self):
    return [self.describe(item["name"]) for item in self.definitions()]

  # ---- 操作 ---------------------------------------------------------------
  def argv(self, name):
    definition = next((item for item in self.definitions() if item["name"] == name), None)
    if definition is None:
      raise ValueError(f"没有这个服务：{name}")
    port = str(definition["port"])
    if name == "dashboard":
      return [self.python, "lgdog.py", "--python", self.python, "ui", "--port", port, "--no-browser"]
    if name == "kline":
      argv = [self.python, "-m", "src.kline", "--port", port, "--no-browser"]
      for path in self.logdirs:
        argv.extend(["--logdir", str(path)])
      return argv
    if name == "tensorboard":
      logdir = str(self.logdirs[0]) if self.logdirs else str(self.root / "logs" / "rsl_rl")
      return [self.python, "-m", "tensorboard.main", "--logdir", logdir,
              "--host", "127.0.0.1", "--port", port, "--reload_multifile=true"]
    raise ValueError(f"没有这个服务：{name}")

  def start(self, name):
    meta = self.describe(name)
    if meta["status"] in {"running", "starting"}:
      return meta
    if meta["status"] == "external":
      raise ValueError(f"{meta['label']} 已经在 {meta['port']} 端口运行（不是这里启动的）")
    self.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = self.log_dir / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(self.root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    kwargs = {"cwd": str(self.root), "env": env, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
      kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
      kwargs["start_new_session"] = True
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
      log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 {meta['label']} (:{meta['port']}) =====\n")
      log.flush()
      process = subprocess.Popen(self.argv(name), stdout=log, stderr=subprocess.STDOUT, **kwargs)
    self.children[name] = {"process": process, "since": time.time()}
    return self.describe(name)

  def wait_ready(self, name, timeout=12.0):
    """等端口起来；起来就返回 running，超时返回当前状态（页面会继续轮询）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
      meta = self.describe(name)
      if meta["status"] in {"running", "failed"}:
        return meta
      time.sleep(0.25)
    return self.describe(name)

  def stop(self, name):
    meta = self.describe(name)
    child = self.children.get(name)
    if child is None or meta["status"] == "external":
      raise ValueError(f"{meta['label']} 不是这里启动的，请在它自己的窗口里停止")
    process = child["process"]
    if process.poll() is None:
      process.terminate()
      try:
        process.wait(timeout=6)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    # Windows 的 venv python 有时是个「壳」，terminate 壳之后真正的解释器还在跑；
    # 端口还开着就按监听者 PID 补一刀（只认命令行里带本服务特征的进程）。
    if port_open(meta["port"]):
      pid = _listener_pid(meta["port"])
      if pid and _looks_like(pid, name):
        try:
          import psutil
          psutil.Process(pid).kill()
          time.sleep(0.5)
        except Exception:
          pass
    self.children.pop(name, None)
    return self.describe(name)

  def tail(self, name, lines=80):
    meta = self.describe(name)
    path = Path(meta["log"])
    if not path.is_file():
      return {"name": name, "text": "", "size": 0, "exists": False}
    from .system import tail_lines
    return {"name": name, "text": "\n".join(tail_lines(path, lines)), "size": path.stat().st_size,
            "exists": True, "path": str(path)}

  def shutdown(self):
    """面板退出时不动子进程（和历史行为一致）；这里只清引用。"""
    self.children.clear()
