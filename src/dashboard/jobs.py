"""Own only launched subprocesses; write stdout directly to durable log files."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import uuid

from .commands import display_command

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class JobManager:
  def __init__(self, root):
    self.root = Path(root).resolve()
    self.directory = self.root / "logs/dashboard/jobs"
    self.directory.mkdir(parents=True, exist_ok=True)
    self.lock = threading.RLock()
    self.jobs = {}
    self.history = []
    for path in sorted(self.directory.glob("*.json"), reverse=True)[:20]:
      try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(data, dict) and {"created", "mode", "pid", "task", "status"} <= data.keys()
            and re.fullmatch(r"[0-9a-f]{12}", data.get("id", ""))):
          data["can_stop"] = False
          if data.get("status") in {"running", "stopping"}:
            data["status"] = "unknown"
          self.history.append(data)
      except (ValueError, OSError, TypeError):
        continue

  def _save(self, record):
    path = self.directory / f"{record['id']}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

  def _refresh(self):
    for record, process in self.jobs.values():
      code = process.poll()
      if code is not None and record["status"] in {"running", "stopping"}:
        stopped = record["status"] == "stopping"
        record.update(status="stopped" if stopped else ("completed" if code == 0 else "failed"),
                      returncode=code, can_stop=False)
        self._save(record)

  def launch(self, argv, mode, task):
    with self.lock:
      self._refresh()
      active = [r for r, _ in self.jobs.values() if r["status"] in {"running", "stopping"}]
      if len(active) >= 4:
        raise ValueError("最多同时运行 4 个操作，请先停止或等待已有任务")
      if any(r["mode"] == mode for r in active):
        raise ValueError("同类操作已在运行，请先停止或等待完成")
      job_id = uuid.uuid4().hex[:12]
      log_path = self.directory / f"{job_id}.log"
      record = {"id": job_id, "mode": mode, "task": task, "command": display_command(argv),
                "created": datetime.now(timezone.utc).isoformat(), "status": "running",
                "returncode": None, "can_stop": True, "log": str(log_path.relative_to(self.root))}
      env = os.environ.copy()
      env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", NO_COLOR="1")
      env["PYTHONPATH"] = str(self.root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
      kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
      with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {record['command']}\n\n")
        stream.flush()
        process = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, shell=False, **kwargs)
      record["pid"] = process.pid
      self.jobs[job_id] = (record, process)
      self._save(record)
      return dict(record)

  def list(self):
    with self.lock:
      self._refresh()
      return sorted([dict(r) for r, _ in self.jobs.values()] + self.history,
                    key=lambda item: item["created"], reverse=True)[:30]

  def tail(self, job_id):
    known = {r["id"] for r in self.list()}
    if job_id not in known:
      raise ValueError("没有找到该运行记录")
    path = self.directory / f"{job_id}.log"
    if not path.exists():
      return "日志文件已不存在"
    with path.open("rb") as stream:
      stream.seek(0, 2)
      offset = max(0, stream.tell() - 65536)
      stream.seek(offset)
      data = stream.read().decode("utf-8", errors="replace")
    if offset and "\n" in data:
      data = data.split("\n", 1)[1]
    return ANSI.sub("", data)

  def stop(self, job_id):
    with self.lock:
      self._refresh()
      if job_id not in self.jobs:
        raise ValueError("只能停止本次面板启动的进程；历史记录不能直接接管")
      record, process = self.jobs[job_id]
      if process.poll() is not None:
        return dict(record)
      record["status"] = "stopping"
      self._save(record)
      if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
      else:
        try:
          os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
          pass
      try:
        process.wait(timeout=3)
      except subprocess.TimeoutExpired:
        if os.name == "nt":
          process.kill()
        else:
          os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
      self._refresh()
      return dict(record)
