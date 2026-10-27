"""Own only launched subprocesses; write stdout directly to durable log files."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from .commands import display_command
from .history import delete_run_records, read_history
from .process_control import legacy_identity, process_info, process_tree, same_process, terminate_tree

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JOB_ID = re.compile(r"[0-9a-f]{12}")
ACTIVE = {"running", "stopping"}
FINISHED = {"completed", "failed", "stopped", "finished"}


class RestoredProcess:
  """A restart can verify identity, but cannot recover the original exit code."""

  def __init__(self, identity):
    self.identity = identity
    self.pid = identity["pid"]

  def poll(self):
    return None if same_process(self.identity) else 0


class JobManager:
  def __init__(self, root, on_event=None):
    self.root = Path(root).resolve()
    self.directory = self.root / "logs/dashboard/jobs"
    self.directory.mkdir(parents=True, exist_ok=True)
    self.lock = threading.RLock()
    self.jobs = {}
    # 记录回调：on_event("start" | "end", record)。只做旁路记录，异常不影响进程管理。
    self.on_event = on_event
    for path in self.directory.glob("*.json"):
      try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(data, dict) and {"created", "mode", "pid", "task", "status"} <= data.keys()
            and isinstance(data.get("id"), str) and JOB_ID.fullmatch(data["id"])
            and path.stem == data["id"]):
          process = None
          if data["status"] in ACTIVE | {"unknown"}:
            identity = data.get("process_identity")
            try:
              if not identity:
                identity = legacy_identity(data, self.root)
              if identity and same_process(identity):
                data.update(process_identity=identity, status="running", recovered=True,
                            stop_reason=None)
                process = RestoredProcess(identity)
              elif identity:
                if any(same_process(item) for item in data.get("stop_targets", [])):
                  data.update(status="running", recovered=True, stop_reason=None)
                  process = RestoredProcess(identity)
                else:
                  data.update(status="finished", stop_reason="原进程已退出或 PID 已被复用")
              else:
                data.update(status="unknown", stop_reason="旧记录缺少可核验的进程身份；可删除记录，不能直接停止")
            except OSError as error:
              data.update(status="unknown", stop_reason=f"无法核验历史进程：{error}")
          data["can_stop"] = process is not None
          data["can_delete"] = process is None
          self.jobs[data["id"]] = (data, process)
          if process is not None:
            self._save(data)  # Persist identity when a legacy task is adopted.
      except (ValueError, OSError, TypeError):
        continue

  def _save(self, record):
    path = self.directory / f"{record['id']}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

  def _emit(self, event, record):
    """Hand a lifecycle event to the history recorder; never break job control."""
    callback = self.on_event
    if callback is None:
      return
    try:
      callback(event, dict(record))
    except Exception:  # noqa: BLE001 - 记录失败不能影响启停
      pass

  def _refresh(self):
    for record, process in self.jobs.values():
      # A stop request owns the entire tree until verification finishes.  An
      # exited wrapper alone must not make the UI report successful shutdown.
      if process is None or record["status"] == "stopping":
        continue
      code = process.poll()
      if code is not None and record["status"] == "running":
        # Captured survivors from an unsuccessful stop remain controllable.
        if any(same_process(item) for item in record.get("stop_targets", [])):
          continue
        restored = isinstance(process, RestoredProcess)
        record.update(status="finished" if restored else ("completed" if code == 0 else "failed"),
                      returncode=None if restored else code, can_stop=False, can_delete=True,
                      stop_reason="进程已退出")
        self._save(record)
        self._emit("end", record)

  def launch(self, argv, mode, task):
    with self.lock:
      self._refresh()
      active = [r for r, _ in self.jobs.values() if r["status"] in ACTIVE]
      if len(active) >= 4:
        raise ValueError("最多同时运行 4 个操作，请先停止或等待已有任务")
      if any(r["mode"] == mode for r in active):
        raise ValueError("同类操作已在运行，请先停止或等待完成")
      job_id = uuid.uuid4().hex[:12]
      log_path = self.directory / f"{job_id}.log"
      record = {"id": job_id, "mode": mode, "task": task, "command": display_command(argv),
                "argv": [str(item) for item in argv], "started_ts": time.time(),
                "created": datetime.now(timezone.utc).isoformat(), "status": "running",
                "returncode": None, "can_stop": True, "can_delete": False,
                "stop_reason": None, "log": str(log_path.relative_to(self.root))}
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
      record["process_identity"] = process_info(process.pid)
      self.jobs[job_id] = (record, process)
      self._save(record)
      self._emit("start", record)
      return dict(record)

  def list(self):
    with self.lock:
      self._refresh()
      return sorted([dict(r) for r, _ in self.jobs.values()],
                    key=lambda item: item["created"], reverse=True)[:30]

  def counts(self):
    """Count every loaded job, including records outside the visible list."""
    with self.lock:
      self._refresh()
      records = [record for record, _ in self.jobs.values()]
      return {
        "total": len(records),
        "active": sum(record["status"] in ACTIVE for record in records),
        "clearable": sum(record["status"] in FINISHED and record.get("can_delete", False)
                         for record in records),
      }

  def tail(self, job_id):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id) or job_id not in self.jobs:
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

  def stop(self, job_id, force=False):
    with self.lock:
      self._refresh()
      if job_id not in self.jobs:
        raise ValueError("没有找到该运行记录，请刷新面板")
      record, process = self.jobs[job_id]
      if record["status"] in FINISHED:
        return dict(record)
      if record["status"] == "stopping":
        return dict(record)
      if process is None:
        raise ValueError(record.get("stop_reason") or "历史进程身份无法核验")
      identity = record.get("process_identity")
      if not identity:
        # A freshly launched process may finish before its identity is read.
        raise ValueError("无法核验进程身份，请刷新后重试")
      tracked = process_tree(identity)
      tracked.extend(item for item in record.get("stop_targets", []) if same_process(item))
      record.update(status="stopping", can_stop=False, can_delete=False, stop_error=None,
                    stop_targets=tracked)
      self._save(record)
    # Waiting for a driver/child process must not block state/log requests.
    try:
      terminate_tree(identity, tracked, force=force)
      if isinstance(process, subprocess.Popen):
        process.wait(timeout=1)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
      with self.lock:
        record.update(status="running", can_stop=True, can_delete=False, stop_error=str(error))
        self._save(record)
      raise ValueError(f"停止失败：{error}") from error
    with self.lock:
      record.update(status="stopped", returncode=None if isinstance(process, RestoredProcess) else process.poll(), can_stop=False, can_delete=True,
                    stop_error=None, stop_reason="进程及已核验的子进程均已退出")
      self._save(record)
      self._emit("end", record)
      return dict(record)

  def delete(self, job_id):
    """Delete a job's panel record/log only; never touch training artifacts."""
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
      raise ValueError("运行记录 ID 无效")
    with self.lock:
      self._refresh()
      pair = self.jobs.get(job_id)
      if pair is not None and not pair[0].get("can_delete"):
        raise ValueError("任务仍在运行或正在停止，请先停止任务再删除记录")
      if pair is None and not any(run["id"] == job_id for run in read_history(self.root, 1000)["runs"]):
        raise ValueError("没有找到该运行记录")
      # Paths are derived from a validated ID, not the path stored in JSON.
      for suffix in (".log", ".json"):
        path = self.directory / f"{job_id}{suffix}"
        path.unlink(missing_ok=True)
      delete_run_records(self.root, {job_id})
      self.jobs.pop(job_id, None)
      return {"deleted": [job_id]}

  def clear_finished(self):
    with self.lock:
      self._refresh()
      selected = {job_id for job_id, (record, _) in self.jobs.items()
                  if record["status"] in FINISHED and record.get("can_delete")}
      selected.update(run["id"] for run in read_history(self.root, 1000)["runs"]
                      if run.get("status") in FINISHED and run["id"] not in self.jobs)
      deleted = []
      for job_id in sorted(selected):
        deleted.extend(self.delete(job_id)["deleted"])
      return {"deleted": deleted}
