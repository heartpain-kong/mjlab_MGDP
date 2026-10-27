"""Durable records for reward edits and training runs.

每文件一行 JSON，写在 `logs/dashboard/history/` 下：

- `reward_edits.jsonl` —— 面板「保存奖励修改」的审计记录（谁把哪个 reward 从多少改成多少）
- `train_runs.jsonl`     —— 面板启动的每个操作的开始/结束事件

追加写、单行 JSON，坏了某一行不会影响其它记录。读取时只取尾部若干行，
所以文件长期增长也不会拖慢面板。
"""

from datetime import datetime
import json
from pathlib import Path
import re
import threading

DIRECTORY = "logs/dashboard/history"
REWARD_LOG = "reward_edits.jsonl"
RUN_LOG = "train_runs.jsonl"
DEFAULT_LIMIT = 120
MAX_LIMIT = 1000

_LOCK = threading.Lock()

# 从 argv 里捡出训练常用参数，方便事后重建一次 run 的意图。
_ARG_PATTERNS = {
  "run_name": re.compile(r"^--agent[._]run-name=(.*)$"),
  "envs": re.compile(r"^--env[._]scene[._]num-envs=(\d+)$"),
  "iterations": re.compile(r"^--agent[._]max-iterations=(\d+)$"),
  "seed": re.compile(r"^--agent[._]seed=(\d+)$"),
  "checkpoint": re.compile(r"^--checkpoint_file=(.*)$"),
  "onnx": re.compile(r"^--onnx=(.*)$"),
  "map": re.compile(r"^--map=?(.*)$"),
  "map_file": re.compile(r"^--map-file=(.*)$"),
}


def _directory(root):
  return Path(root).resolve() / DIRECTORY


def _append(root, name, record):
  """追加一行；失败时返回 None，绝不向上抛（记录不该影响面板操作）。"""
  directory = _directory(root)
  try:
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **record},
                      ensure_ascii=False)
  except (OSError, TypeError, ValueError):
    return None
  with _LOCK:
    try:
      with (directory / name).open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    except OSError:
      return None
  return record


def _tail(root, name, limit):
  path = _directory(root) / name
  try:
    with _LOCK:
      lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  except OSError:
    return []
  records = []
  for line in lines[-limit:]:
    line = line.strip()
    if not line:
      continue
    try:
      record = json.loads(line)
    except ValueError:
      continue
    if isinstance(record, dict):
      records.append(record)
  return records


def summarize_argv(argv):
  """Pick training-relevant flags out of an argv list."""
  summary = {}
  for item in argv or []:
    for key, pattern in _ARG_PATTERNS.items():
      match = pattern.match(str(item))
      if match:
        value = match.group(1)
        if key == "envs" or key == "iterations" or key == "seed":
          try:
            value = int(value)
          except ValueError:
            continue
        if value not in ("", None):
          summary[key] = value
        break
  return summary


def record_reward_edit(root, *, task, profile, before, after, source, backup=None):
  """Store one reward save.

  ``before`` / ``after`` 都是 ``{reward 名: {"weight": float, "params": dict}}``，
  取值来自 ``reward_editor.read_rewards``。只有真正变化的 reward 才写入记录。
  """
  changes = []
  for name in sorted(set(before) | set(after)):
    old = before.get(name) or {}
    new = after.get(name) or {}
    old_weight = old.get("weight")
    new_weight = new.get("weight")
    old_params = old.get("params") or {}
    new_params = new.get("params") or {}
    if old_weight == new_weight and old_params == new_params:
      continue
    changes.append({"name": name,
                    "weight": [old_weight, new_weight],
                    "params": [old_params, new_params]})
  if not changes:
    return None
  return _append(root, REWARD_LOG, {"kind": "reward_edit", "task": task, "profile": profile,
                                    "changes": changes, "source": str(source),
                                    "backup": str(backup) if backup else None})


def record_run_event(root, event, record):
  """Store one job lifecycle event. ``event`` is ``start`` or ``end``."""
  summary = summarize_argv(record.get("argv") or [])
  payload = {"kind": "run_" + event, "id": record.get("id"), "mode": record.get("mode"),
             "task": record.get("task"), "pid": record.get("pid"),
             "command": record.get("command"), "log": record.get("log")}
  if event == "end":
    payload.update(status=record.get("status"), returncode=record.get("returncode"))
    started = record.get("started_ts")
    if isinstance(started, (int, float)):
      payload["duration_s"] = round(datetime.now().timestamp() - started, 1)
  else:
    payload.update(summary)
  return _append(root, RUN_LOG, payload)


def delete_run_records(root, job_ids):
  """Remove only the specified run events, preserving reward edits and files.

  Hold the same lock used by append so a concurrently finishing task cannot
  lose its lifecycle event during the atomic rewrite.
  """
  job_ids = set(job_ids)
  if not job_ids:
    return
  path = _directory(root) / RUN_LOG
  with _LOCK:
    if not path.is_file():
      return
    keep = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
      try:
        item = json.loads(line)
      except ValueError:
        keep.append(line)
        continue
      if not isinstance(item, dict) or item.get("id") not in job_ids:
        keep.append(line)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    temp.replace(path)


def read_history(root, limit=DEFAULT_LIMIT):
  """Return ``{"reward_edits": [...], "runs": [...]}``, newest first."""
  try:
    limit = max(1, min(int(limit), MAX_LIMIT))
  except (TypeError, ValueError):
    limit = DEFAULT_LIMIT
  edits = _tail(root, REWARD_LOG, limit)
  events = _tail(root, RUN_LOG, limit * 2)
  runs, order = {}, []
  for event in events:
    job_id = event.get("id")
    if not job_id:
      continue
    if event.get("kind") == "run_start":
      runs[job_id] = {"id": job_id, "mode": event.get("mode"), "task": event.get("task"),
                      "command": event.get("command"), "log": event.get("log"),
                      "started": event.get("time"), "ended": None, "status": "unknown",
                      "returncode": None, "duration_s": None}
      for key in ("run_name", "envs", "iterations", "seed", "checkpoint", "onnx", "map"):
        if key in event:
          runs[job_id][key] = event[key]
      order.append(job_id)
    elif event.get("kind") == "run_end":
      item = runs.setdefault(job_id, {"id": job_id, "mode": event.get("mode"),
                                      "task": event.get("task"), "command": event.get("command"),
                                      "log": event.get("log"), "started": None})
      if job_id not in order:
        order.append(job_id)
      item.update(ended=event.get("time"), status=event.get("status"),
                  returncode=event.get("returncode"), duration_s=event.get("duration_s"))
  runs = [runs[job_id] for job_id in reversed(order)][:limit]
  edits.reverse()
  return {"reward_edits": edits, "runs": runs}
