"""HTTP API routes for the workbench.

All handlers are pure functions of an :class:`ApiContext` plus the request
query/body.  Response shapes and status codes are unchanged from the original
``server.py`` implementation; ``server.py`` now only does HTTP plumbing
(loopback check, token/origin check, JSON encoding, static files).

Adding an endpoint means adding one function and one table entry - no more
``if path == ...`` chains inside the request handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Any, Callable
from src.config.mgdp_pretrain import pretrain_resources

from . import history
from .catalog import (
  ALGORITHMS,
  DEFAULT_TASK,
  SCENES,
  SIM2SIM_MAPS,
  TASKS,
  resolve_task,
  task_payload,
)
from .commands import build_command, discover_models, display_command, integer
from .reward_editor import read_rewards, reward_detail, save_rewards
from .run_naming import next_run_name
from .terrain_editor import read_terrain, save_terrain

# Document links shown in the panel.  Kept here so the router owns the payload
# shape and the server stays a thin transport.
DOCUMENTS = {
  "start": ("使用说明", "doc/WORKBENCH.md"),
  "rewards": ("奖励调参", "doc/REWARDS.md"),
  "mgdp": ("MGDP 深度感知", "doc/MGDP_MJLAB.md"),
  "standard": ("标准地形 HIM", "doc/LGDOG_HIM_STANDARD.md"),
  "reward-code": ("奖励配置源码", "src/tasks/velocity/config/lgdog1/reward_cfg.py"),
}


@dataclass
class ApiContext:
  root: Path
  python: Path
  token: str
  jobs: Any
  gpu: Callable[[], dict]
  metrics: Callable[[str], dict]


# --------------------------------------------------------------------------- #
# GET
# --------------------------------------------------------------------------- #
def _task_state(ctx: ApiContext, alias: str) -> dict:
  """Task payload plus live prerequisite state (checkpoint on disk)."""
  payload = task_payload(alias)
  cap = payload["capabilities"]
  checkpoint = cap.get("pretrain_checkpoint") or ""
  if cap.get("needs_pretrain"):
    cap["pretrain_ready"] = bool(checkpoint) and (ctx.root / checkpoint).is_file()
  else:
    cap["pretrain_ready"] = True
  return payload


def state(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, {
    "token": ctx.token, "root": str(ctx.root), "python": str(ctx.python),
    "tasks": [_task_state(ctx, alias) for alias in TASKS],
    "algorithms": [{"alias": alias, **item} for alias, item in ALGORITHMS.items()],
    "scenes": [{"alias": alias, **item} for alias, item in SCENES.items()],
    "sim2sim_maps": [{"alias": alias, **item} for alias, item in SIM2SIM_MAPS.items()],
    "gpu": ctx.gpu(), "jobs": ctx.jobs.list(), "job_counts": ctx.jobs.counts(),
    "documents": [{"id": key, "label": value[0]} for key, value in DOCUMENTS.items()],
  }


def gpu(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, ctx.gpu()


def run_name(ctx: ApiContext, query) -> tuple[int, dict]:
  alias, task = resolve_task(query.get("task", [DEFAULT_TASK])[0])
  seed = integer(query.get("seed", [42])[0], "seed", 0, 2147483647)
  return 200, {"task": alias, "seed": seed,
               "name": next_run_name(ctx.root / "logs/rsl_rl" / task["experiment"], seed)}


def models(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, {"models": discover_models(ctx.root, query.get("task", [DEFAULT_TASK])[0])}


def metrics(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, ctx.metrics(query.get("task", [DEFAULT_TASK])[0])


def rewards(ctx: ApiContext, query) -> tuple[int, dict]:
  alias, task = resolve_task(query.get("task", [DEFAULT_TASK])[0])
  return 200, {"task": alias, **read_rewards(ctx.root, task["profile"])}


def terrain(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, read_terrain(ctx.root)


def reward_detail(ctx: ApiContext, query) -> tuple[int, dict]:
  alias, task = resolve_task(query.get("task", [DEFAULT_TASK])[0])
  return 200, {"task": alias, **reward_detail(ctx.root, task["profile"], query.get("name", [""])[0])}


def log(ctx: ApiContext, query) -> tuple[int, dict]:
  return 200, {"text": ctx.jobs.tail(query.get("id", [""])[0])}


def history_runs(ctx: ApiContext, query) -> tuple[int, dict]:
  ctx.jobs.list()  # Refresh exits before reading the lifecycle events.
  result = history.read_history(ctx.root, query.get("limit", [history.DEFAULT_LIMIT])[0])
  with ctx.jobs.lock:
    for run in result["runs"]:
      pair = ctx.jobs.jobs.get(run["id"])
      if pair:
        record = pair[0]
        run.update(status=record["status"], can_delete=record.get("can_delete", False))
      else:
        run["can_delete"] = True  # History-only record, no managed process.
  return 200, result


def document(ctx: ApiContext, query) -> tuple[int, dict]:
  entry = DOCUMENTS.get(query.get("id", [""])[0])
  if not entry:
    raise ValueError("未知文档")
  path = ctx.root / entry[1]
  return 200, {"title": entry[0], "path": str(path),
               "text": path.read_text(encoding="utf-8-sig")}


# --------------------------------------------------------------------------- #
# POST
# --------------------------------------------------------------------------- #
def preview(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  argv = build_command(ctx.root, ctx.python, data)
  return 200, {"command": display_command(argv), "argv": argv}


def run(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  argv = build_command(ctx.root, ctx.python, data)
  alias, _ = resolve_task(data.get("task", DEFAULT_TASK))
  return 200, {"job": ctx.jobs.launch(argv, data.get("mode", "train"), alias)}


def pretrain(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  """Launch MGDP perception pretraining from the workbench."""
  alias, task = resolve_task(data.get("task","mgdp"))
  if not task.get("requires_pretrain"):
    raise ValueError("该任务不使用 MGDP 感知预训练")
  script = ctx.root / "scripts" / "pretrain_mgdp.py"
  if not script.is_file():
    raise ValueError("找不到 scripts/pretrain_mgdp.py")
  defaults = pretrain_resources(task["id"])
  envs = integer(data.get("envs", defaults.num_envs), "环境数", 1, 4096)
  rows = integer(data.get("terrain_rows", defaults.terrain_rows), "地形行数", 1, 100)
  batch = integer(data.get("batch_size", defaults.batch_size), "训练批大小", 1, 4096)
  steps = integer(data.get("steps", 4000), "采集步数", 10, 200000)
  iterations = integer(data.get("iterations", 3000), "训练迭代", 1, 200000)
  output = ctx.root / task["pretrain_checkpoint"]
  argv = [
    str(ctx.python), str(script),
    "--task", task["id"],
    "--num-envs", str(envs),
    "--steps", str(steps),
    "--iterations", str(iterations),
    "--terrain-rows", str(rows),
    "--batch-size", str(batch),
  ]
  cols = data.get("terrain_cols", defaults.terrain_cols)
  if cols is not None:
    cols = integer(cols, "地形列数", defaults.terrain_cols or 1, 100)
    argv.extend(["--terrain-cols", str(cols)])
  argv.extend(["--out", str(output)])
  return 200, {"job": ctx.jobs.launch(argv, "pretrain", alias), "output": str(output)}


def reward_save(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  alias, task = resolve_task(data.get("task", DEFAULT_TASK))
  before = {item["name"]: {"weight": item["weight"], "params": item["params"]}
            for item in read_rewards(ctx.root, task["profile"])["rewards"]}
  result = save_rewards(ctx.root, task["profile"], data.get("changes"))
  after = {item["name"]: {"weight": item["weight"], "params": item["params"]}
           for item in result.get("rewards") or []}
  history.record_reward_edit(ctx.root, task=alias, profile=task["profile"], before=before,
                             after=after, source=result.get("source"),
                             backup=result.get("backup"))
  return 200, {"task": alias, **result}


def terrain_save(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  return 200, save_terrain(ctx.root, data)


def terrain_view(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  alias, task = resolve_task(data.get("task", DEFAULT_TASK))
  script = ctx.root / "scripts" / "visualize_terrain.py"
  if not script.is_file():
    raise ValueError("找不到 scripts/visualize_terrain.py")
  # A previous task's viewer may still own8080; return this launch's port.
  with socket.socket() as listener:
    listener.bind(("127.0.0.1",0))
    port=listener.getsockname()[1]
  job = ctx.jobs.launch([str(ctx.python),str(script),"--port",str(port),"--task",task["id"]],"terrain",alias)
  return 200, {"job":job,"url":f"http://127.0.0.1:{port}"}


def stop(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  return 200, {"job": ctx.jobs.stop(str(data.get("id", "")), force=data.get("force") is True)}


def jobs_delete(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  return 200, ctx.jobs.delete(str(data.get("id", "")))


def jobs_clear(ctx: ApiContext, data: dict) -> tuple[int, dict]:
  return 200, ctx.jobs.clear_finished()


GET_ROUTES = {
  "/api/state": state,
  "/api/gpu": gpu,
  "/api/run-name": run_name,
  "/api/models": models,
  "/api/metrics": metrics,
  "/api/rewards": rewards,
  "/api/terrain": terrain,
  "/api/reward-detail": reward_detail,
  "/api/log": log,
  "/api/history": history_runs,
  "/api/document": document,
}

POST_ROUTES = {
  "/api/preview": preview,
  "/api/run": run,
  "/api/rewards/save": reward_save,
  "/api/terrain/save": terrain_save,
  "/api/terrain/view": terrain_view,
  "/api/pretrain": pretrain,
  "/api/stop": stop,
  "/api/jobs/delete": jobs_delete,
  "/api/jobs/clear-finished": jobs_clear,
}


def dispatch_get(ctx: ApiContext, path: str, query) -> tuple[int, dict]:
  handler = GET_ROUTES.get(path)
  if handler is None:
    return 404, {"error": "页面不存在"}
  return handler(ctx, query)


def dispatch_post(ctx: ApiContext, path: str, data: dict) -> tuple[int, dict]:
  handler = POST_ROUTES.get(path)
  if handler is None:
    return 404, {"error": "操作不存在"}
  return handler(ctx, data)


