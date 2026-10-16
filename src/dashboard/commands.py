"""Validated argv shared by the browser and the short CLI. Never uses a shell."""

from pathlib import Path
import math
import re
import shlex
import subprocess
import os

from .catalog import resolve_task


def discover_models(root, task_value="standard"):
  _, task = resolve_task(task_value)
  base = Path(root) / "logs/rsl_rl" / task["experiment"]
  if not base.is_dir():
    return []
  result = []
  for run in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
    files = [p for p in run.iterdir() if p.is_file() and (re.fullmatch(r"model_\d+\.pt", p.name) or p.name == "policy.onnx")]
    files.sort(key=lambda p: int(p.stem.split("_")[-1]) if p.suffix == ".pt" else -1, reverse=True)
    for path in files:
      result.append({"path": str(path.relative_to(root)), "name": path.name,
                     "run": run.name, "kind": "checkpoint" if path.suffix == ".pt" else "onnx",
                     "bytes": path.stat().st_size})
      if len(result) >= 500:
        return result
  return result


def integer(value, name, low, high):
  try:
    parsed = int(str(value))
  except (ValueError, TypeError):
    raise ValueError(f"{name} 必须为整数") from None
  if not low <= parsed <= high:
    raise ValueError(f"{name} 需在 {low}–{high} 之间")
  return parsed


def number(value, name, low, high):
  try:
    parsed = float(value)
  except (ValueError, TypeError):
    raise ValueError(f"{name} 必须为数字") from None
  if not math.isfinite(parsed) or not low <= parsed <= high:
    raise ValueError(f"{name} 需在 {low}–{high} 之间")
  return parsed


def model_path(root, task_value, value, kind):
  _, task = resolve_task(task_value)
  if not value:
    if task.get("shared"):
      raise ValueError("Flat/Rough 共用实验目录，请手动选择与任务匹配的模型")
    candidates = [m for m in discover_models(root, task_value) if m["kind"] == kind]
    if not candidates:
      raise ValueError(f"该任务尚无{ ' checkpoint' if kind == 'checkpoint' else ' ONNX' }，请先训练或指定模型文件")
    value = candidates[0]["path"]
  path = Path(value).expanduser()
  path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
  suffix = ".pt" if kind == "checkpoint" else ".onnx"
  if not path.is_file() or path.suffix != suffix:
    raise ValueError(f"模型文件不存在或类型不匹配（需要 {suffix}）：{path}")
  return path


def build_command(root, python, options):
  root = Path(root).resolve()
  mode = options.get("mode", "train")
  task_value = options.get("task", "standard")
  _, task = resolve_task(task_value)
  scripts = root / "scripts"
  if mode == "train":
    envs = integer(options.get("envs", 1024), "环境数", 1, 8192)
    iterations = integer(options.get("iterations", 16000), "训练轮数", 1, 1000000)
    seed = integer(options.get("seed", 42), "seed", 0, 2147483647)
    name = str(options.get("run_name", "")).strip()
    if name and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", name):
      raise ValueError("实验名只能使用 1–80 位英文字母、数字、下划线、点和短横线")
    argv = [str(python), str(scripts / "train.py"), task["id"],
            f"--env.scene.num-envs={envs}", f"--agent.max-iterations={iterations}",
            "--agent.logger=tensorboard", f"--agent.seed={seed}", "--enable-nan-guard", "True"]
    if name:
      argv.append(f"--agent.run-name={name}")
    if options.get("resume"):
      checkpoint = model_path(root, task_value, options.get("checkpoint"), "checkpoint")
      expected = (root / "logs/rsl_rl" / task["experiment"]).resolve()
      if checkpoint.parent.parent != expected:
        raise ValueError("续训请选择当前任务实验目录内的 checkpoint，不能跨任务直接恢复")
      argv += ["--agent.resume=True", f"--agent.load-run={re.escape(checkpoint.parent.name)}$",
               f"--agent.load-checkpoint={re.escape(checkpoint.name)}$"]
    return argv
  if mode == "play":
    checkpoint = model_path(root, task_value, options.get("checkpoint"), "checkpoint")
    viewer = options.get("viewer", "viser")
    if viewer not in {"viser", "native"}:
      raise ValueError("play viewer 只能是 viser 或 native")
    return [str(python), str(scripts / "play.py"), task["id"], f"--checkpoint_file={checkpoint}",
            f"--num_envs={integer(options.get('envs', 4), '环境数', 1, 128)}", "--viewer", viewer]
  if mode == "sim":
    path = model_path(root, task_value, options.get("onnx"), "onnx")
    map_name = options.get("map", "flat")
    if map_name not in {"flat", "stairs", "slope", "obstacles", "mixed", "lgdog1"}:
      raise ValueError("未知 sim2sim 测试地图")
    argv = [str(python), str(scripts / "sim2sim.py"), task["id"], f"--onnx={path}",
            "--map", map_name, "--vx", str(number(options.get("vx", 0.4), "vx", -3, 3)),
            "--vy", str(number(options.get("vy", 0), "vy", -3, 3)),
            "--wz", str(number(options.get("wz", 0), "wz", -3, 3)),
            "--duration", str(number(options.get("duration", 20), "时长", 0.02, 3600)),
            "--spawn-y", str(number(options.get("spawn_y", 0), "出生 y", -30, 30))]
    if options.get("show_viewer", True):
      argv.append("--viewer")
    if not options.get("keyboard", False):
      argv.append("--no-keyboard")
    return argv
  if mode == "rewards":
    argv = [str(python), str(scripts / "list_rewards.py"), task["id"]]
    term = str(options.get("term", "")).strip()
    if term:
      if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", term):
        raise ValueError("奖励名格式错误")
      argv += ["--term", term]
    return argv
  if mode == "curves":
    port = integer(options.get("port", 6006), "曲线端口", 1024, 65535)
    return [str(python), "-m", "tensorboard.main", "--logdir", str(root / "logs/rsl_rl"),
            "--host", "127.0.0.1", "--port", str(port)]
  raise ValueError(f"未知操作 {mode!r}")


def display_command(argv):
  return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
