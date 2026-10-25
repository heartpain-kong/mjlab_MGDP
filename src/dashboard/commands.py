"""Validated argv shared by the browser and the short CLI. Never uses a shell."""

from pathlib import Path
import math
import re
import shlex
import subprocess
import os

from .catalog import TASKS, resolve_task
from .run_naming import next_run_name


def discover_models(root, task_value="standard"):
  alias, task = resolve_task(task_value)
  aliases = [alias, *task.get("resume_sources", ())]
  result = []
  for source_alias in aliases:
    source_task = TASKS[source_alias]
    base = Path(root) / "logs/rsl_rl" / source_task["experiment"]
    if not base.is_dir():
      continue
    for run in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
      files = [p for p in run.iterdir() if p.is_file() and (re.fullmatch(r"model_\d+\.pt", p.name) or p.name == "policy.onnx")]
      files.sort(key=lambda p: int(p.stem.split("_")[-1]) if p.suffix == ".pt" else -1, reverse=True)
      for path in files:
        result.append({"path": str(path.relative_to(root)), "name": path.name,
                       "run": run.name, "kind": "checkpoint" if path.suffix == ".pt" else "onnx",
                       "source": source_alias, "source_label": source_task["label"],
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


def model_path(root, task_value, value, kind, *, include_compatible=False):
  alias, task = resolve_task(task_value)
  if not value:
    if task.get("shared"):
      raise ValueError("Flat/Rough 共用实验目录，请手动选择与任务匹配的模型")
    candidates = [m for m in discover_models(root, task_value)
                  if m["kind"] == kind and (include_compatible or m.get("source") == alias)]
    if not candidates:
      raise ValueError(f"该任务尚无{ ' checkpoint' if kind == 'checkpoint' else ' ONNX' }，请先训练或指定模型文件")
    value = candidates[0]["path"]
  path = Path(value).expanduser()
  path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
  suffix = ".pt" if kind == "checkpoint" else ".onnx"
  if not path.is_file() or path.suffix != suffix:
    raise ValueError(f"模型文件不存在或类型不匹配（需要 {suffix}）：{path}")
  return path


def terrain_xml_path(root, value):
  """Resolve an imported sim2sim terrain XML without invoking a shell."""
  if not value:
    return None
  path = Path(value).expanduser()
  path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
  if not path.is_file() or path.suffix.lower() != ".xml":
    raise ValueError(f"地形 XML 不存在或类型不匹配：{path}")
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
    terrain_rows = str(options.get("terrain_num_rows", "")).strip()
    if terrain_rows:
      terrain_rows = integer(terrain_rows, "地形训练行数", 2, 100)
    terrain_difficulty = str(options.get("terrain_difficulty_max", "")).strip()
    if terrain_difficulty:
      terrain_difficulty = number(terrain_difficulty, "最大地形难度倍率", 0.5, 1.5)
    name = str(options.get("run_name", "")).strip()
    if name and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", name):
      raise ValueError("实验名只能使用 1–80 位英文字母、数字、下划线、点和短横线")
    argv = [str(python), str(scripts / "train.py"), task["id"],
            f"--env.scene.num-envs={envs}", f"--agent.max-iterations={iterations}",
            "--agent.logger=tensorboard", f"--agent.seed={seed}", "--enable-nan-guard", "True"]
    if terrain_rows != "":
      argv.append(f"--terrain-num-rows={terrain_rows}")
    if terrain_difficulty != "":
      argv.append(f"--terrain-difficulty-max={terrain_difficulty}")
    auto_name = next_run_name(root / "logs/rsl_rl" / task["experiment"], seed)
    argv.append(f"--agent.run-name={name or auto_name}")
    if options.get("resume"):
      checkpoint = model_path(root, task_value, options.get("checkpoint"), "checkpoint", include_compatible=True)
      expected = (root / "logs/rsl_rl" / task["experiment"]).resolve()
      compatible = {expected}
      for source_alias in task.get("resume_sources", ()):
        compatible.add((root / "logs/rsl_rl" / TASKS[source_alias]["experiment"]).resolve())
      if checkpoint.parent.parent not in compatible:
        raise ValueError("续训请选择当前任务或兼容阶段的 checkpoint")
      argv.append("--agent.resume=True")
      if checkpoint.parent.parent == expected:
        argv += [f"--agent.load-run={re.escape(checkpoint.parent.name)}$",
                 f"--agent.load-checkpoint={re.escape(checkpoint.name)}$"]
      else:
        argv.append(f"--resume-checkpoint={checkpoint}")
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
    argv += ["--spawn-x", str(number(options.get("spawn_x", 0), "出生 x", -30, 30)),
             "--spawn-yaw", str(number(options.get("spawn_yaw", 0), "出生朝向", -180, 180)),
             "--speed-x", str(number(options.get("speed_x", 1.0), "键盘前进速度", 0.05, 3)),
             "--speed-y", str(number(options.get("speed_y", 0.5), "键盘横移速度", 0.05, 3)),
             "--speed-yaw", str(number(options.get("speed_yaw", 1.0), "键盘转向速度", 0.05, 3)),
             "--action-clip", str(number(options.get("action_clip", 1.0), "动作裁剪上限", 0.05, 3))]
    imported_map = terrain_xml_path(root, str(options.get("map_file", "")).strip())
    if imported_map is not None:
      argv.append(f"--map-file={imported_map}")
    if options.get("check_obs", False):
      argv.append("--check-obs")
    if options.get("heading_hold", True):
      argv.append("--heading-hold")
    else:
      argv.append("--no-heading-hold")
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
