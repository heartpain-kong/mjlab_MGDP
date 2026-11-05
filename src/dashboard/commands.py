"""Validated argv shared by the browser and the short CLI. Never uses a shell."""

from pathlib import Path
import math
import re
import shlex
import subprocess
import os

from .catalog import DEFAULT_TASK, SIM2SIM_MAPS, TASKS, resolve_task
from .run_naming import next_run_name
from src.config.race_training import MIN_RACE_TERRAIN_COLS


def discover_models(root, task_value=DEFAULT_TASK):
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


def camera_inset_flags(options, tyro=False):
  size = integer(options.get("camera_inset_size", 256), "相机小窗尺寸", 64, 512)
  fps = number(options.get("camera_inset_fps", 10), "相机小窗帧率", 1, 60)
  corner = options.get("camera_inset_corner", "bottom-left")
  if corner not in {"bottom-left", "bottom-right", "top-left", "top-right"}:
    raise ValueError("未知相机小窗位置")
  enabled = options.get("camera_inset", True) is not False
  toggle = [f"--camera-inset={enabled}"] if tyro else ([] if enabled else ["--no-camera-inset"])
  return toggle + [
    f"--camera-inset-size={size}", f"--camera-inset-fps={fps}", f"--camera-inset-corner={corner}"]


def build_command(root, python, options):
  root = Path(root).resolve()
  mode = options.get("mode", "train")
  task_value = options.get("task", DEFAULT_TASK)
  _, task = resolve_task(task_value)
  if mode == "train" and task.get("requires_pretrain") and not options.get("resume"):
    # Capability-driven prerequisite: fail before launching a doomed run and
    # tell the user the exact command instead of crashing inside env setup.
    checkpoint = root / str(task.get("pretrain_checkpoint") or "")
    if not checkpoint.is_file():
      raise ValueError(task.pretrain_hint())
  scripts = root / "scripts"
  if mode == "train":
    defaults = task.get("train_presets", {}).get("full", {})
    requested_envs = options.get("envs")
    envs = integer(defaults.get("envs",1024) if requested_envs is None else requested_envs,
                   "环境数", 1, 8192)
    iterations = integer(options.get("iterations", 16000), "训练轮数", 1, 1000000)
    seed = integer(options.get("seed", 42), "seed", 0, 2147483647)
    mini_batches = options.get("ppo_mini_batches")
    if mini_batches is not None and str(mini_batches).strip():
      mini_batches = integer(mini_batches, "PPO小批次数", 1, 64)
    terrain_rows = str(options.get("terrain_num_rows") or defaults.get("terrain_num_rows", "")).strip()
    if terrain_rows:
      terrain_rows = integer(terrain_rows, "地形训练行数", 2, 100)
    raw_cols = options.get("terrain_num_cols")
    terrain_cols = str(defaults.get("terrain_num_cols", "")
                       if raw_cols is None or str(raw_cols).strip()=="" else raw_cols).strip()
    if terrain_cols:
      terrain_cols = integer(terrain_cols, "地形训练列数",
                             MIN_RACE_TERRAIN_COLS if task.get("directed_passage") else 1, 100)
    terrain_difficulty = str(options.get("terrain_difficulty_max") or "").strip()
    if terrain_difficulty:
      terrain_difficulty = number(terrain_difficulty, "最大地形难度倍率", 0.5, 1.5)
    velocity_min = str(options.get("velocity_min_abs") or "").strip()
    if velocity_min:
      velocity_min = number(velocity_min, "最低速度绝对值", 0.05, 3.0)
    velocity_max = str(options.get("velocity_max_abs") or "").strip()
    if velocity_max:
      velocity_max = number(velocity_max, "最高速度绝对值", 0.05, 3.0)
    if velocity_min and velocity_max and velocity_min > velocity_max:
      raise ValueError("最低速度绝对值不能大于最高速度绝对值")
    curriculum = options.get("terrain_curriculum", "manual")
    if curriculum not in {"manual", "uniform", "lp_acrl"}:
      raise ValueError("未知地形课程；可选 manual、uniform、lp_acrl")
    if task.get("directed_passage") and (curriculum != "manual" or velocity_min or velocity_max):
      raise ValueError("定向越障使用通过率门控；请保留 manual，速度由任务低速课程管理")
    curriculum_flags = []
    if curriculum != "manual":
      if task.get("scene") == "flat":
        raise ValueError("平地任务不支持自动地形课程，请选择崎岖地形")
      temperature = number(options.get("lp_temperature", 2.0), "LP温度", 0.000001, 10000)
      mix = number(options.get("lp_uniform_mix", 0.5), "LP基础地形比例混合量", 0, 1)
      minimum = integer(options.get("lp_min_episodes", 32), "LP最少回合数", 1, 100000)
      interval = number(options.get("lp_update_interval_s", 100), "LP更新间隔", 0.02, 3600)
      progress_alpha = number(options.get("lp_progress_alpha", 0.25), "LP进度平滑系数", 0.000001, 1)
      probability_alpha = number(options.get("lp_probability_alpha", 0.2), "LP概率更新系数", 0.000001, 1)
      max_probability = number(options.get("lp_max_task_probability", 0.25), "LP单任务概率上限", 0.000001, 1)
      curriculum_flags = [f"--terrain-curriculum={curriculum}", f"--lp-temperature={temperature}",
                          f"--lp-uniform-mix={mix}", f"--lp-min-episodes={minimum}",
                          f"--lp-update-interval-s={interval}", f"--lp-progress-alpha={progress_alpha}",
                          f"--lp-probability-alpha={probability_alpha}",
                          f"--lp-max-task-probability={max_probability}"]
    name = str(options.get("run_name", "")).strip()
    if name and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", name):
      raise ValueError("实验名只能使用 1–80 位英文字母、数字、下划线、点和短横线")
    argv = [str(python), str(scripts / "train.py"), task["id"],
            f"--env.scene.num-envs={envs}", f"--agent.max-iterations={iterations}",
            "--agent.logger=tensorboard", f"--agent.seed={seed}", "--enable-nan-guard", "True"]
    argv += curriculum_flags
    if options.get("optimize_vram") is False:
      argv.append("--optimize-vram=False")
    if mini_batches is not None and str(mini_batches).strip():
      argv.append(f"--agent.algorithm.num-mini-batches={mini_batches}")
    if terrain_rows != "":
      argv.append(f"--terrain-num-rows={terrain_rows}")
    if terrain_cols != "":
      argv.append(f"--terrain-num-cols={terrain_cols}")
    if terrain_difficulty != "":
      argv.append(f"--terrain-difficulty-max={terrain_difficulty}")
    if velocity_min != "":
      argv.append(f"--velocity-min-abs={velocity_min}")
    if velocity_max != "":
      argv.append(f"--velocity-max-abs={velocity_max}")
    auto_name = next_run_name(root / "logs/rsl_rl" / task["experiment"], seed)
    argv.append(f"--agent.run-name={name or auto_name}")
    if options.get("resume"):
      checkpoint = model_path(root, task_value, options.get("checkpoint"), "checkpoint", include_compatible=True)
      expected = (root / "logs/rsl_rl" / task["experiment"]).resolve()
      compatible = {expected}
      for source_alias in task.get("resume_sources", ()):
        compatible.add((root / "logs/rsl_rl" / TASKS[source_alias]["experiment"]).resolve())
      if checkpoint.parent.parent not in compatible:
        raise ValueError("续训只能选择当前算法和场景的 checkpoint；平地不能续训崎岖或障碍场景")
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
            f"--num_envs={integer(options.get('envs', 4), '环境数', 1, 128)}", "--viewer", viewer] + (
              ["--optimize-vram=False"] if options.get("optimize_vram") is False else []) + camera_inset_flags(options, tyro=True)
  if mode == "sim":
    path = model_path(root, task_value, options.get("onnx"), "onnx")
    map_name = options.get("map", "flat")
    if map_name not in SIM2SIM_MAPS:
      raise ValueError(f"未知 sim2sim 测试地图：{map_name}")
    argv = [str(python), str(scripts / "sim2sim.py"), task["id"], f"--onnx={path}",
            "--map", map_name, "--vx", str(number(options.get("vx", 0.4), "vx", -3, 3)),
            "--vy", str(number(options.get("vy", 0), "vy", -3, 3)),
            "--wz", str(number(options.get("wz", 0), "wz", -3, 3)),
            "--spawn-y", str(number(options.get("spawn_y", 0), "出生 y", -30, 30))]
    if options.get("duration") is not None:
      argv += ["--duration", str(number(options["duration"], "时长", 0.02, 3600))]
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
    return argv + camera_inset_flags(options)
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

