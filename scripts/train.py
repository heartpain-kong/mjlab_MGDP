"""Script to train RL agent with RSL-RL."""

import logging
import faulthandler
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

if __name__ == "__main__":
  faulthandler.enable(all_threads=True)
  print("[init] importing training libraries...", flush=True)

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from src.tasks.velocity.rl.action_monitor import ActionMonitorVecEnvWrapper
from src.tasks.velocity.mdp.curriculums import (
  make_velocity_stages,
  velocity_ranges_from_bounds,
)
from src.dashboard.run_naming import resolve_run_name
from src.scene_diagnostics import report_scene_initialization
from src.tasks.velocity.config.lgdog1.terrain_curriculum import configure_terrain_curriculum


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  terrain_num_rows: int | None = None
  terrain_num_cols: int | None = None
  terrain_difficulty_max: float | None = None
  terrain_curriculum: Literal["manual", "uniform", "lp_acrl"] = "manual"
  lp_temperature: float = 2.0
  lp_uniform_mix: float = 0.5
  lp_min_episodes: int = 32
  lp_update_interval_s: float = 100.0
  lp_progress_alpha: float = 0.25
  lp_probability_alpha: float = 0.2
  lp_max_task_probability: float = 0.25
  velocity_min_abs: float | None = None
  velocity_max_abs: float | None = None
  resume_checkpoint: str | None = None
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  terrain = cfg.env.scene.terrain
  if any(value is not None for value in (
    cfg.terrain_num_rows, cfg.terrain_num_cols, cfg.terrain_difficulty_max)):
    if terrain is None or terrain.terrain_generator is None:
      raise ValueError("地形扩展参数只适用于 generator 地形任务")
    generator = terrain.terrain_generator
    if cfg.terrain_num_rows is not None:
      if cfg.terrain_num_rows < 2:
        raise ValueError("--terrain-num-rows 必须大于等于 2")
      generator.num_rows = int(cfg.terrain_num_rows)
      print(f"[INFO] Terrain curriculum rows: {generator.num_rows}")
    if cfg.terrain_num_cols is not None:
      minimum = len(generator.sub_terrains) if "passage" in cfg.env.curriculum else 1
      if cfg.terrain_num_cols < minimum:
        raise ValueError(f"--terrain-num-cols 必须大于等于 {minimum}，保留所有障碍类型")
      generator.num_cols = int(cfg.terrain_num_cols)
      print(f"[INFO] Terrain curriculum columns: {generator.num_cols}")
    if cfg.terrain_difficulty_max is not None:
      if not 0.5 <= cfg.terrain_difficulty_max <= 1.5:
        raise ValueError("--terrain-difficulty-max 必须在 0.5 到 1.5 之间")
      lower, _ = generator.difficulty_range
      generator.difficulty_range = (lower, float(cfg.terrain_difficulty_max))
      print(f"[INFO] Terrain difficulty range: {generator.difficulty_range}")

  if cfg.terrain_curriculum != "manual" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
    raise ValueError("当前自动地形课程仅支持单GPU；多GPU训练请使用 manual")
  configure_terrain_curriculum(
    cfg.env, cfg.terrain_curriculum, temperature=cfg.lp_temperature,
    uniform_mix=cfg.lp_uniform_mix, min_episodes=cfg.lp_min_episodes,
    update_interval_s=cfg.lp_update_interval_s,
    progress_alpha=cfg.lp_progress_alpha, probability_alpha=cfg.lp_probability_alpha,
    max_task_probability=cfg.lp_max_task_probability,
    status_path=log_dir / "curriculum" / "terrain_sampling.json",
  )

  if "passage" in cfg.env.curriculum and (cfg.velocity_min_abs is not None or cfg.velocity_max_abs is not None):
    raise ValueError("定向越障速度由通过率门控管理，不接受正负速度包络覆盖")
  if cfg.velocity_min_abs is not None or cfg.velocity_max_abs is not None:
    min_abs = 0.4 if cfg.velocity_min_abs is None else float(cfg.velocity_min_abs)
    max_abs = 0.8 if cfg.velocity_max_abs is None else float(cfg.velocity_max_abs)
    final_ranges = velocity_ranges_from_bounds(min_abs, max_abs)
    initial_ranges = velocity_ranges_from_bounds(min_abs * 0.5, max_abs * 0.5)
    twist = cfg.env.commands.get("twist")
    if twist is None:
      raise ValueError("当前任务没有 twist 速度指令，不能设置速度课程")
    command_curriculum = cfg.env.curriculum.get("command_vel")
    if command_curriculum is None:
      # Legacy PPO/terrain profiles without a command curriculum still honor
      # the workbench values as a fixed final command range.
      twist.ranges.lin_vel_x = final_ranges["lin_vel_x"]
      twist.ranges.lin_vel_y = final_ranges["lin_vel_y"]
      twist.ranges.ang_vel_z = final_ranges["ang_vel_z"]
      print("[INFO] Task has no command curriculum; using the final speed range as fixed limits")
    else:
      twist.ranges.lin_vel_x = initial_ranges["lin_vel_x"]
      twist.ranges.lin_vel_y = initial_ranges["lin_vel_y"]
      twist.ranges.ang_vel_z = final_ranges["ang_vel_z"]
      if "velocity_stages" in command_curriculum.params:
        command_curriculum.params["velocity_stages"] = make_velocity_stages(min_abs, max_abs)
      else:
        command_curriculum.params["lin_vel_x_limit"] = final_ranges["lin_vel_x"]
        command_curriculum.params["lin_vel_y_limit"] = final_ranges["lin_vel_y"]
    print(
      "[INFO] Velocity curriculum: "
      f"min_abs={min_abs:g}, max_abs={max_abs:g}, "
      f"final_x={final_ranges['lin_vel_x']}, "
      f"final_y={final_ranges['lin_vel_y']}, "
      f"final_yaw={final_ranges['ang_vel_z']}"
    )

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  grid = cfg.env.scene.terrain.terrain_generator if cfg.env.scene.terrain else None
  if grid is not None:
    print(f"[init] building terrain {grid.num_rows}x{grid.num_cols}, "
          f"envs={cfg.env.scene.num_envs}...", flush=True)
  with report_scene_initialization(cfg.env):
    env = ManagerBasedRlEnv(
      cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )
  print("[init] physics environment ready; creating PPO runner...", flush=True)

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load from the current experiment, or from an explicitly compatible
      # stage such as Flat-HIM -> Rough-HIM.
      if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
          raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
      else:
        resume_path = get_checkpoint_path(
          log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  # Penalize only the part of the policy output that lies outside the
  # executable normalized action range.  This prevents PPO from hiding a
  # diverging Gaussian behind the environment clip.
  env = ActionMonitorVecEnvWrapper(
    env, clip_actions=cfg.agent.clip_actions, raw_action_penalty_coef=1.0e-4
  )

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = (Path("logs") / "rsl_rl" / args.agent.experiment_name).resolve()
  args.agent.run_name = resolve_run_name(log_root_path, args.agent.run_name, args.agent.seed)
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
