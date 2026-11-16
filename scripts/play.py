"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViserPlayViewer

from src.tasks.velocity.rl.action_monitor import ActionMonitorVecEnvWrapper
from src.him_contract import apply_him_play_contract


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  """显式指定 checkpoint。不给则自动取【最新 run 的最新 model_*.pt】（方便训练中随时看）。"""
  perception_checkpoint: str | None = None
  """旧MGDP checkpoint缺少配套快照时，显式指定原训练感知文件。"""
  load_run: str = ".*"
  """只在不给 checkpoint_file 时生效：run 目录名的正则（按名字排序取最后一个）。"""
  load_checkpoint: str = "model_.*.pt"
  """只在不给 checkpoint_file 时生效：checkpoint 文件名的正则。"""
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  camera_inset: bool = True
  camera_inset_size: int = 256
  camera_inset_fps: float = 10.0
  camera_inset_corner: Literal["bottom-left", "bottom-right", "top-left", "top-right"] = "bottom-left"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  optimize_vram: bool = True

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      # 原本这里检查 cfg.registry_name（已随 W&B 逻辑一并删除，引用会 AttributeError）。
      raise ValueError(
        "跟踪类任务用 dummy agent 时必须给本地动作文件：\n"
        "  --motion-file /path/to/motion.npz\n"
        "（本项目已移除 W&B registry 下载逻辑，--registry-name 不可用）"
      )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      # ★ 不给 --checkpoint_file 时，自动取【最新 run 的最新 checkpoint】。
      #   这正是「训练过程中随时 play 看当前策略」需要的 —— 训练每 save_interval
      #   存一个 model_N.pt，重跑一次 play 就能看到最新进度。
      #   本项目移除了 W&B，原版这里是走 get_wandb_checkpoint_path，我们改用本地的
      #   get_checkpoint_path（run 目录按名字排序=按时间戳，checkpoint 有补零排序，
      #   所以取到的一定是时间上最新的那个）。
      if not log_root_path.exists():
        raise FileNotFoundError(
          f"找不到日志目录 {log_root_path}（任务 {task_id} 还没训练过？）"
          f"\n  也可以直接指定：--checkpoint_file=<某个 model_*.pt>"
        )
      # ★ 从最新 run 往回找，取【第一个真的有 checkpoint 的 run】。
      #   为什么不能直接用 get_checkpoint_path：它只认最新那个 run，而刚启动的
      #   run 目录会先建出来、checkpoint 要等第一次 save 才有（新建训练要等 100+ 轮），
      #   这期间 play 会失败。训练中随时看进度不该被这个窗口期卡住。
      import re as _re

      candidates = sorted(
        (
          d.name
          for d in log_root_path.iterdir()
          if d.is_dir() and _re.match(cfg.load_run, d.name)
        ),
        reverse=True,  # run 名以时间戳开头，倒序 = 最新的在前
      )
      skipped: list[str] = []
      for name in candidates:
        try:
          resume_path = get_checkpoint_path(
            log_root_path, _re.escape(name), cfg.load_checkpoint
          )
          break
        except ValueError:
          skipped.append(name)
      else:
        raise ValueError(
          f"在 {log_root_path} 下没有找到任何匹配 {cfg.load_checkpoint!r} 的 checkpoint。\n"
          f"  该任务已有的 run：\n"
          + "\n".join(f"    {r}" for r in sorted(candidates))
        )
      msg = (
        f"[INFO]: 未指定 --checkpoint_file，自动选用最新 checkpoint：\n"
        f"        run        {resume_path.parent.name}\n"
        f"        checkpoint {resume_path.name}\n"
      )
      if skipped:
        msg += (
          f"        （已跳过 {len(skipped)} 个尚无 checkpoint 的更新 run："
          f"{', '.join(skipped)}）\n"
        )
      msg += (
        "        （想看别的：--checkpoint_file=<路径>，"
        "或用 --load-run/--load-checkpoint 过滤）"
      )
      print(msg)
    log_dir = resume_path.parent

  if TRAINED_MODE:
    from src.mgdp_checkpoint import bind_checkpoint_encoder
    bind_checkpoint_encoder(env_cfg, resume_path, getattr(cfg, "perception_checkpoint", None))
    from src.tasks.velocity.mdp.passage import PassageCommandCfg
    if isinstance(env_cfg.commands.get('twist'),PassageCommandCfg):
      from src.config.mgdp_skill_training import restore_skill_for_play
      restore_skill_for_play(env_cfg,torch.load(resume_path,map_location='cpu',weights_only=False))
      from src.config.passage_speed import restore_fixed_speed_for_play
      fixed=restore_fixed_speed_for_play(env_cfg,torch.load(resume_path,map_location='cpu',weights_only=False))
      from src.config.tire_guidance import restore_tire_guidance_for_play
      restore_tire_guidance_for_play(env_cfg,torch.load(resume_path,map_location='cpu',weights_only=False))
      from src.config.mgdp_course_contract import restore_for_play
      restore_for_play(env_cfg,torch.load(resume_path,map_location='cpu',weights_only=False))
      if fixed is not None:
        print(f'[INFO] Restored passage fixed cruise: {fixed:g} m/s',flush=True)

  if TRAINED_MODE and hasattr(agent_cfg.actor, "history_length"):
    # Read only the recorded protocol before allocating physics/rollout buffers.
    # New training defaults to H=3, while old H=6 policies stay playable as H=6.
    saved = torch.load(resume_path, map_location="cpu", weights_only=True)
    history_length = apply_him_play_contract(env_cfg, agent_cfg, saved.get("him_contract"))
    del saved
    print(f"[INFO]: HIM checkpoint history: {history_length} x {agent_cfg.actor.frame_dim}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  from src.sim_memory import compact_model_allocations
  with compact_model_allocations(cfg.optimize_vram):
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = ActionMonitorVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  action_monitor = env
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    try:
      runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
      )
    except (RuntimeError, ValueError) as e:
      # ★ 自动选 checkpoint 时最容易踩的坑：不同任务共用一个 experiment_name。
      #   例如 LgDog1-Flat 和 LgDog1-Rough 都是 `lgdog1_velocity`，自动取「最新 run」
      #   会拿到另一个任务的权重（观测 234 维 vs 47 维），报一堆看不懂的 size mismatch。
      hint = ""
      if cfg.checkpoint_file is None:
        hint = (
          f"\n\n  ★ 这个 checkpoint 是自动选的：\n"
          f"      run {resume_path.parent.name} / {resume_path.name}\n"
          f"    任务 {task_id} 的 experiment_name 是 {agent_cfg.experiment_name!r}，"
          f"该目录下可能有别的任务的 run。\n"
          f"    请用 --load-run=<正则> 指定正确的 run，或直接 --checkpoint_file=<路径>。\n"
          f"    目录里的 run：\n"
          + "\n".join(
            f"      {d.name}" for d in sorted(resume_path.parent.parent.iterdir())
            if d.is_dir()
          )
        )
      raise type(e)(f"{e}{hint}") from None
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    from src.camera_inset import native_viewer_with_inset
    native_viewer_with_inset(env, policy, cfg.camera_inset, cfg.camera_inset_size,
                            cfg.camera_inset_fps, cfg.camera_inset_corner).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  if action_monitor.action_monitor_steps:
    print(
      "[动作监控] 原始 abs max={:.3f}，最近 abs mean={:.3f}，最近饱和率={:.2%}，步数={}（裁剪前统计）".format(
        action_monitor.raw_action_abs_max_seen,
        action_monitor.raw_action_abs_mean_last,
        action_monitor.action_saturation_fraction_last,
        action_monitor.action_monitor_steps,
      )
    )
    if (action_monitor.raw_action_abs_max_seen > 3.0
        or action_monitor.action_saturation_fraction_last > 0.05):
      print("[动作监控] ⚠️ 原始动作偏大或饱和率偏高，回放中出现腿部抽动时优先检查该策略。")

  env.close()


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

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
