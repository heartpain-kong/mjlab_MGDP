"""Distil the privileged height-scan wall teacher into a depth-only student.

The high-wall course is stuck at ~0% success under plain RL.  Following the CMU
Extreme Parkour teacher-student recipe, this script runs DAgger from a teacher
that sees the 187-dim ``height_scan`` (privileged) into a student that only sees
the deployable 252-dim blind observation (proprioception + depth feature).

Workflow expected by this script::

  1. (Optional, warm start) Expand today's 252-dim blind wall actor into a
     439-dim teacher initialisation checkpoint, keeping its MGDP perception
     contract/sidecar::

       python scripts/distill_wall.py \
         --teacher logs/rsl_rl/lgdog1_velocity_race_mgdp_wall_v5/<run>/model_10000.pt \
         --write-teacher-init logs/distill/teacher_init.pt

  2. RL fine-tune the privileged teacher on the height-augmented task (the
     teacher task adds a separate ``"privileged"`` observation group, so the
     warm start and MGDP contracts keep working)::

       python -c "import sys, runpy; import src.tasks; \
         from src.tasks.velocity.rl.distill import register_teacher_task; \
         register_teacher_task(); sys.argv=['scripts/train.py', \
         'LgDog1-Race-MGDP-Wall-Teacher']; \
         runpy.run_path('scripts/train.py', run_name='__main__')" \
         --warm-start-checkpoint logs/distill/teacher_init.pt

  3. Distil the teacher into the blind student::

       python scripts/distill_wall.py \
         --teacher logs/rsl_rl/lgdog1_velocity_race_mgdp_wall_teacher_v1/<run>/model_XXXX.pt \
         --iterations 500 --lr 3e-4 --device cuda

  4. The ``--out`` checkpoint is a normal rsl_rl checkpoint whose
     ``actor_state_dict`` is the 252-dim blind student (same input order as
     today's wall policy), so it can be loaded with ``scripts/play.py`` once the
     teacher run's MGDP perception sidecar is available next to it.

``--teacher-order``: a 439-dim teacher trained through ``register_teacher_task``
uses the canonical ``[proprio | depth | height]`` order (default).  Only a teacher
trained by plain PPO on ``build_distill_env_cfg`` (height appended to the actor
group) needs ``--teacher-order env``.

Per-lesson teachers (``--lesson``)
----------------------------------
By default the teacher is trained on all ten wall lessons mixed together.  The
high wall (``high_wall``) and the stepping stones (``stepping_stones``) can each
get a teacher of their own (fixed terrain, concentrated difficulty)::

    python scripts/distill_wall.py --list-lessons            # lessons + teacher ids
    python scripts/distill_wall.py --lesson high_wall --register-teacher
    # then RL fine-tune LgDog1-Race-MGDP-Wall-Teacher-high_wall (see printed cmd)
    python scripts/distill_wall.py --lesson high_wall \
      --teacher logs/rsl_rl/lgdog1_wall_teacher_high_wall/<run>/model_XXXX.pt \
      --out logs/distill/wall_student_high_wall/model.pt

``--lesson all`` (the default) keeps the previous ten-lesson behaviour and the
original teacher task id ``LgDog1-Race-MGDP-Wall-Teacher``.  A per-lesson run
filters ``cfg.scene.terrain.terrain_generator.sub_terrains`` to that one lesson
(``proportion=1.0``) and prints a ``[distill] lesson=<name> rows=.. cols=..`` line.

If no privileged teacher exists yet, the current 252-dim wall checkpoint can also
be passed directly as ``--teacher`` (zero-padded, so it ignores the height scan)
or used via ``--student-init``; distillation alone then reproduces the old
behaviour -- the teacher has to be RL fine-tuned first to add new skills.

Run ``python scripts/distill_wall.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import torch

from src.tasks.velocity.rl.distill import (
  ACTION_DIM,
  DEFAULT_ACTIVATION,
  DEFAULT_INIT_STD,
  INPUT_ORDER_CANONICAL,
  INPUT_ORDER_ENV,
  LESSON_ALL,
  WALL_TASK_ID,
  MaskedActor,
  available_lessons,
  build_distill_env,
  expand_blind_actor_state_dict,
  extract_actor_state_dict,
  infer_actor_architecture,
  load_actor_state_dict,
  prepare_actor_state_dict,
  register_teacher_task,
  resolve_checkpoint_path,
  save_teacher_init_checkpoint,
  teacher_task_id,
  train_distill,
)

DEFAULT_OUT = str(Path("logs") / "distill" / "wall_student" / "model.pt")

# The wall course's ten terrain lessons (``passage_terrain_cfg.KINDS``).  Kept as
# a static tuple so ``--help`` does not need to import the simulation stack; the
# authoritative list is printed by ``--list-lessons`` and validated at runtime
# by :func:`filter_terrain_lessons` (``LGDOG_WALL_LESSONS`` can shrink it).
LESSON_CHOICES = (
  LESSON_ALL,
  "start_flat",
  "stepping_stones",
  "balance_beam",
  "gap_steps",
  "high_wall",
  "clearance_frame",
  "tire_array",
  "random_hills",
  "mjlab_rough",
  "slalom_brick",
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="DAgger distillation: privileged height-scan teacher -> blind depth student "
                "(LgDog1 high-wall task).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--teacher", default=None,
                      help="teacher checkpoint (.pt) trained on the height-augmented task; "
                           "required unless --list-lessons/--register-teacher is used")
  parser.add_argument("--out", default=DEFAULT_OUT,
                      help="output checkpoint file (.pt) or directory")
  parser.add_argument("--iterations", type=int, default=500,
                      help="number of DAgger rounds (each collects steps-per-iteration env steps)")
  parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
  parser.add_argument("--device", default="cuda", help="torch device (falls back to CPU if CUDA is unavailable)")
  parser.add_argument("--task", default=WALL_TASK_ID, help="wall training task to build the env from")
  parser.add_argument("--lesson", choices=LESSON_CHOICES, default=LESSON_ALL,
                      help="terrain lesson to train/distil in isolation: 'all' keeps the full "
                           "ten-lesson course, otherwise the terrain roster is filtered to that "
                           "one lesson (proportion=1.0) and the teacher task id becomes "
                           "LgDog1-Race-MGDP-Wall-Teacher-<lesson>")
  parser.add_argument("--list-lessons", action="store_true",
                      help="print the available terrain lessons and their teacher task ids, then exit")
  parser.add_argument("--register-teacher", action="store_true",
                      help="register the teacher RL task for --lesson (or the base task for 'all') "
                           "and exit; scripts/train.py needs the registration in its own process "
                           "(see the printed command)")
  parser.add_argument("--perception-checkpoint", default=None,
                      help="optional frozen MGDP depth encoder override (must match the current "
                           "perception architecture and camera/sensing contract)")
  parser.add_argument("--num-envs", type=int, default=1,
                      help="number of environments (the design is single-env; >1 is allowed but untested)")
  parser.add_argument("--steps-per-iteration", type=int, default=32,
                      help="student-driven env steps collected per DAgger round")
  parser.add_argument("--updates-per-iteration", type=int, default=1,
                      help="gradient passes over each collected buffer")
  parser.add_argument("--minibatch-size", type=int, default=64, help="mini-batch size of the regression updates")
  parser.add_argument("--loss", choices=("mse", "huber"), default="mse", help="imitation loss")
  parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber transition point (--loss huber)")
  parser.add_argument("--action-noise-std", type=float, default=0.0,
                      help="Gaussian noise added to the student action executed in the env (0 = pure student)")
  parser.add_argument("--clip-actions", type=float, default=1.0,
                      help="clip executed actions to [-value, value]; negative disables clipping")
  parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm; 0 disables")
  parser.add_argument("--save-every", type=int, default=50, help="save a checkpoint every N rounds; 0 disables")
  parser.add_argument("--log-every", type=int, default=10, help="print progress every N rounds; 0 disables")
  parser.add_argument("--seed", type=int, default=42, help="torch/env seed")
  parser.add_argument("--teacher-input", choices=("full", "privileged_only"), default="full",
                      help="full: teacher sees student+privileged observations (recommended: how a PPO "
                           "teacher is trained). privileged_only: strict complementary masks from the "
                           "design brief (teacher sees only the height scan).")
  parser.add_argument("--teacher-order", choices=(INPUT_ORDER_CANONICAL, INPUT_ORDER_ENV),
                      default=INPUT_ORDER_CANONICAL,
                      help="observation ordering of a 439-dim teacher checkpoint: canonical = "
                           "[proprio|depth|height] (what register_teacher_task() produces; this "
                           "module's layout); env = plain PPO on the actor-group-appended config "
                           "([proprio|height|depth])")
  parser.add_argument("--write-teacher-init", default=None, metavar="PATH",
                      help="instead of distilling, expand the --teacher blind (252-dim) actor into "
                           "a 439-dim teacher-initialisation checkpoint and exit; use it with "
                           "scripts/train.py <teacher task> --warm-start-checkpoint PATH")
  parser.add_argument("--student-init", default=None,
                      help="optional student initialisation checkpoint (e.g. today's 252-dim wall "
                           "policy, zero-padded over the privileged slots)")
  parser.add_argument("--student-order", choices=(INPUT_ORDER_CANONICAL, INPUT_ORDER_ENV),
                      default=INPUT_ORDER_CANONICAL,
                      help="observation ordering of a 439-dim --student-init checkpoint")
  parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD,
                      help="Gaussian std written into the exported student actor")
  parser.add_argument("--no-normalizer-update", action="store_true",
                      help="freeze the student observation normalizer during distillation")
  return parser


def resolve_device(requested: str) -> str:
  requested = str(requested)
  if requested.startswith("cuda") and not torch.cuda.is_available():
    print(
      f"[distill] requested --device {requested!r} but CUDA is not available; using CPU instead",
      flush=True,
    )
    return "cpu"
  return requested


def _describe_actor(name: str, actor: MaskedActor) -> str:
  params = actor.parameter_count(trainable_only=False)
  trainable = actor.parameter_count(trainable_only=True)
  return (
    f"[distill] {name:<10} input={actor.input_dim:<4} action_dim={actor.output_dim:<3} "
    f"params={params:>12,} trainable={trainable:>12,} "
    f"hidden={tuple(actor.hidden_dims)} activation={actor.activation}"
  )


def _resolve_lesson(args: argparse.Namespace) -> str | None:
  """Map the CLI ``--lesson all`` sentinel to the API's ``lesson=None``."""
  return None if args.lesson == LESSON_ALL else str(args.lesson)


def _list_lessons(task_id: str) -> int:
  """Print one line per lesson with the teacher task id it maps to."""
  lessons = available_lessons(task_id)
  print(f"[distill] task {task_id}: {len(lessons)} terrain lessons", flush=True)
  print(f"[distill]   {LESSON_ALL:<20} -> {teacher_task_id()}  (all lessons)", flush=True)
  for name in lessons:
    print(f"[distill]   {name:<20} -> {teacher_task_id(name)}", flush=True)
  print(
    "[distill] use --lesson <name> to build/train/distil one lesson, e.g. "
    f"--lesson high_wall (teacher task {teacher_task_id('high_wall')})",
    flush=True,
  )
  return 0


def _register_teacher(args: argparse.Namespace, lesson: str | None) -> int:
  """Register the teacher task for ``lesson`` and print how to train it."""
  task_id = register_teacher_task(lesson=lesson, source_task_id=args.task)
  print(f"[distill] teacher task id: {task_id}", flush=True)
  print(
    "  python -c \"import sys, runpy; import src.tasks; "
    "from src.tasks.velocity.rl.distill import register_teacher_task; "
    f"register_teacher_task(lesson={lesson!r}); "
    f"sys.argv=['scripts/train.py', '{task_id}']; "
    "runpy.run_path('scripts/train.py', run_name='__main__')\" "
    "--warm-start-checkpoint <teacher_init.pt> ...",
    flush=True,
  )
  return 0


def _write_run_report(path: Path, args: argparse.Namespace, layout, result: dict) -> None:
  report = {
    "script": "scripts/distill_wall.py",
    "task": args.task,
    "lesson": args.lesson,
    "teacher_task_id": teacher_task_id(_resolve_lesson(args)),
    "device": args.device,
    "teacher": str(args.teacher),
    "out": str(args.out),
    "layout": layout.describe(),
    "teacher_input": args.teacher_input,
    "teacher_order": args.teacher_order,
    "perception_checkpoint": (
      None if args.perception_checkpoint is None else str(args.perception_checkpoint)
    ),
    "student_init": None if args.student_init is None else str(args.student_init),
    "iterations": args.iterations,
    "lr": args.lr,
    "steps_per_iteration": args.steps_per_iteration,
    "updates_per_iteration": args.updates_per_iteration,
    "minibatch_size": args.minibatch_size,
    "loss": args.loss,
    "history": result.get("history", []),
    "saved_checkpoints": result.get("saved_checkpoints", []),
    "final_checkpoint": result.get("final_checkpoint"),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
  print(f"[distill] run report written to {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  lesson = _resolve_lesson(args)
  if args.list_lessons:
    return _list_lessons(args.task)
  if args.register_teacher:
    return _register_teacher(args, lesson)
  if not args.teacher:
    parser.error("--teacher is required unless --list-lessons or --register-teacher is given")

  torch.manual_seed(int(args.seed))
  device = resolve_device(args.device)
  clip_actions = None if args.clip_actions is None or args.clip_actions < 0 else float(args.clip_actions)

  try:
    lessons = available_lessons(args.task)
  except ValueError as error:
    print(f"[distill][error] {error}", flush=True)
    return 2
  print(f"[distill] available lessons (--lesson): {', '.join((LESSON_ALL, *lessons))}", flush=True)

  print("=" * 100, flush=True)
  print("[distill] LgDog1 high-wall teacher-student DAgger (single env, input masks)", flush=True)
  print(
    f"[distill] task={args.task} lesson={args.lesson} teacher_task="
    f"{teacher_task_id(lesson)} device={device} teacher={args.teacher}",
    flush=True,
  )
  print("=" * 100, flush=True)

  env = build_distill_env(
    args.task,
    device=device,
    num_envs=args.num_envs,
    seed=args.seed,
    perception_checkpoint=args.perception_checkpoint,
    lesson=lesson,
  )
  try:
    layout = env.layout
    print(
      f"[distill] observation layout: student={layout.student_dim} + "
      f"privileged={layout.privileged_dim} = {layout.total_dim} "
      f"(privileged term {layout.privileged_group}/{layout.privileged_term})",
      flush=True,
    )

    teacher_state_dict, _teacher_infos = extract_actor_state_dict(args.teacher)
    architecture = infer_actor_architecture(teacher_state_dict)
    print(
      f"[distill] teacher checkpoint: obs={architecture.obs_dim} "
      f"hidden={architecture.hidden_dims} action={architecture.output_dim}",
      flush=True,
    )
    if int(architecture.output_dim) != int(env.num_actions):
      raise SystemExit(
        f"teacher action dim {architecture.output_dim} != env action dim {env.num_actions}"
      )

    if args.write_teacher_init is not None:
      expanded = expand_blind_actor_state_dict(teacher_state_dict, layout)
      init_path = save_teacher_init_checkpoint(
        args.write_teacher_init, expanded, source_checkpoint=args.teacher
      )
      teacher_id = teacher_task_id(lesson)
      print("=" * 100, flush=True)
      print(f"[distill] teacher-init checkpoint written: {init_path}", flush=True)
      print(
        f"[distill] actor input {layout.student_dim} -> {layout.total_dim} "
        f"(zeroed {layout.privileged_dim}-dim privileged columns); infos and the MGDP "
        "perception sidecar were inherited from the blind checkpoint.",
        flush=True,
      )
      print(
        "[distill] next: register the teacher task and RL fine-tune it so the height "
        "columns become non-zero, e.g.",
        flush=True,
      )
      print(
        "  python -c \"import sys, runpy; import src.tasks; "
        "from src.tasks.velocity.rl.distill import register_teacher_task; "
        f"register_teacher_task(lesson={lesson!r}); sys.argv=['scripts/train.py', "
        f"'{teacher_id}']; "
        "runpy.run_path('scripts/train.py', run_name='__main__')\" "
        f"--warm-start-checkpoint {init_path}",
        flush=True,
      )
      return 0

    teacher = MaskedActor(
      layout,
      hidden_dims=architecture.hidden_dims,
      activation=DEFAULT_ACTIVATION,
      output_dim=architecture.output_dim,
      obs_normalization="obs_normalizer._mean" in teacher_state_dict,
      teacher_sees_student=(args.teacher_input == "full"),
    )
    teacher_state = prepare_actor_state_dict(
      teacher_state_dict, layout, input_order=args.teacher_order, label="teacher"
    )
    load_actor_state_dict(teacher, teacher_state, strict=True)

    student = MaskedActor(
      layout,
      hidden_dims=architecture.hidden_dims,
      activation=DEFAULT_ACTIVATION,
      output_dim=architecture.output_dim,
      obs_normalization=True,
      teacher_sees_student=False,
    )
    if args.student_init is not None:
      student_state_dict, _ = extract_actor_state_dict(args.student_init)
      student_state = prepare_actor_state_dict(
        student_state_dict, layout, input_order=args.student_order, label="student"
      )
      load_actor_state_dict(student, student_state, strict=True)
      print(f"[distill] student warm-started from {args.student_init}", flush=True)

    # Required before training starts: action dimensions and parameter counts.
    print(_describe_actor("teacher", teacher), flush=True)
    print(_describe_actor("student", student), flush=True)
    print(
      f"[distill] action dims: teacher={teacher.output_dim} "
      f"student={student.output_dim} env={env.num_actions}",
      flush=True,
    )
    print(
      f"[distill] deployable student: input={layout.student_dim} "
      f"action_dim={student.output_dim} params={student.deployable_parameter_count():,} "
      "(privileged input columns dropped)",
      flush=True,
    )
    if args.teacher_input == "privileged_only":
      print(
        "[distill][warn] --teacher-input privileged_only: the teacher sees only the "
        "height scan. A teacher trained with plain PPO on the height-augmented task "
        "expects full observations; use --teacher-input full unless the teacher was "
        "trained with this mask.",
        flush=True,
      )

    final_path = resolve_checkpoint_path(args.out, final=True)
    log_path = final_path.parent / "distill_log.jsonl"
    print(f"[distill] output checkpoint: {final_path}", flush=True)
    print(f"[distill] per-iteration log: {log_path}", flush=True)

    result = train_distill(
      env,
      teacher,
      student,
      iterations=args.iterations,
      lr=args.lr,
      steps_per_iteration=args.steps_per_iteration,
      updates_per_iteration=args.updates_per_iteration,
      mini_batch_size=args.minibatch_size,
      loss=args.loss,
      huber_delta=args.huber_delta,
      action_noise_std=args.action_noise_std,
      clip_actions=clip_actions,
      max_grad_norm=args.max_grad_norm,
      save_every=args.save_every,
      out=args.out,
      log_every=args.log_every,
      log_path=log_path,
      device=device,
      update_normalizer=not args.no_normalizer_update,
      seed=args.seed,
      teacher_checkpoint=args.teacher,
      init_std=args.init_std,
    )

    _write_run_report(final_path.parent / "distill_report.json", args, layout, result)

    print("=" * 100, flush=True)
    print(f"[distill] done: {result['iterations']} DAgger rounds", flush=True)
    print(f"[distill] final student checkpoint: {result['final_checkpoint']}", flush=True)
    print(
      "[distill] the checkpoint's actor_state_dict is a standard rsl_rl actor for the "
      f"{layout.student_dim}-dim blind policy (same input order as today's wall policy). "
      "To play it, the teacher run's MGDP perception sidecar must be available next to "
      "the checkpoint (copy the file referenced by infos['mgdp_perception']).",
      flush=True,
    )
    return 0
  finally:
    env.close()


if __name__ == "__main__":
  sys.exit(main())
