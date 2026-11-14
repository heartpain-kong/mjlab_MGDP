"""Check MGDP physics/sensing before long training; write a machine-readable report.

Run from project root: python -m scripts.check_mgdp_runtime --device cuda:0
The timing includes physics, sensing and observations, with optional perception.
Zero actions test contact integrity, not locomotion or obstacle success.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import uuid

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from src.cuda_initialization import trace_cuda_initialization, trace_cuda_execution
from src.scene_diagnostics import report_scene_initialization, report_device_memory
from src.sim_memory import compact_model_allocations
from src.runtime_profile import StageProfiler
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg


def run(args):
  cfg = lgdog1_race_mgdp_env_cfg(
    checkpoint=args.perception or "", with_perception=bool(args.perception),
    terrain_only_sensing=not args.legacy_sensing)
  cfg.scene.num_envs = args.envs
  cfg.seed = args.seed
  from src.tasks.velocity.mdp.mgdp_inference import configure_perception_runtime
  configure_perception_runtime(cfg, getattr(args, "perception_execution", None),
                               getattr(args, "perception_precision", None),
                               getattr(args, "perception_batch_size", None))
  cfg.scene.terrain.terrain_generator.num_rows = args.rows
  cfg.scene.terrain.terrain_generator.num_cols = args.cols
  result = {"device": args.device, "envs": args.envs, "grid": [args.rows, args.cols],
            "optimized": not args.native, "perception": args.perception,
            "terrain_only_sensing": not args.legacy_sensing,
            "steps": args.steps, "warmup": args.warmup, "passed": False}
  if args.perception:
    result["perception_runtime"] = {
      key: value for key, value in cfg.observations["perception"].terms["depth_feature"].params.items()
      if key.startswith("inference_")}
  trace_dir = Path(args.output).parent / "runtime_init" / uuid.uuid4().hex[:12]
  result["trace_directory"] = str(trace_dir)
  result["phase"] = "construction"
  env = None
  def synchronize():
    wp.synchronize_device(args.device)
  try:
    started = time.perf_counter()
    with report_scene_initialization(cfg), compact_model_allocations(not args.native, getattr(args, "collision_mode", "auto"), not getattr(args,"native_kinematics",False)), \
         trace_cuda_initialization(args.device, trace_dir):
      env = ManagerBasedRlEnv(cfg, device=args.device)
    rc = env.sim._sensor_context.render_context
    result.update(ngeom=env.sim.wp_model.ngeom, bvh_leaves=rc.lower.shape[0],
                  shared_bvh=bool(getattr(rc, "_lgdog_shared_terrain", False)),
                  sparse_pairs=hasattr(env.sim.wp_model, "_lgdog_sparse_pairs"))
    from src._mjwarp_static_collision import available
    result["static_collision_bvh"] = available(env.sim.wp_model)
    from src._mjwarp_active_kinematics import available as active_available
    result["active_geom_kinematics"] = active_available(env.sim.wp_model)
    workspace = getattr(env.sim.wp_data, "_lgdog_sap_workspace", None)
    result["sap_workspace_prepared"] = workspace is not None
    result["sap_sort_prepared"] = bool(workspace and workspace[1].get("sort_prepared", False))
    debug = (trace_cuda_execution(args.device, trace_dir)
             if getattr(args, "debug_sync", False) else nullcontext())
    with debug:
      result["phase"] = "reset"
      env.reset()
      synchronize()
    result["init_seconds"] = time.perf_counter() - started
    actions = torch.zeros((args.envs, env.action_space.shape[-1]), device=args.device)
    with torch.inference_mode():
      result["phase"] = "warmup"
      for _ in range(args.warmup):
        env.step(actions)
      synchronize()
      started = time.perf_counter()
      result["phase"] = "sampling"
      with StageProfiler(args.device) as profiler:
        if getattr(args, "profile_stages", False):
          for method in ("step", "forward", "sense"):
            profiler.watch(env.sim, method, "simulation_" + method)
          for label in ("observation", "reward", "termination", "command"):
            profiler.watch(getattr(env, label + "_manager"), "compute", label, stream="torch")
          if args.perception:
            module = env._mgdp_perception_state["module"]
            profiler.watch(module, "feature_batched", "perception", stream="torch")
        for _ in range(args.steps):
          env.step(actions)
        synchronize()
      if getattr(args, "profile_stages", False):
        result["stage_timings"] = profiler.report()
    result["sample_seconds"] = time.perf_counter() - started
    result["env_steps_per_second"] = args.envs * args.steps / result["sample_seconds"]
    if args.perception:
      runtime = getattr(env._mgdp_perception_state["module"], "_frozen_depth_runtime", None)
      result["perception_graph_count"] = len(runtime.graphs) if runtime else 0
      result["perception_graph_replays"] = runtime.graph_replays if runtime else 0
      if runtime and runtime.execution == "cuda_graph" and str(args.device).startswith("cuda") and not runtime.graph_replays:
        raise RuntimeError("Requested perception CUDA graphs were not replayed")
    robot = env.scene["robot"].data
    contacts = env.scene["feet_ground_contact"].data.found
    result["foot_contact_mean"] = float(contacts.float().mean())
    result["nacon"] = int(env.sim.wp_data.nacon.numpy()[0])
    result["root_z_mean"] = float(robot.root_link_pos_w[:, 2].mean())
    result["finite_physics"] = bool(torch.isfinite(robot.root_link_pos_w).all())
    result["finite_sensors"] = all(bool(torch.isfinite(env.scene[name].data.distances).all())
                                   for name in ("terrain_scan", "depth_rays"))
    result["finite_perception"] = (not args.perception or
                                   bool(torch.isfinite(env.obs_buf["perception"]).all()))
    result["memory"] = report_device_memory(args.device, "runtime-check")
    if not result["finite_physics"] or not result["finite_sensors"] or not result["finite_perception"]:
      raise RuntimeError("Nonfinite physics or sensor values; do not start long training")
    if result["nacon"] == 0 or result["foot_contact_mean"] == 0:
      raise RuntimeError("No standing contacts at the entry pads; inspect collision backend before training")
    result["passed"] = True
    result["phase"] = "complete"
  except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MGDP_RUNTIME " + json.dumps(result, ensure_ascii=False), flush=True)
    if env is not None:
      env.close()
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--envs", type=int, default=64)
  p.add_argument("--rows", type=int, default=2)
  p.add_argument("--cols", type=int, default=10)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--warmup", type=int, default=20)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--perception", help="Include frozen encoder inference using this checkpoint")
  p.add_argument("--perception-execution", choices=("eager", "cuda_graph"))
  p.add_argument("--perception-precision", choices=("fp32", "bf16"))
  p.add_argument("--perception-batch-size", type=int)
  p.add_argument("--native", action="store_true", help="Disable allocation/backend adaptations")
  p.add_argument("--native-kinematics", action="store_true", help="Keep native all-geom thread grid for A/B")
  p.add_argument("--collision-mode", choices=("auto", "sap"), default="auto")
  p.add_argument("--profile-stages", action="store_true", help="Queue physics/forward/sensing stage timings; small profiling overhead")
  p.add_argument("--legacy-sensing", action="store_true", help="Use pre-refactor robot-inclusive rays")
  p.add_argument("--debug-sync", action="store_true", help="Synchronize and trace reset forward/sense boundaries")
  p.add_argument("--output", default=".artifacts/mgdp-runtime/report.json")
  args = p.parse_args()
  if min(args.envs, args.rows, args.steps) < 1 or args.warmup < 0 or args.cols < 10:
    p.error("envs/rows/steps must be positive, warmup nonnegative, cols >=10")
  run(args)


if __name__ == "__main__":
  main()
