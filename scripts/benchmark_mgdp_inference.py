"""Sequential BVH+perception A/B: eager FP32, graph FP32, graph BF16.

Validates recurrent features first in independent processes. A numerical BF16
check does not establish long-run policy quality. No competing training jobs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--envs", type=int, default=5120)
  p.add_argument("--rows", type=int, default=9)
  p.add_argument("--cols", type=int, default=10)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--warmup", type=int, default=20)
  p.add_argument("--repeats", type=int, default=2)
  p.add_argument("--perception", required=True)
  p.add_argument("--batch-size", type=int, default=1024)
  p.add_argument("--include-bf16", action="store_true")
  p.add_argument("--output", default=".artifacts/mgdp-inference")
  args = p.parse_args()
  if min(args.envs, args.rows, args.steps, args.repeats) < 1 or args.batch_size < 0 or args.warmup < 0 or args.cols < 10:
    p.error("positive envs/rows/steps/repeats, nonnegative batch/warmup, cols >=10 required")
  root = Path(__file__).resolve().parents[1]
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  variants = [("native_eager_fp32", "eager", "fp32"), ("eager_fp32", "eager", "fp32"), ("graph_fp32", "cuda_graph", "fp32")]
  if args.include_bf16:
    variants.append(("graph_bf16", "cuda_graph", "bf16"))
  def run(command, log):
    print("Running:", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
      result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
      print(log.read_text(encoding="utf-8")[-8000:], flush=True)
      raise RuntimeError(f"Exit {result.returncode}; inspect {log}")
  for name, execution, precision in variants:
    if name == "native_eager_fp32":
      continue  # Same numerical inference path as eager_fp32.
    run([sys.executable, "-u", "-m", "scripts.check_mgdp_inference", "--device", args.device,
         "--perception", args.perception, "--execution", execution, "--precision", precision,
         "--batch-size", str(args.batch_size), "--envs", str(min(args.envs, 2*(args.batch_size or args.envs)+3)),
         "--output", str(output / (name + "_validation.json"))], output / (name + "_validation.log"))
  reports = {name: [] for name, _, _ in variants}
  for repeat in range(args.repeats):
    for name, execution, precision in (variants if repeat % 2 == 0 else list(reversed(variants))):
      stem = output / f"{name}_{repeat}"
      command = [sys.executable, "-u", "-m", "scripts.check_mgdp_runtime", "--collision-mode", "auto",
                 "--perception-execution", execution, "--perception-precision", precision,
                 "--perception-batch-size", str(args.batch_size), "--output", str(stem.with_suffix(".json"))]
      for field in ("device", "envs", "rows", "cols", "steps", "warmup", "perception"):
        command.extend(("--"+field, str(getattr(args, field))))
      if name == "native_eager_fp32":
        command.append("--native-kinematics")
      run(command, stem.with_suffix(".log"))
      report = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
      assert report["passed"] and report["static_collision_bvh"]
      if execution == "cuda_graph" and args.device.startswith("cuda"):
        assert report["perception_graph_replays"] > 0
      reports[name].append(report)
      print(f"{name}: {report['sample_seconds']:.3f}s; {report['env_steps_per_second']:.1f} env steps/s", flush=True)
  means = {name: sum(r["sample_seconds"] for r in runs)/len(runs) for name, runs in reports.items()}
  summary = {"mean_seconds": means, "speedup_vs_native_eager": {name: means["native_eager_fp32"]/value for name,value in means.items()}}
  (output/"comparison.json").write_text(json.dumps({**summary, "reports": reports}, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
  main()
