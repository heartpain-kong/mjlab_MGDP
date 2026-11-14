"""Sequential same-workload SAP/BVH rollout comparison in fresh processes.

Run with no competing training process. CUDA wall time is measured after warmup;
initial compilation/construction is excluded. This is a standing-contact probe,
not a convergence benchmark. Pass --profile-stages for diagnostic stage spans.
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
  p.add_argument("--profile-stages", action="store_true")
  p.add_argument("--output", default=".artifacts/mgdp-speed")
  args = p.parse_args()
  if min(args.envs, args.rows, args.cols, args.steps, args.repeats) < 1:
    p.error("counts must be positive")
  root = Path(__file__).resolve().parents[1]
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  reports = {"sap": [], "auto": []}
  for repeat in range(args.repeats):
    # Alternate the order to reduce systematic warm-cache bias.
    for mode in (("sap", "auto") if repeat % 2 == 0 else ("auto", "sap")):
      stem = output / f"{mode}_{repeat}"
      command = [sys.executable, "-u", "-m", "scripts.check_mgdp_runtime",
                 "--collision-mode", mode, "--output", str(stem.with_suffix(".json"))]
      for name in ("device", "envs", "rows", "cols", "steps", "warmup", "perception"):
        command.extend(("--" + name, str(getattr(args, name))))
      if args.profile_stages:
        command.append("--profile-stages")
      print("Running:", " ".join(command), flush=True)
      with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
      report = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
      if not report["passed"] or (mode == "auto" and not report["static_collision_bvh"]):
        raise RuntimeError(f"{mode}: probe failed or BVH fallback; inspect {stem}.log")
      reports[mode].append(report)
      print(f"{mode}: {report['sample_seconds']:.3f}s, "
            f"{report['env_steps_per_second']:.1f} env steps/s", flush=True)
  seconds = {mode: sum(r["sample_seconds"] for r in runs) / len(runs)
             for mode, runs in reports.items()}
  result = {"mean_seconds": seconds, "speedup": seconds["sap"] / seconds["auto"],
            "reports": reports, "profiled": args.profile_stages}
  (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({"mean_seconds": seconds, "speedup": result["speedup"]}, indent=2))


if __name__ == "__main__":
  main()
