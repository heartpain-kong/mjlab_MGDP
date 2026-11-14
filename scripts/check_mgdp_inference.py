"""Compare new inference against the old chunked CNN+GRU over partial resets."""
import argparse
import json
from pathlib import Path

import torch

from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception
from src.mgdp_checkpoint import encoder_fingerprint


@torch.no_grad()
def validate(module, device, execution, precision, batch_size=1024, batch=2051, steps=8):
  fingerprint = encoder_fingerprint(module)
  generator = torch.Generator(device=device).manual_seed(42)
  reference_hidden = actual_hidden = None
  maximum = 0.
  for step in range(steps):
    depth = torch.rand((batch, module.cfg.frames, *module.cfg.depth_hw),
                       device=device, generator=generator)
    if step % 3 == 2:
      reference_hidden[:, ::3] = 0
      actual_hidden[:, ::3] = 0
    reference, states = [], []
    chunk = batch_size or batch
    for start in range(0, batch, chunk):
      hidden = None if reference_hidden is None else reference_hidden[:, start:start+chunk].contiguous()
      feature, hidden = module.feature(depth[start:start+chunk], hidden)
      reference.append(feature)
      states.append(hidden)
    expected = torch.cat(reference)
    reference_hidden = torch.cat(states, dim=1)
    actual, actual_hidden = module.feature_batched(depth, actual_hidden, batch_size, execution, precision)
    assert actual.dtype == actual_hidden.dtype == torch.float32
    atol, rtol = ((.01, .05) if precision == "bf16" else (2e-5, 2e-4))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_hidden, reference_hidden, atol=atol, rtol=rtol)
    maximum = max(maximum, float((actual - expected).abs().max()))
  runtime = module._frozen_depth_runtime
  if execution == "cuda_graph" and torch.device(device).type == "cuda":
    assert runtime.graph_replays > 0
  assert encoder_fingerprint(module) == fingerprint
  return {"passed": True, "device": str(device), "execution": execution, "precision": precision,
          "batch": batch, "steps": steps, "max_abs_feature_error": maximum,
          "graph_count": len(runtime.graphs), "graph_replays": runtime.graph_replays}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--perception", required=True)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--execution", choices=("eager", "cuda_graph"), default="cuda_graph")
  p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
  p.add_argument("--batch-size", type=int, default=1024)
  p.add_argument("--envs", type=int, default=2051)
  p.add_argument("--output", default=".artifacts/mgdp-inference/validation.json")
  args = p.parse_args()
  if args.batch_size < 0 or args.envs < 1:
    p.error("batch size must be nonnegative and envs positive")
  # map_location moves checkpoint tensors, not the freshly constructed module.
  module = MGPDPerception.from_checkpoint(args.perception, map_location="cpu").to(args.device).eval()
  report = validate(module, args.device, args.execution, args.precision, args.batch_size, args.envs)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print("MGDP_INFERENCE " + json.dumps(report), flush=True)


if __name__ == "__main__":
  main()
