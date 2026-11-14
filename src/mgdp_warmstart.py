"""Explicit policy transfer to a changed MGDP course, separate from full resume."""
from pathlib import Path
import torch
from src.mgdp_checkpoint import validate_runtime_encoder, file_digest


def warm_start_mgdp_policy(runner, path):
  env = runner.env.unwrapped
  if "perception" not in env.cfg.observations:
    raise ValueError("--warm-start-checkpoint is only supported for MGDP policy transfer")
  checkpoint = torch.load(path, map_location=runner.device, weights_only=True)
  if not (checkpoint.get("infos") or {}).get("mgdp_perception"):
    raise ValueError("Policy transfer requires a checkpoint with the MGDP encoder contract")
  validate_runtime_encoder(env, checkpoint)
  # Strict loading includes policy normalization and Gaussian std, but deliberately
  # leaves the new critic, optimizer, iteration and curriculum untouched.
  runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
  runner._mgdp_warm_start = {
      "path": str(Path(path).resolve()), "sha256": file_digest(path),
      "source_iteration": checkpoint.get("iter"), "mode": "actor_only_new_curriculum"}
  env.reset()
  print(f"[INFO] MGDP policy transferred from {path}; new critic/optimizer/iteration/course", flush=True)
  return runner._mgdp_warm_start
