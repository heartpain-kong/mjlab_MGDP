"""HIM CPU/GPU integration smoke, export parity and resume-state verification.

Example: python scripts/check_lgdog_him.py --device cpu --output logs/him_smoke
Standard rough: add --task LgDog1-Rough-Blind-HIM-Standard.
The produced weights are smoke-test artifacts, not locomotion policies.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.policy_history import PolicyHistory, validate_policy_history
from src.tasks.velocity.rl.him import HIMRunner
from src.tasks.velocity.config.lgdog1.blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", choices=("LgDog1-Rough-Blind-HIM", "LgDog1-Rough-Blind-HIM-Standard"),
                      default="LgDog1-Rough-Blind-HIM")
  args = parser.parse_args()
  cfg = load_env_cfg(args.task)
  standard = args.task.endswith("-Standard")
  if standard:
    baseline = lgdog1_blind_terrain_env_cfg(
      reward_profile="him_standard", gentle_terrain=False, critic_height_scan=False,
    )
    assert "terrain_scan" not in {sensor.name for sensor in cfg.scene.sensors}
    assert all("height_scan" not in group.terms for group in cfg.observations.values())
  else:
    baseline = load_env_cfg("LgDog1-Rough-Blind-TerrainV4")
  critic_dim = 74 if standard else 261
  assert cfg.actions == baseline.actions
  assert cfg.scene.entities == baseline.scene.entities
  assert cfg.observations["critic"] == baseline.observations["critic"]
  assert cfg.rewards == baseline.rewards
  cfg.scene.num_envs = 4
  cfg.scene.terrain.terrain_generator.num_rows = 2
  cfg.scene.terrain.terrain_generator.num_cols = 20
  cfg.seed = 17
  cfg.sim.nan_guard.enabled = True
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (4, 6, 47)
    assert obs["critic"].shape == (4, critic_dim)
    assert obs["him_target"].shape == (4, 47)
    assert obs["him_velocity"].shape == (4, 3)
    deploy = PolicyHistory()
    for _ in range(9):
      flat = deploy.push(obs["actor"][0, -1].cpu().numpy())
      np.testing.assert_array_equal(flat, obs["actor"][0].cpu().numpy().reshape(-1))
      obs, rewards, _, _, _ = env.step(torch.zeros(4, 12, device=args.device))
      assert torch.isfinite(rewards).all()
    obs, _ = env.reset(env_ids=torch.tensor([0], device=args.device))
    torch.testing.assert_close(obs["actor"][0], obs["actor"][0, -1:].expand(6, -1))
    print(f"PASS: actual actor [4,6,47], critic [4,{critic_dim}], temporal order and partial reset")

    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(args.task)
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.num_steps_per_env = 12
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 2
    agent_cfg.run_name = "smoke_only_not_a_trained_policy"
    runner = HIMRunner(wrapped, asdict(agent_cfg), str(args.output), args.device)
    initial_encoder = [p.detach().clone() for p in runner.alg.actor.estimator.encoder.parameters()]
    runner.learn(2, init_at_random_ep_len=False)
    assert any(not torch.equal(a, b) for a, b in zip(initial_encoder, runner.alg.actor.estimator.encoder.parameters()))
    print("PASS: two PPO + HIM updates, estimator weights changed, checkpoint/export completed")

    policy = args.output / "policy.onnx"
    onnx.checker.check_model(str(policy), full_check=True)
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    validate_policy_history(session.get_inputs()[0].shape, metadata, 6)
    rng = np.random.default_rng(17)
    for raw in (np.zeros((1, 282), np.float32), rng.normal(size=(1, 282)).astype(np.float32)):
      td = TensorDict({"actor": torch.from_numpy(raw).reshape(1, 6, 47).to(args.device)}, batch_size=[1])
      with torch.no_grad():
        expected = runner.alg.actor(td).cpu().numpy()
      actual = session.run(None, {session.get_inputs()[0].name: raw})[0]
      np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
      assert actual.shape == (1, 12) and np.isfinite(actual).all()
    print("PASS: ONNX Runtime / PyTorch parity [1,282] -> [1,12] and history metadata")

    # Load only our newly generated trusted checkpoint through the real runner.
    resumed = HIMRunner(wrapped, asdict(agent_cfg), None, args.device)
    resumed.load(str(args.output / "model_1.pt"), map_location=args.device)
    probe = wrapped.get_observations()
    with torch.no_grad():
      torch.testing.assert_close(resumed.alg.actor(probe), runner.alg.actor(probe))
    assert resumed.alg.estimator_optimizer.state_dict()["state"]
    assert resumed.current_learning_iteration == 1
    print("PASS: runner checkpoint restores actor, estimator, normalizer and auxiliary optimizer")
  finally:
    env.close()


if __name__ == "__main__":
  main()
