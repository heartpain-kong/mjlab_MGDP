"""Validate LgDog task isolation, model contract and CPU ray geometry.

Run in the project's mjlab 1.2 environment:
  python scripts/check_lgdog_terrain_v4.py
Optional GPU manager smoke test:
  python scripts/check_lgdog_terrain_v4.py --device cuda:0
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

import src.tasks  # noqa: F401
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg
from src.tasks.velocity.config.lgdog1.blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default=None)
  args = parser.parse_args()
  old = load_env_cfg("LgDog1-Rough-Blind-Gentle")
  before = deepcopy(old.scene.terrain.terrain_generator)
  cfg = lgdog1_blind_terrain_env_cfg()
  lgdog1_blind_terrain_env_cfg(play=True)
  assert old.scene.terrain.terrain_generator == before, "Task config contamination"
  actor = cfg.observations["actor"]
  assert actor == old.observations["actor"], "Deployable actor contract changed"
  assert cfg.actions == old.actions
  assert cfg.scene.entities["robot"] == old.scene.entities["robot"]
  assert cfg.decimation == old.decimation == 4
  assert cfg.sim.mujoco.timestep == old.sim.mujoco.timestep == .005
  assert "height_scan" in cfg.observations["critic"].terms
  assert "height_scan" not in actor.terms
  for sensor in cfg.scene.sensors:
    if sensor.name.endswith("_ground_ray"):
      origins, directions = sensor.pattern.generate_rays(None, "cpu")
      torch.testing.assert_close(origins, torch.tensor([[0., 0., .5]]))
      torch.testing.assert_close(directions, torch.tensor([[0., 0., -1.]]))
      assert sensor.include_geom_groups == (0,)
  robot = Entity(cfg.scene.entities["robot"])
  model = robot.spec.compile()
  assert model.nu == 12
  assert len(robot.joint_names) == 12
  action_ids, action_names = robot.find_joints_by_actuator_names((".*",))
  assert tuple(action_names) == robot.joint_names
  print("PASS: actor config unchanged (47 inputs), 12 actions in natural joint order")
  print("PASS: task isolation, 50 Hz control, terrain-only foot ray patterns")
  print("Joint order:", action_names)
  print("Model mass (kg):", float(model.body_mass.sum()))
  print("Actuator force limits:", model.actuator_forcerange.tolist())

  # MuJoCo's actual ray API: robot visuals must not occlude terrain (group 0).
  spec = robot.spec
  terrain = spec.worldbody.add_body(name="test_terrain")
  terrain.add_geom(name="raised_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=(0, 0, .01), pos=(0, 0, .2), group=0)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  geom_id = np.array([-1], dtype=np.int32)
  distance = mujoco.mj_ray(model, data, np.array([0., 0., .8]),
    np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
    True, -1, geom_id)
  assert abs(distance - .6) < 1e-6
  print("PASS: native MuJoCo ray hits raised terrain while excluding robot meshes")

  if args.device:
    from mjlab.envs import ManagerBasedRlEnv
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 20
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
      obs, _ = env.reset()
      assert obs["actor"].shape == (2, 47)
      assert obs["critic"].shape == (2, 261)
      for _ in range(10):
        obs, reward, terminated, truncated, extras = env.step(
          torch.zeros((2, 12), device=args.device)
        )
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.isfinite(reward).all()
      env.reset(env_ids=torch.tensor([0], device=args.device))
      print("PASS: live managers, raycasts, reset, 10 control steps, finite outputs")
    finally:
      env.close()


if __name__ == "__main__":
  main()
