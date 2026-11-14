"""Differential collision candidates and trajectories against native SAP."""

from unittest.mock import patch

import mujoco
import mujoco_warp as mw
import numpy as np
import pytest
import warp as wp
from mujoco_warp._src import collision_driver as cd
from mujoco_warp._src.types import BroadphaseFilter, BroadphaseType

from src import _mjwarp_static_collision as static
from src import _mjwarp_sap as sap
from src.sim_memory import compact_model_allocations


def xml(extra="", sensor="", contact=""):
  boxes = ''.join(f'<geom name="tile{i}" type="box" size=".16 .15 .08" '
                  f'pos="{(i%6)*.4-1.} {(i//6)*.4-.5} .08"/>' for i in range(24))
  return f'''<mujoco><option timestep=".005"/><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>{boxes}{extra}
    <body pos="0 0 .19"><freejoint/><geom name="a" type="sphere" size=".1" margin=".01"/>
      <geom name="b" type="capsule" size=".04 .08" pos=".2 0 0"/></body>
    <body pos=".1 .1 .19"><freejoint/><geom name="c" type="box" size=".08 .08 .1"/></body>
  </worldbody>{sensor}{contact}</mujoco>'''


def make(model, worlds=3):
  with compact_model_allocations(collision_mode="sap"):
    wm = mw.put_model(model)
    wd = mw.make_data(model, nworld=worlds, nconmax=200, njmax=400)
  wm.opt.broadphase = BroadphaseType.SAP_SEGMENTED
  return wm, wd


def candidates(model, data, accelerated):
  ctx = cd.create_collision_context(data.naconmax)
  data.ncollision.zero_()
  if accelerated:
    assert static.broadphase(model, data, ctx)
  else:
    state = getattr(model, "_lgdog_static_collision", None)
    model._lgdog_static_collision = None
    try:
      sap.sap_broadphase_i64(model, data, ctx)
    finally:
      model._lgdog_static_collision = state
  count = int(data.ncollision.numpy()[0])
  assert count < data.naconmax
  return sorted((int(w), *p.tolist(), *ids.tolist()) for w,p,ids in zip(
    ctx.collision_worldid.numpy()[:count], ctx.collision_pair.numpy()[:count],
    ctx.collision_pairid.numpy()[:count]))


@pytest.mark.parametrize("filters", [BroadphaseFilter.SPHERE | BroadphaseFilter.PLANE,
  BroadphaseFilter.SPHERE | BroadphaseFilter.PLANE | BroadphaseFilter.OBB,
  BroadphaseFilter.SPHERE | BroadphaseFilter.PLANE | BroadphaseFilter.AABB])
def test_candidates_match_sap_after_distinct_world_motion(filters):
  with wp.ScopedDevice("cpu"):
    host = mujoco.MjModel.from_xml_string(xml())
    m,d = make(host)
    m.opt.broadphase_filter = filters
    assert static.build(host, m, force=True)
    for offset in (0., .1, 1.5):
      q = d.qpos.numpy()
      q[:,0] = np.array([-.8,.0,.9])+offset
      q[:,7] = np.array([-.7,.1,1.1])+offset
      d.qpos.assign(q)
      mw.forward(m,d)
      assert candidates(m,d,True) == candidates(m,d,False)


def test_native_sap_contacts_and_short_trajectories_match():
  with wp.ScopedDevice("cpu"):
    host = mujoco.MjModel.from_xml_string(xml())
    m,fast = make(host)
    assert static.build(host,m,force=True)
    reference = mw.make_data(host,nworld=3,nconmax=200,njmax=400)
    state=m._lgdog_static_collision
    for _ in range(8):
      m._lgdog_static_collision=None
      mw.step(m,reference)
      m._lgdog_static_collision=state
      mw.step(m,fast)
      assert int(fast.nacon.numpy()[0]) > 0
    np.testing.assert_allclose(fast.qpos.numpy(),reference.qpos.numpy(),atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(fast.qvel.numpy(),reference.qvel.numpy(),atol=2e-5,rtol=2e-4)


def test_explicit_contacts_keep_generic_path():
  with wp.ScopedDevice("cpu"):
    for contacts, expected in (( '<contact><pair geom1="a" geom2="b"/></contact>', False),
                               ('<contact><pair geom1="tile0" geom2="tile1"/></contact>', False)):
      host=mujoco.MjModel.from_xml_string(xml(contact=contacts))
      m,d=make(host)
      assert static.build(host,m,force=True) is expected


def test_distance_sensors_mocap_and_changed_geometry_keep_generic_path():
  with wp.ScopedDevice("cpu"):
    host=mujoco.MjModel.from_xml_string(xml(sensor='<sensor><distance geom1="a" geom2="tile0" cutoff="2"/></sensor>'))
    m,_=make(host)
    assert not static.build(host,m,force=True)
    host=mujoco.MjModel.from_xml_string(xml(extra='<body mocap="true"><geom size=".1"/></body>'))
    m,_=make(host)
    assert not static.build(host,m,force=True)
    host=mujoco.MjModel.from_xml_string(xml())
    m,d=make(host)
    assert static.build(host,m,force=True)
    m.geom_size=wp.array(np.repeat(m.geom_size.numpy(),3,axis=0),dtype=wp.vec3)
    assert not static.available(m)
    assert not static.broadphase(m,d,cd.create_collision_context(d.naconmax))


def test_source_guard_and_small_scene_auto_fallback():
  with patch("src._mjwarp_static_collision.version",return_value="3.6.0"):
    assert not static.verified.__wrapped__()
  with wp.ScopedDevice("cpu"):
    host=mujoco.MjModel.from_xml_string(xml())
    m,_=make(host)
    assert not static.build(host,m)


def test_real_race_terrain_short_rollout_matches_sap():
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
  trajectories = []
  for mode in ("sap", "auto"):
    cfg = lgdog1_race_mgdp_env_cfg(with_perception=False)
    cfg.scene.num_envs = 10
    cfg.seed = 42
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    with compact_model_allocations(collision_mode=mode, active_kinematics=(mode == "auto")):
      env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      assert static.available(env.sim.wp_model) is (mode == "auto")
      env.reset()
      frames = []
      with torch.inference_mode():
        actions = torch.zeros((10, env.action_space.shape[-1]))
        for i in range(8):
          actions[:] = .03 * np.sin(i)
          _, reward, _, _, _ = env.step(actions)
          frames.append((env.sim.wp_data.qpos.numpy().copy(),
                         env.sim.wp_data.qvel.numpy().copy(), reward.numpy().copy(),
                         env.scene["depth_rays"].data.distances.numpy().copy()))
      trajectories.append(frames)
    finally:
      env.close()
  for baseline, fast in zip(*trajectories):
    for expected, actual in zip(baseline, fast):
      np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_static_bvh_cuda_forward_before_step_and_recapture():
  from mjlab.sim import Simulation, SimulationCfg
  if not wp.is_cuda_available():
    pytest.skip("requires real CUDA graph replay")
  extra = ''.join(f'<geom type="box" size=".1 .1 .1" pos="{i+10} 0 0"/>'
                  for i in range(256))
  with wp.ScopedDevice("cuda:0"), compact_model_allocations():
    sim = Simulation(3, SimulationCfg(nconmax=200, njmax=400),
                     mujoco.MjModel.from_xml_string(xml(extra=extra)), "cuda:0")
    assert static.available(sim.wp_model)
    if not sim.use_cuda_graph:
      pytest.skip("CUDA graph mempool support unavailable")
    for _ in range(2):
      sim.reset()
      sim.forward()
      wp.synchronize_device("cuda:0")
      assert int(sim.wp_data.nacon.numpy()[0]) > 0
      sim.step()
      sim.create_graph()
