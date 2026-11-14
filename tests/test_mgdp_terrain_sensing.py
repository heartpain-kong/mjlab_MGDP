"""Compare shared terrain sensing to native BVHs across distinct worlds."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mujoco_warp._src import bvh, io
from mujoco_warp._src.types import vec6

from src._mjwarp_terrain_sensing import context_factory
from src.sim_memory import compact_model_allocations


XML = """<mujoco>
  <asset>
    <mesh name="tetra" vertex="0 0 0  .3 0 0  0 .3 0  0 0 .3"/>
    <hfield name="hill" nrow="3" ncol="3" size="1 1 .3 .1"
      elevation="0 0 0  0 1 0  0 0 0"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 .1" group="0"/>
    <body pos="1 0 .4"><geom type="box" size=".3 .2 .1" group="0"/></body>
    <geom type="mesh" mesh="tetra" pos="2 0 .2" group="0"/>
    <geom type="hfield" hfield="hill" pos="-2 0 0" group="0"/>
    <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2" group="1"/></body>
    <camera name="cam" pos="0 -3 2" xyaxes="1 0 0 0 .6 .8" resolution="4 4"/>
  </worldbody>
</mujoco>"""


def make_context(model, worlds, groups=(0,), cameras=False):
  return mjwarp.create_render_context(model, nworld=worlds, enabled_geom_groups=list(groups),
                                      use_textures=False, cam_active=[cameras])


def test_shared_leaves_avoid_full_allocation_and_match_native_rays_after_world_motion():
  with wp.ScopedDevice("cpu"):
    model = mujoco.MjModel.from_xml_string(XML)
    worlds = 4
    native = make_context(model, worlds)
    zeros = wp.zeros
    allocations = []
    def record(shape, *args, **kwargs):
      allocations.append((shape, kwargs.get("dtype")))
      return zeros(shape, *args, **kwargs)
    with compact_model_allocations(), patch.object(wp, "zeros", side_effect=record):
      shared = make_context(model, worlds)
    assert shared._lgdog_shared_terrain
    assert shared.lower.shape == (shared.bvh_ngeom,)
    assert (worlds * shared.bvh_ngeom, wp.vec3) not in allocations
    assert (worlds * shared.bvh_ngeom, int) not in allocations
    assert shared.group_root.shape == (worlds,)
    wm = mjwarp.put_model(model)
    wd = mjwarp.make_data(model, nworld=worlds, nconmax=40, njmax=100)
    qpos = wd.qpos.numpy()
    qpos[:, 0] = [-.4, 0., .7, 1.]
    wd.qpos.assign(qpos)
    mjwarp.forward(wm, wd)
    bvh.refit_scene_bvh(wm, wd, native)
    before = shared.lower.numpy().copy()
    bvh.refit_scene_bvh(wm, wd, shared)
    np.testing.assert_array_equal(shared.lower.numpy(), before)
    np.testing.assert_allclose(before, native.lower.numpy()[:shared.bvh_ngeom])
    starts = np.array([[0,0,3], [1,0,3], [2.05,.05,3], [-2,0,3], [9,9,3]], dtype=np.float32)
    origins = np.broadcast_to(starts, (worlds, len(starts), 3)).copy()
    origins[:, :, 1] += np.arange(worlds)[:, None] * .02
    pnt = wp.array(origins, dtype=wp.vec3)
    vec = wp.array(np.broadcast_to([0,0,-1], origins.shape).astype(np.float32).copy(), dtype=wp.vec3)
    def cast(rc):
      output = (wp.empty((worlds,5), dtype=float), wp.empty((worlds,5), dtype=int),
                wp.empty((worlds,5), dtype=wp.vec3))
      mjwarp.rays(wm, wd, pnt, vec, vec6(1,0,0,0,0,0), True,
                  wp.array([-1]*5, dtype=int), *output, rc=rc)
      return [x.numpy() for x in output]
    expected, actual = cast(native), cast(shared)
    for a, e in zip(actual, expected):
      np.testing.assert_allclose(a, e, atol=1e-6, rtol=1e-6)
    assert np.all(actual[0][:, :4] > 0)


def test_moving_geometry_and_camera_contexts_keep_independent_trees():
  with wp.ScopedDevice("cpu"), compact_model_allocations():
    model = mujoco.MjModel.from_xml_string(XML)
    for groups, cameras in (((0,1), False), ((0,), True)):
      rc = make_context(model, 3, groups, cameras)
      assert not rc._lgdog_shared_terrain
      assert rc.lower.shape == (3 * rc.bvh_ngeom,)
    mocap = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body mocap="true"><geom type="box" size=".1 .1 .1" group="0"/></body>
    </worldbody></mujoco>''')
    rc = mjwarp.create_render_context(mocap, nworld=3, enabled_geom_groups=[0], use_textures=False)
    assert not rc._lgdog_shared_terrain


def test_new_mgdp_configs_use_only_terrain_and_keep_all_obstacle_types():
  from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
  cfg = lgdog1_race_mgdp_env_cfg(with_perception=False)
  assert len(cfg.scene.terrain.terrain_generator.sub_terrains) == 10
  rays = [s for s in cfg.scene.sensors if hasattr(s, "include_geom_groups")]
  assert {s.name for s in rays} == {"terrain_scan", "depth_rays"}
  assert all(s.include_geom_groups == (0,) for s in rays)
  legacy = lgdog1_race_mgdp_env_cfg(with_perception=False, terrain_only_sensing=False)
  assert {s.name: s.include_geom_groups for s in legacy.scene.sensors
          if hasattr(s, "include_geom_groups")} == {"terrain_scan": (0,1,2), "depth_rays": None}


def test_unverified_context_source_is_rejected():
  with patch("src._mjwarp_terrain_sensing._matches", return_value=False):
    assert context_factory.__wrapped__() is None


def test_masked_gru_resets_do_not_synchronize_through_tensor_any():
  from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature
  class Module:
    def feature_batched(self, stack, hidden, size):
      torch.testing.assert_close(stack[0, 0], stack[0, 1])
      assert torch.count_nonzero(hidden[:, 0]) == 0
      assert torch.all(hidden[:, 1] == 1)
      return torch.zeros(2, 64), hidden
  env = NS(device="cpu", episode_length_buf=torch.tensor([0, 5]),
           scene={"depth_rays": NS(data=NS(distances=torch.ones(2, 256)))},
           _mgdp_perception_state={"module": Module(), "prev": torch.zeros(2,1,16,16),
                                   "hidden": torch.ones(1,2,64)})
  # Old `if reset.any()` forced a device-to-host sync every observation.
  with patch.object(torch.Tensor, "any", side_effect=AssertionError("host synchronization")):
    mgdp_depth_feature(env)
