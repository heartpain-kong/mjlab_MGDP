# Copyright 2026 The Newton Developers
# Copyright 2026 LgDog contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verified ray-only terrain contexts with one static BVH, including at allocation.

The native geometry intersection routines are retained. Only the leaf address
changes: a shared tree's leaf is already local, independent of the ray's world.
RGB/camera contexts, moving geoms, mocap and flex retain per-world BVHs.
"""

import ast
import hashlib
import inspect
from functools import lru_cache
from importlib.metadata import version
from types import FunctionType

import warp as wp
from mujoco_warp._src.ray import _ray_geom_mesh_bvh
from mujoco_warp._src.types import MJ_MAXVAL, vec6

from src._mjwarp_bvh import SHARING_ALLOWED, static_terrain_only

_HASHES = {
  "context": "6c2540b8025ab5a90cffa618da6eff0c3d3b06c96fcd1d0bbbb8c1ff10322985",
  "rays": "32b9e66e472cfebdf5951e16eceb326a5b9c1581a9b238c273b02b38577ee9e7",
  "kernel": "12172e328b4a15cf01a05a2a7abf14f1e78ed235743d6b21dc5509dd58ac07a8",
  "refit": "4fef107c5b86b496963c6c55f80ae3d3e45a67d3482810a55bfdbbe43b9b1a04",
}


def _matches(function, name):
  tree = ast.parse(inspect.getsource(function))
  canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
  return hashlib.sha256(canonical.encode()).hexdigest() == _HASHES[name]


def _can_share(model, ids, active_cameras):
  return SHARING_ALLOWED.get() and active_cameras == 0 and static_terrain_only(model, ids)


@lru_cache(maxsize=1)
def context_factory():
  """Adapt only the three leaf allocation sizes in the verified factory."""
  from mujoco_warp._src import bvh, io, ray
  try:
    if version("mujoco-warp") != "3.5.0" or not all((
      _matches(io.create_render_context, "context"), _matches(ray.rays, "rays"),
      _matches(ray._ray_bvh.func, "kernel"), _matches(bvh.refit_scene_bvh, "refit"),
    )):
      return None
    tree = ast.parse(inspect.getsource(io.create_render_context))
    body = tree.body[0].body
    index = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "rc" for t in node.targets))
    rc_call = body[index].value
    for keyword in rc_call.keywords:
      if keyword.arg in ("lower", "upper", "group"):
        keyword.value.args[0] = ast.parse("_leaf_worlds * bvh_ngeom", mode="eval").body
    body[index:index] = ast.parse(
      "_shared = _lgdog_can_share(mjm, geom_enabled_idx, ncam)\n"
      "_leaf_worlds = 1 if _shared else nworld\n").body
    body.insert(index + 3, ast.parse("rc._lgdog_shared_terrain = _shared").body[0])
    namespace = dict(io.create_render_context.__globals__, _lgdog_can_share=_can_share)
    exec(compile(ast.fix_missing_locations(tree), "<lgdog-terrain-context>", "exec"), namespace)
    return namespace["create_render_context"]
  except (OSError, TypeError, SyntaxError, AttributeError, ValueError):
    return None


def install_runtime():
  """Persistent dispatch for MJLab's public rays API, selected per context.

  Other contexts keep native kernels. No IDs of dead contexts are retained.
  """
  from mujoco_warp._src import bvh, ray
  if getattr(ray, "_lgdog_terrain_runtime", False):
    return
  original_rays = ray.rays
  original_refit = bvh.refit_scene_bvh
  shared_rays = FunctionType(original_rays.__code__,
    dict(original_rays.__globals__, _ray_bvh=_ray_shared_terrain),
    original_rays.__name__, original_rays.__defaults__, original_rays.__closure__)

  def dispatch(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, dist, geomid, normal, rc=None):
    fn = shared_rays if getattr(rc, "_lgdog_shared_terrain", False) else original_rays
    return fn(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, dist, geomid, normal, rc)

  def refit(m, d, rc):
    if getattr(rc, "_lgdog_shared_terrain", False):
      return
    return original_refit(m, d, rc)

  import mujoco_warp
  from mjlab.sensor import raycast_sensor
  ray.rays = dispatch
  mujoco_warp.rays = dispatch
  # RayCastSensor 1.2.0 calls mjwarp.rays, but keep direct imports safe too.
  if getattr(raycast_sensor, "rays", None) is original_rays:
    raycast_sensor.rays = dispatch
  bvh.refit_scene_bvh = refit
  ray._lgdog_terrain_runtime = True


# Intersection kernel adapted from mujoco-warp 3.5.0 (Apache-2.0).
# Copyright 2026 The Newton Developers.
@wp.kernel(module="unique", enable_backward=False)
def _ray_shared_terrain(
  # Model:
  ngeom: int,
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array(dtype=int),
  bvh_id: wp.uint64,
  group_root: wp.array(dtype=int),
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  # Out:
  dist_out: wp.array2d(dtype=float),
  geomid_out: wp.array2d(dtype=int),
  normal_out: wp.array2d(dtype=wp.vec3),
):
  worldid, rayid = wp.tid()

  ray_origin = pnt[worldid, rayid]
  ray_dir = vec[worldid, rayid]
  body_exclude = bodyexclude[rayid]

  min_dist = float(MJ_MAXVAL)
  min_geomid = int(-1)
  min_normal = wp.vec3()

  query = wp.bvh_query_ray(bvh_id, ray_origin, ray_dir, group_root[worldid])
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, min_dist):
    bvh_local = bounds_nr
    geomid = enabled_geom_ids[bvh_local]

    dist, normal = _ray_geom_mesh_bvh(
      body_weldid,
      geom_type,
      geom_bodyid,
      geom_dataid,
      geom_matid,
      geom_group,
      geom_size,
      geom_rgba,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      pnt[worldid, rayid],
      vec[worldid, rayid],
      geomgroup,
      flg_static,
      body_exclude,
      geomid,
      mesh_bvh_id,
      hfield_bvh_id,
      min_dist,
    )

    if dist >= 0.0 and dist < min_dist:
      min_dist = dist
      min_geomid = geomid
      min_normal = normal

  if min_dist >= MJ_MAXVAL:
    dist_out[worldid, rayid] = -1.0
  else:
    dist_out[worldid, rayid] = min_dist
  geomid_out[worldid, rayid] = min_geomid
  normal_out[worldid, rayid] = min_normal
