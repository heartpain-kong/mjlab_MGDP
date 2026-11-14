# Copyright 2026 LgDog contributors
# Licensed under the Apache License, Version 2.0.
"""Shared static-sphere BVH broadphase for terrain-dominated MJWarp scenes.

Query only moving geoms against one static tree, handle infinite planes
separately, and check the small moving-moving pair list. Final eligibility and
OBB/sphere/plane filters are the verified native filters; narrowphase is intact.
"""

import ast
import hashlib
import inspect
from functools import lru_cache
from importlib.metadata import version

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import collision_driver as cd
from mujoco_warp._src.types import BroadphaseFilter, BroadphaseType

_HASHES = {
  "_broadphase_filter": "12814f73f2f2cc016f8ba0c17163e23e7daed178b303fe2f5007cdf26bf33372",
  "_sphere_filter": "1795247b1dd4d9930a34ab311aa934d011a5eb34220bfb974d1ed19fd273e843",
  "_plane_filter": "88289bec678713c88136c11d4585873b169904115a8174a82295fcf184b6edef",
}
_GEOMETRY_FIELDS = ("geom_aabb", "geom_rbound", "geom_margin", "geom_size",
                    "geom_pos", "geom_quat", "body_pos", "body_quat")


@lru_cache(maxsize=1)
def verified():
  try:
    if version("mujoco-warp") != "3.5.0":
      return False
    for name, expected in _HASHES.items():
      fn = getattr(cd, name)
      tree = ast.parse(inspect.getsource(getattr(fn, "func", fn)))
      canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
      if hashlib.sha256(canonical.encode()).hexdigest() != expected:
        return False
    return True
  except (OSError, TypeError, SyntaxError, AttributeError):
    return False


def _signature(model):
  return tuple((name, array.ptr, array.shape) for name in _GEOMETRY_FIELDS
               if (array := getattr(model, name, None)) is not None)


def build(mjm, model, force=False):
  """Build outside capture. Unsupported cases keep the existing SAP path."""
  from src._mjwarp_sap import sparse_available, _require_uncaptured_allocation
  if (not verified() or not sparse_available(model) or mjm.nmocap or mjm.nflex
      or model.nsensorcollision or not model.opt.broadphase_filter & BroadphaseFilter.SPHERE):
    return False
  if any(getattr(model, name).shape[0] != 1 for name in _GEOMETRY_FIELDS
         if getattr(model, name, None) is not None):
    return False
  device = model.geom_rbound.device
  _require_uncaptured_allocation(device, "static collision BVH")
  weld = np.asarray(mjm.body_weldid)[np.asarray(mjm.geom_bodyid)]
  keys = model._lgdog_sparse_pairs["np_keys"]
  values = model._lgdog_sparse_pairs["np_vals"]
  if np.any(values[:, 0] >= 0):
    return False  # Explicit contact pairs retain the generic broadphase semantics.
  first, second = keys // mjm.ngeom, keys % mjm.ngeom
  participating = np.unique(np.concatenate((first, second))).astype(np.int32)
  moving = participating[weld[participating] != 0]
  static = participating[weld[participating] == 0]
  if not len(moving) or len(moving) > 128 or len(moving) * mjm.ngeom > 8_000_000:
    return False
  if not force and (len(static) < 256 or len(static) < 4 * len(moving)):
    return False
  radius = model.geom_rbound.numpy()[0]
  margin = model.geom_margin.numpy()[0]
  if (not np.isfinite(radius).all() or not np.isfinite(margin).all()
      or (margin < 0).any() or (radius[moving] <= 0).any()):
    return False
  if np.any((weld[first] == 0) & (weld[second] == 0)):
    return False  # Explicit static-static contacts require the generic path.
  local = np.full(mjm.ngeom, -1, dtype=np.int32)
  local[moving] = np.arange(len(moving), dtype=np.int32)
  lookup = np.empty((len(moving), mjm.ngeom, 2), dtype=np.int32)
  lookup[:] = (-2, -1)
  mask = local[first] >= 0
  lookup[local[first[mask]], second[mask]] = values[mask]
  mask = local[second] >= 0
  lookup[local[second[mask]], first[mask]] = values[mask]
  both = (local[first] >= 0) & (local[second] >= 0)
  dynamic_pairs = np.column_stack((local[first[both]], first[both], second[both])).astype(np.int32)
  planes = static[radius[static] == 0]
  finite = static[radius[static] > 0]
  host_data = mujoco.MjData(mjm)
  mujoco.mj_kinematics(mjm, host_data)
  positions = host_data.geom_xpos[finite].astype(np.float32)
  extent = (radius[finite] + margin[finite])[:, None]
  # Conservative rounding only affects candidate traversal; native filters
  # below decide exactly which candidates reach narrowphase.
  padding = 2e-5 * (1 + np.abs(positions) + extent)
  lower = wp.array(positions - extent - padding, dtype=wp.vec3, device=device)
  upper = wp.array(positions + extent + padding, dtype=wp.vec3, device=device)
  tree = wp.Bvh(lower, upper, constructor="sah") if len(finite) else None
  model._lgdog_static_collision = {
    "signature": _signature(model), "tree": tree, "lower": lower, "upper": upper,
    "static_ids": wp.array(finite, dtype=int, device=device),
    "moving_ids": wp.array(moving, dtype=int, device=device),
    "plane_ids": wp.array(planes, dtype=int, device=device),
    "moving_pairs": wp.array(dynamic_pairs, dtype=wp.vec3i, device=device),
    "lookup": wp.array(lookup, dtype=wp.vec2i, device=device),
  }
  # Route large NXN terrain scenes through the same guarded dispatch too.
  model.opt.broadphase = BroadphaseType.SAP_SEGMENTED
  print(f"[speed] shared static collision BVH: static={len(finite)}, planes={len(planes)}, "
        f"moving={len(moving)}, moving-pairs={len(dynamic_pairs)}; "
        "per-world terrain sorting and static-static sweeps bypassed", flush=True)
  return True


def available(model):
  state = getattr(model, "_lgdog_static_collision", None)
  return bool(state is not None and state["signature"] == _signature(model)
              and model.opt.broadphase_filter & BroadphaseFilter.SPHERE)


@lru_cache(maxsize=None)
def _query_kernel(flags, naabb, nrbound, nmargin):
  from src._mjwarp_sap import _add_pair_value

  @wp.func
  def emit(geom_type: wp.array(dtype=int), aabb: wp.array3d(dtype=wp.vec3),
           rbound: wp.array2d(dtype=float), margin: wp.array2d(dtype=float),
           xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
           world: int, first: int, second: int, value: wp.vec2i, capacity: int,
           direction: wp.vec3,
           count: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i),
           pairids: wp.array(dtype=wp.vec2i), worlds: wp.array(dtype=int)):
    if value[0] >= -1 or value[1] >= 0:
      if (wp.static(cd._broadphase_filter(flags, naabb, nrbound, nmargin))(
          aabb, rbound, margin, xpos, xmat, first, second, world) or value[1] >= 0):
        # Preserve SAP's projected-lower ordering, including stable geom-ID ties.
        p1 = wp.dot(direction, xpos[world, first]) - (rbound[world % rbound.shape[0], first] + margin[world % margin.shape[0], first])
        p2 = wp.dot(direction, xpos[world, second]) - (rbound[world % rbound.shape[0], second] + margin[world % margin.shape[0], second])
        if rbound[world % rbound.shape[0], first] == 0.0:
          p1 = -1e10
        if rbound[world % rbound.shape[0], second] == 0.0:
          p2 = -1e10
        lo, hi = first, second
        if p2 < p1 or (p1 == p2 and second < first):
          lo, hi = second, first
        _add_pair_value(geom_type, capacity, lo, hi, world, value,
                        count, pairs, pairids, worlds)

  @wp.kernel(module="unique", enable_backward=False)
  def query(geom_type: wp.array(dtype=int), aabb: wp.array3d(dtype=wp.vec3),
            rbound: wp.array2d(dtype=float), margin: wp.array2d(dtype=float),
            xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
            static_ids: wp.array(dtype=int), moving: wp.array(dtype=int),
            planes: wp.array(dtype=int), moving_pairs: wp.array(dtype=wp.vec3i),
            lookup: wp.array2d(dtype=wp.vec2i), tree_id: wp.uint64, capacity: int, direction: wp.vec3,
            count: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i),
            pairids: wp.array(dtype=wp.vec2i), worlds: wp.array(dtype=int)):
    world, work = wp.tid()
    if work >= moving.shape[0]:
      item = moving_pairs[work - moving.shape[0]]
      emit(geom_type, aabb, rbound, margin, xpos, xmat, world,
           item[1], item[2], lookup[item[0], item[2]], capacity, direction, count, pairs, pairids, worlds)
    else:
      geom = moving[work]
      pos = xpos[world, geom]
      radius = rbound[world % rbound.shape[0], geom] + margin[world % margin.shape[0], geom]
      if not wp.isnan(pos[0]) and not wp.isnan(pos[1]) and not wp.isnan(pos[2]):
        pad = 2e-5 * (1.0 + wp.max(wp.abs(pos[0]), wp.max(wp.abs(pos[1]), wp.abs(pos[2]))) + radius)
        extent = wp.vec3(radius + pad, radius + pad, radius + pad)
        if static_ids.shape[0] > 0:
          traversal = wp.bvh_query_aabb(tree_id, pos - extent, pos + extent)
          leaf = int(0)
          while wp.bvh_query_next(traversal, leaf):
            other = static_ids[leaf]
            emit(geom_type, aabb, rbound, margin, xpos, xmat, world, geom, other,
                 lookup[work, other], capacity, direction, count, pairs, pairids, worlds)
        for i in range(planes.shape[0]):
          other = planes[i]
          emit(geom_type, aabb, rbound, margin, xpos, xmat, world, geom, other,
               lookup[work, other], capacity, direction, count, pairs, pairids, worlds)

  return query


def broadphase(model, data, context):
  """Returns False when runtime geometry changes require the generic path."""
  if not available(model):
    return False
  s = model._lgdog_static_collision
  wp.launch(_query_kernel(int(model.opt.broadphase_filter), model.geom_aabb.shape[0],
                          model.geom_rbound.shape[0], model.geom_margin.shape[0]),
    dim=(data.nworld, s["moving_ids"].size + s["moving_pairs"].size),
    inputs=[model.geom_type, model.geom_aabb, model.geom_rbound, model.geom_margin,
            data.geom_xpos, data.geom_xmat, s["static_ids"], s["moving_ids"],
            s["plane_ids"], s["moving_pairs"], s["lookup"],
            s["tree"].id if s["tree"] is not None else wp.uint64(0), data.naconmax,
            wp.normalize(wp.vec3(0.5935, 0.7790, 0.1235))],
    outputs=[data.ncollision, context.collision_pair, context.collision_pairid, context.collision_worldid],
    device=data.geom_xpos.device)
  return True
