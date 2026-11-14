# Copyright 2026 The Newton Developers
# Copyright 2026 LgDog contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Prune threads which native geom kinematics would immediately return from.

Static geom poses are already initialized by native make_data. Keep all native
body/site kernels and all pose math; index only geoms satisfying its predicate.
"""
import ast
import hashlib
import inspect
from functools import lru_cache
from importlib.metadata import version

import numpy as np
import warp as wp
from mujoco_warp._src import math, smooth

_HASHES = {"kinematics": "64d69ce6c44956c062c73ccb5b547605202d0313c7de2b3184fc081d3962a0be",
           "_geom_local_to_global": "b448b983580b8e569daecee5dcd26aa1cf72e68c51e50da5d42c19a93597d691"}
_TOPOLOGY = ("geom_bodyid", "body_weldid", "body_mocapid", "body_rootid")
_ORIGINAL = smooth.kinematics


@wp.kernel
def _active_geom_local_to_global(
  active: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int), body_weldid: wp.array(dtype=int),
  body_mocapid: wp.array(dtype=int), geom_bodyid: wp.array(dtype=int),
  geom_pos: wp.array2d(dtype=wp.vec3), geom_quat: wp.array2d(dtype=wp.quat),
  xpos_in: wp.array2d(dtype=wp.vec3), xquat_in: wp.array2d(dtype=wp.quat),
  geom_xpos_out: wp.array2d(dtype=wp.vec3), geom_xmat_out: wp.array2d(dtype=wp.mat33),
):
  worldid, index = wp.tid()
  geomid = active[index]
  bodyid = geom_bodyid[geomid]
  if body_weldid[bodyid] == 0 and body_mocapid[body_rootid[bodyid]] == -1:
    return
  xpos = xpos_in[worldid, bodyid]
  xquat = xquat_in[worldid, bodyid]
  geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)
  geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))


@lru_cache(maxsize=1)
def factory():
  if version("mujoco-warp") != "3.5.0":
    return None
  try:
    for name, expected in _HASHES.items():
      fn = _ORIGINAL if name == "kinematics" else getattr(smooth, name)
      tree = ast.parse(inspect.getsource(getattr(fn, "func", fn)))
      canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
      if hashlib.sha256(canonical.encode()).hexdigest() != expected:
        return None
    tree = ast.parse(inspect.getsource(_ORIGINAL))
    tree.body[0].name = "active_kinematics"
    matched = 0
    for node in ast.walk(tree):
      if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0],ast.Name)
          and node.args[0].id == "_geom_local_to_global"):
        node.args[0].id = "_active_geom_local_to_global"
        for kw in node.keywords:
          if kw.arg == "dim":
            kw.value = ast.parse("(d.nworld, m._lgdog_active_geoms.size)", mode="eval").body
          if kw.arg == "inputs":
            kw.value.elts.insert(0, ast.parse("m._lgdog_active_geoms", mode="eval").body)
        matched += 1
    if matched != 1:
      return None
    namespace = dict(vars(smooth), _active_geom_local_to_global=_active_geom_local_to_global)
    exec(compile(ast.fix_missing_locations(tree), "<lgdog-active-kinematics>", "exec"), namespace)
    return namespace["active_kinematics"]
  except (OSError, TypeError, SyntaxError, AttributeError):
    return None


def _signature(model):
  return tuple((name, getattr(model,name).ptr, getattr(model,name).shape) for name in _TOPOLOGY)


def available(model):
  return (hasattr(model,"_lgdog_active_geoms") and
          model._lgdog_active_topology == _signature(model))


def build(host, model):
  adapted = factory()
  if adapted is None or getattr(host,"ngeom",0) < 256:
    return False
  body = np.asarray(host.geom_bodyid)
  active = np.flatnonzero((np.asarray(host.body_weldid)[body] != 0) |
                         (np.asarray(host.body_mocapid)[np.asarray(host.body_rootid)[body]] != -1)).astype(np.int32)
  if len(active) * 4 > host.ngeom or not len(active):
    return False
  from src._mjwarp_sap import _require_uncaptured_allocation
  _require_uncaptured_allocation(model.geom_pos.device,"active geom kinematics")
  model._lgdog_active_geoms = wp.array(active,dtype=int,device=model.geom_pos.device)
  model._lgdog_active_topology = _signature(model)
  if not getattr(smooth.kinematics,"_lgdog_active_dispatch",False):
    def dispatch(m,d):
      return adapted(m,d) if available(m) else _ORIGINAL(m,d)
    dispatch._lgdog_active_dispatch = True
    smooth.kinematics = dispatch
  print(f"[speed] active geom kinematics: {len(active)}/{host.ngeom} geom threads per world",flush=True)
  return True
