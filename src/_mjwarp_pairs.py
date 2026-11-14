# Copyright 2025 The Newton Developers
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

"""Bounded-memory adaptation of MJWarp 3.5.0 io.put_model pair construction.

The filtering rules and ordering follow the upstream implementation. Build its
lookup in bounded chunks. Verified sparse mode never constructs a dense table
on the host or device; all retained pair/filter/sensor ordering is unchanged.
"""

import time

import numpy as np

PAIR_CHUNK_SIZE = 262_144


def build_collision_pairs(mjm, model, opt, types, chunk_size=PAIR_CHUNK_SIZE, sparse=False):
  if chunk_size < 1:
    raise ValueError("collision pair chunk size must be positive")
  n = mjm.ngeom
  count = n * (n - 1) // 2
  print(f"[memory] streaming collision pairs: geoms={n}, pairs={count}, "
        f"chunk={chunk_size}; {'avoided' if sparse else 'retained'} dense lookup={count*8/1024**3:.2f} GiB; "
        "skipping unused full geom-pair table", flush=True)
  started = last_report = time.monotonic()
  lookup = None if sparse else np.empty((count, 2), dtype=np.int32)
  if lookup is not None:
    lookup[:, 0] = -2
    lookup[:, 1] = -1

  def upper_tri_index(i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * n - i - 3)) // 2 + j - 1

  # Explicit contacts override all automatic collision filters, last entry wins.
  explicit = {upper_tri_index(int(mjm.pair_geom1[i]), int(mjm.pair_geom2[i])): i
              for i in range(mjm.npair)}
  explicit_ids = np.array(sorted(explicit), dtype=np.int64)
  explicit_values = np.array([explicit[i] for i in explicit_ids], dtype=np.int32)

  import mujoco
  collision_types = (mujoco.mjtSensor.mjSENS_GEOMDIST, mujoco.mjtSensor.mjSENS_GEOMNORMAL,
                     mujoco.mjtSensor.mjSENS_GEOMFROMTO)
  sensor_ids = np.flatnonzero(np.isin(mjm.sensor_type, collision_types))
  sensor_pairs, sensor_start = {}, []
  for sensor in sensor_ids:
    objid, refid = mjm.sensor_objid[sensor], mjm.sensor_refid[sensor]
    if mjm.sensor_objtype[sensor] == types.ObjType.BODY:
      id1, n1 = mjm.body_geomadr[objid], mjm.body_geomnum[objid]
    else:
      id1, n1 = objid, 1
    if mjm.sensor_reftype[sensor] == types.ObjType.BODY:
      id2, n2 = mjm.body_geomadr[refid], mjm.body_geomnum[refid]
    else:
      id2, n2 = refid, 1
    for geom1 in range(id1, id1 + n1):
      for geom2 in range(id2, id2 + n2):
        pair = upper_tri_index(geom1, geom2)
        if pair not in sensor_pairs:
          sensor_pairs[pair] = len(sensor_pairs)
        sensor_start.append(sensor_pairs[pair])
  model.sensor_collision_start_adr = np.array(sensor_start, dtype=np.int32)
  sensor_pair_ids = np.array(sorted(sensor_pairs), dtype=np.int64)
  sensor_pair_values = np.array([sensor_pairs[i] for i in sensor_pair_ids], dtype=np.int32)

  # Triangular row starts permit exact flat-index -> (i,j) conversion per chunk.
  rows = np.arange(n, dtype=np.int64)
  row_starts = rows * (2*n - rows - 1) // 2
  filterparent = not (mjm.opt.disableflags & types.DisableBit.FILTERPARENT)
  geom_weld = mjm.body_weldid[mjm.geom_bodyid]
  weld_parent = mjm.body_weldid[mjm.body_parentid[geom_weld]]
  # A terrain-heavy scene has thousands of geoms welded to world and only a
  # few robot geoms. Automatic static-static contacts are filtered upstream.
  # Enumerate pairs incident to a moving weld directly; retain ALL explicit
  # contacts and distance/normal/fromto sensor pairs, including static-static.
  candidates = None
  dynamic = np.flatnonzero(geom_weld != 0)
  if sparse and len(dynamic) * 4 < n:
    static_ids = np.flatnonzero(geom_weld == 0)
    parts = [explicit_ids, sensor_pair_ids]
    for geom in dynamic:
      left = np.arange(geom, dtype=np.int64)
      parts.append(left * (2*n-left-3)//2 + geom-1)
      right = static_ids[np.searchsorted(static_ids, geom, side="right"):]
      parts.append(geom * (2*n-geom-3)//2 + right-1)
    candidates = np.unique(np.concatenate(parts))
    print(f"[memory] static terrain pair pruning: candidates={len(candidates)} "
          f"of {count}; explicit contacts/sensor pairs retained", flush=True)
  work_count = count if candidates is None else len(candidates)
  type_count = len(types.GeomType)
  histogram = np.zeros(type_count * (type_count + 1) // 2, dtype=np.int64)
  geom_chunks, id_chunks = [], []
  sensor_collision_count = 0
  for offset in range(0, work_count, chunk_size):
    end = min(offset + chunk_size, work_count)
    flat = (np.arange(offset, end, dtype=np.int64) if candidates is None
            else candidates[offset:end])
    start, stop = int(flat[0]), int(flat[-1])+1
    geom1 = np.searchsorted(row_starts, flat, side="right") - 1
    geom2 = flat - row_starts[geom1] + geom1 + 1
    body1, body2 = mjm.geom_bodyid[geom1], mjm.geom_bodyid[geom2]
    weld1, weld2 = geom_weld[geom1], geom_weld[geom2]
    parent_child = (filterparent & (weld1 != 0) & (weld2 != 0) &
                    ((weld1 == weld_parent[geom2]) | (weld2 == weld_parent[geom1])))
    mask = ((mjm.geom_contype[geom1] & mjm.geom_conaffinity[geom2]) |
            (mjm.geom_contype[geom2] & mjm.geom_conaffinity[geom1])) != 0
    exclude = np.isin((body1 << 16) + body2, mjm.exclude_signature)
    view = np.empty((len(flat), 2), dtype=np.int32) if sparse else lookup[start:stop]
    view[:, 1] = -1
    view[:, 0] = np.where(mask & (weld1 != weld2) & ~parent_child & ~exclude, -1, -2)
    left, right = np.searchsorted(explicit_ids, (start, stop))
    view[np.searchsorted(flat, explicit_ids[left:right]), 0] = explicit_values[left:right]
    left, right = np.searchsorted(sensor_pair_ids, (start, stop))
    view[np.searchsorted(flat, sensor_pair_ids[left:right]), 1] = sensor_pair_values[left:right]
    sensor_mask = view[:, 1] >= 0
    sensor_collision_count += np.count_nonzero(sensor_mask)
    include = (view[:, 0] > -2) | sensor_mask
    selected1, selected2 = geom1[include], geom2[include]
    if len(selected1):
      geom_chunks.append(np.column_stack((selected1, selected2)).astype(np.int32))
      id_chunks.append(view[include])
      a, b = mjm.geom_type[selected1], mjm.geom_type[selected2]
      lo, hi = np.minimum(a, b), np.maximum(a, b)
      type_ids = lo * (2*type_count - lo - 1) // 2 + hi
      histogram += np.bincount(type_ids, minlength=len(histogram))
    if time.monotonic() - last_report >= 10:
      print(f"[memory] collision pair filtering: {end}/{work_count} candidates "
            f"({100*end/work_count:.0f}%)", flush=True)
      last_report = time.monotonic()

  empty = np.empty((0, 2), dtype=np.int32)
  model.nxn_geom_pair = empty
  model.nxn_geom_pair_filtered = np.concatenate(geom_chunks) if geom_chunks else empty
  model.nxn_pairid_filtered = np.concatenate(id_chunks) if id_chunks else empty
  model.nsensorcollision = int(sensor_collision_count)
  model.geom_pair_type_count = tuple(histogram)
  active = len(model.nxn_geom_pair_filtered)
  if active < 250_000:
    opt.broadphase = types.BroadphaseType.NXN
  elif n < 1000:
    opt.broadphase = types.BroadphaseType.SAP_TILE
  else:
    opt.broadphase = types.BroadphaseType.SAP_SEGMENTED

  if sparse:
    from src._mjwarp_sap import set_sparse_pairs
    pairs = model.nxn_geom_pair_filtered
    keys = pairs[:, 0].astype(np.int64) * n + pairs[:, 1].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    set_sparse_pairs(model, keys[order], model.nxn_pairid_filtered[order])
    # 稠密表置换成 1 项的占位表，避免上传 2.89 GiB；真实查询走稀疏二分。
    model.nxn_pairid = np.zeros((1, 2), dtype=np.int32)
    print(f"[memory] sparse pair lookup: {active} sorted keys "
          f"({active*16/1024**2:.1f} MiB) replaces {count*8/1024**3:.2f} GiB dense table",
          flush=True)
  else:
    model.nxn_pairid = lookup
  print(f"[memory] collision pairs ready: active={active}, "
        f"elapsed={time.monotonic()-started:.2f}s; uploading model without full pair indices", flush=True)
  return sensor_ids
