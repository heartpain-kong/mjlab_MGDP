"""int64 版 SAP 宽相：绕开 MuJoCo-Warp 3.5.0 里 int32 累计和的上限。

背景（详见 doc/MGDP_CONTACT_PROBE.md）：
`mujoco_warp._src.collision_driver.sap_broadphase` 用一张**全局 int32 前缀和**
（`wp.utils.array_scan`，只支持 int32/float32）来定位"第几个世界的第几对 geom"。
前缀和的总量 = 所有世界里 AABB 重叠对数 ≈ 世界数 × geom 数 × 平均重叠数。
当"世界数 × geom 数"超过约 2^24（1.6e7）时，前缀和溢出成负数，
`sap_broadphase` 里的 `while worldgeomid < nworkpackages` 直接不执行 ——
**整个场景一个碰撞对都不产出**，接触全为 0，而且不报错。

本模块保留原有的投影 / 分段排序 / range 三个内核，只把
"前缀和 + 扫描"换成 int64（torch.cumsum on int64），并把扫描内核的索引改成 int64。
显存代价 = 世界数 × geom 数 × 8 B（4096×27838 ≈ 0.9 GB），可接受。
"""

from __future__ import annotations

import hashlib
import inspect

import warp as wp

from mujoco_warp._src import collision_driver as _cd
from mujoco_warp._src.math import upper_tri_index
from mujoco_warp._src.types import BroadphaseType

# 只在确认过源码版本时启用（版本守卫与 sim_memory 里其它补丁一致）。
SAP_BROADPHASE_SHA256 = "1e6e965c9a45b4d4d8334d3c26480cd3192b27f58d4a7d290626100536f729bd"

# 超过这个"世界数 × geom 数"就切到 int64 路径；低于它用原生实现（行为完全一致）。
INT64_THRESHOLD = 8_000_000

_cd._int64_sap_state = getattr(_cd, "_int64_sap_state", {"installed": False, "original": None})


@wp.func
def _binary_search_i64(
  values: wp.array(dtype=wp.int64), value: wp.int64, lower: int, upper: int
) -> int:
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1
  return upper


def _sap_sweep_i64(
  opt_broadphase_filter: int, ngeom_aabb: int, ngeom_rbound: int, ngeom_margin: int
):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ngeom: int,
    geom_type: wp.array(dtype=int),
    geom_aabb: wp.array3d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_margin: wp.array2d(dtype=float),
    nxn_pairid: wp.array(dtype=wp.vec2i),
    # Data in:
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    nworld_in: int,
    naconmax_in: int,
    # In:
    sort_index_in: wp.array2d(dtype=int),
    cumulative_sum_in: wp.array(dtype=wp.int64),
    nsweep_in: int,
    # Data out:
    ncollision_out: wp.array(dtype=int),
    # Out:
    collision_pair_out: wp.array(dtype=wp.vec2i),
    collision_pairid_out: wp.array(dtype=wp.vec2i),
    collision_worldid_out: wp.array(dtype=int),
  ):
    worldgeomid = wp.tid()

    nworldgeom = nworld_in * ngeom
    nworkpackages = cumulative_sum_in[nworldgeom - 1]

    while wp.int64(worldgeomid) < nworkpackages:
      # binary search to find current and next geom pair indices
      i = _binary_search_i64(cumulative_sum_in, wp.int64(worldgeomid), 0, nworldgeom)
      j = wp.int64(i) + wp.int64(worldgeomid) + wp.int64(1)

      if i > 0:
        j -= cumulative_sum_in[i - 1]

      worldid = wp.int32(wp.int64(i) // wp.int64(ngeom))
      gi = wp.int32(wp.int64(i) % wp.int64(ngeom))
      gj = wp.int32(j % wp.int64(ngeom))

      geom1 = sort_index_in[worldid, gi]
      geom2 = sort_index_in[worldid, gj]

      if geom2 < geom1:
        idx = upper_tri_index(ngeom, geom2, geom1)
      else:
        idx = upper_tri_index(ngeom, geom1, geom2)

      worldgeomid += nsweep_in
      pairid = nxn_pairid[idx]
      if pairid[0] < -1 and pairid[1] < 0:
        continue

      if (
        wp.static(
          _cd._broadphase_filter(
            opt_broadphase_filter, ngeom_aabb, ngeom_rbound, ngeom_margin
          )
        )(geom_aabb, geom_rbound, geom_margin, geom_xpos_in, geom_xmat_in, geom1, geom2, worldid)
        or pairid[1] >= 0
      ):
        _cd._add_geom_pair(
          geom_type,
          nxn_pairid,
          naconmax_in,
          geom1,
          geom2,
          worldid,
          idx,
          ncollision_out,
          collision_pair_out,
          collision_pairid_out,
          collision_worldid_out,
        )

  return kernel


def _cumsum_i64(range_wp):
  """range_（int32, 世界×geom）→ int64 全局前缀和（warp 数组，复用缓冲）。"""
  import torch

  rng = wp.to_torch(range_wp).reshape(-1)
  key = (str(rng.device), rng.numel())
  cache = getattr(_cumsum_i64, "_cache", None)
  if cache is None:
    cache = {}
    _cumsum_i64._cache = cache
  buf = cache.get(key)
  if buf is None:
    buf = torch.empty(rng.numel(), dtype=torch.int64, device=rng.device)
    cache[key] = buf
  torch.cumsum(rng, dim=0, dtype=torch.int64, out=buf)
  return wp.from_torch(buf, dtype=wp.int64)


def sap_broadphase_i64(m, d, ctx):
  """与原生 ``sap_broadphase`` 等价的 int64 实现。"""
  nworldgeom = d.nworld * m.ngeom

  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  projection_lower = wp.empty((d.nworld, m.ngeom, 2), dtype=float)
  projection_upper = wp.empty((d.nworld, m.ngeom), dtype=float)
  sort_index = wp.empty((d.nworld, m.ngeom, 2), dtype=int)
  range_ = wp.empty((d.nworld, m.ngeom), dtype=int)
  segmented_index = wp.empty(
    d.nworld + 1 if m.opt.broadphase == BroadphaseType.SAP_SEGMENTED else 0, dtype=int
  )

  wp.launch(
    kernel=_cd._sap_project(m.opt.broadphase),
    dim=(d.nworld, m.ngeom),
    inputs=[m.ngeom, m.geom_rbound, m.geom_margin, d.geom_xpos, d.nworld, direction],
    outputs=[
      projection_lower.reshape((-1, m.ngeom)),
      projection_upper,
      sort_index.reshape((-1, m.ngeom)),
      segmented_index,
    ],
  )

  if m.opt.broadphase == BroadphaseType.SAP_TILE:
    wp.launch_tiled(
      kernel=_cd._segmented_sort(m.ngeom),
      dim=d.nworld,
      inputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
      outputs=[projection_lower.reshape((-1, m.ngeom)), sort_index.reshape((-1, m.ngeom))],
      block_dim=m.block_dim.segmented_sort,
    )
  else:
    wp.utils.segmented_sort_pairs(
      projection_lower.reshape((-1, m.ngeom)),
      sort_index.reshape((-1, m.ngeom)),
      nworldgeom,
      segmented_index,
    )

  wp.launch(
    kernel=_cd._sap_range,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.ngeom,
      projection_lower.reshape((-1, m.ngeom)),
      projection_upper,
      sort_index.reshape((-1, m.ngeom)),
    ],
    outputs=[range_],
  )

  cumulative_sum = _cumsum_i64(range_)

  nsweep = 5 * nworldgeom
  wp.launch(
    kernel=_sap_sweep_i64(
      m.opt.broadphase_filter,
      m.geom_aabb.shape[0],
      m.geom_rbound.shape[0],
      m.geom_margin.shape[0],
    ),
    dim=nsweep,
    inputs=[
      m.ngeom,
      m.geom_type,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.nxn_pairid,
      d.geom_xpos,
      d.geom_xmat,
      d.nworld,
      d.naconmax,
      sort_index.reshape((-1, m.ngeom)),
      cumulative_sum,
      nsweep,
    ],
    outputs=[d.ncollision, ctx.collision_pair, ctx.collision_pairid, ctx.collision_worldid],
  )


def _verify_source():
  import ast
  import textwrap

  try:
    src = textwrap.dedent(inspect.getsource(_cd.sap_broadphase))
  except (OSError, TypeError):
    return False
  canon = ast.dump(ast.parse(src), include_attributes=False).replace(", type_params=[]", "")
  return hashlib.sha256(canon.encode()).hexdigest() == SAP_BROADPHASE_SHA256


def install(force: bool = False, threshold: int = INT64_THRESHOLD) -> bool:
  """把 int64 版挂到 collision_driver 上；返回是否成功。"""
  state = _cd._int64_sap_state
  if state["installed"]:
    return True
  if not force and not _verify_source():
    return False
  state["original"] = _cd.sap_broadphase

  def dispatch(m, d, ctx):
    if force or d.nworld * m.ngeom >= threshold:
      return sap_broadphase_i64(m, d, ctx)
    return state["original"](m, d, ctx)

  _cd.sap_broadphase = dispatch
  state["installed"] = True
  return True


def uninstall():
  state = _cd._int64_sap_state
  if state["installed"] and state["original"] is not None:
    _cd.sap_broadphase = state["original"]
  state["installed"] = False
