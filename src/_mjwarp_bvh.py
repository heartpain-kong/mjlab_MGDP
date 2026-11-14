"""共享地形 BVH：静态地形只建一份，所有世界共用。

背景：`mujoco_warp._src.bvh.build_scene_bvh` 为每个世界复制一套叶子
（`lower/upper` 形状 `nworld × bvh_ngeom`），4096 世界 × 27842 geom ≈ 6.7 GB。
但赛道场景里地形在所有世界中完全相同且静止，射线传感器
（`include_geom_groups=(0,)`）只查地形组，因此一份 BVH 就能服务所有世界。

判定条件（不满足就退回原生逐世界 BVH）：
- 所有启用的 geom 都焊接到 worldbody，且不属于 mocap/flex；
- 所有这些 geom 的 group 都是 0（地形组）。

仅在经源码校验的 ray-only context 中共享；相机仍保留逐世界 BVH。
地形静止 ⇒ refit 不再需要，运行时按 context 标记分流。
"""

from __future__ import annotations

from contextvars import ContextVar

import numpy as np
import warp as wp

# Visual field expansion cannot reuse a tree built from the host base model.
SHARING_ALLOWED = ContextVar("lgdog_static_bvh_allowed", default=True)


@wp.kernel(module="unique", enable_backward=False)
def _broadcast_bounds(
  lower_in: wp.array(dtype=wp.vec3),
  upper_in: wp.array(dtype=wp.vec3),
  group_in: wp.array(dtype=int),
  bvh_ngeom: int,
  lower_out: wp.array(dtype=wp.vec3),
  upper_out: wp.array(dtype=wp.vec3),
  group_out: wp.array(dtype=int),
):
  world, geom = wp.tid()
  index = world * bvh_ngeom + geom
  lower_out[index] = lower_in[geom]
  upper_out[index] = upper_in[geom]
  group_out[index] = world


def static_terrain_only(mjm, ids) -> bool:
  ids = np.asarray(ids, dtype=np.int64)
  if ids.size == 0:
    return False
  body = np.asarray(mjm.geom_bodyid)[ids]
  group = np.asarray(mjm.geom_group)[ids]
  weld = np.asarray(mjm.body_weldid)[body]
  static = (body == 0) | (weld == 0)
  # Mocap bodies can be welded to world but still move independently per world.
  return bool(not mjm.nflex and np.all(static) and np.all(group == 0)
              and np.all(np.asarray(mjm.body_mocapid)[body] < 0))


def build_scene_bvh(mjm, mjd, rc, nworld):
  """静态地形场景只建一份 BVH；否则退回原生逐世界实现。"""
  from mujoco_warp._src import bvh

  # Only the verified context factory may opt in: it avoids the large leaf
  # allocation BEFORE build_scene_bvh and installs matching runtime ray dispatch.
  if getattr(rc, "_lgdog_shared_terrain", False):
    return _build_shared(mjm, mjd, rc, nworld)

  avoided = max(nworld - 1, 0) * mjm.ngeom * 60
  print(f"[memory] shared BVH initialization: geoms={mjm.ngeom}, worlds={nworld}; "
        f"avoids={avoided/1024**3:.2f} GiB repeated pose/size staging; "
        "per-world BVH and runtime refit unchanged", flush=True)
  geom_type = wp.array(mjm.geom_type, dtype=int)
  geom_dataid = wp.array(mjm.geom_dataid, dtype=int)
  geom_size = wp.array(mjm.geom_size[None, :, :], dtype=wp.vec3)
  geom_xpos = wp.array(mjd.geom_xpos[None, :, :], dtype=wp.vec3)
  geom_xmat = wp.array(mjd.geom_xmat.reshape(1, mjm.ngeom, 3, 3), dtype=wp.mat33)
  lower = wp.empty(rc.bvh_ngeom, dtype=wp.vec3)
  upper = wp.empty(rc.bvh_ngeom, dtype=wp.vec3)
  group = wp.empty(rc.bvh_ngeom, dtype=int)
  wp.launch(
    kernel=bvh._compute_bvh_bounds,
    dim=(1, rc.bvh_ngeom),
    inputs=[geom_type, geom_dataid, geom_size, geom_xpos, geom_xmat,
            rc.bvh_ngeom, rc.enabled_geom_ids, rc.mesh_bounds_size,
            rc.hfield_bounds_size, lower, upper, group],
  )
  wp.launch(
    kernel=_broadcast_bounds,
    dim=(nworld, rc.bvh_ngeom),
    inputs=[lower, upper, group, rc.bvh_ngeom],
    outputs=[rc.lower, rc.upper, rc.group],
  )
  rc.bvh = wp.Bvh(rc.lower, rc.upper, groups=rc.group, constructor="sah")
  rc.bvh_id = rc.bvh.id
  wp.launch(kernel=bvh.compute_bvh_group_roots, dim=nworld,
            inputs=[rc.bvh_id], outputs=[rc.group_root])


def _build_shared(mjm, mjd, rc, nworld):
  from mujoco_warp._src import bvh

  geom_type = wp.array(mjm.geom_type, dtype=int)
  geom_dataid = wp.array(mjm.geom_dataid, dtype=int)
  geom_size = wp.array(mjm.geom_size[None, :, :], dtype=wp.vec3)
  geom_xpos = wp.array(mjd.geom_xpos[None, :, :], dtype=wp.vec3)
  geom_xmat = wp.array(mjd.geom_xmat.reshape(1, mjm.ngeom, 3, 3), dtype=wp.mat33)
  lower, upper, group = rc.lower, rc.upper, rc.group
  wp.launch(
    kernel=bvh._compute_bvh_bounds,
    dim=(1, rc.bvh_ngeom),
    inputs=[geom_type, geom_dataid, geom_size, geom_xpos, geom_xmat,
            rc.bvh_ngeom, rc.enabled_geom_ids, rc.mesh_bounds_size,
            rc.hfield_bounds_size, lower, upper, group],
  )
  rc.lower, rc.upper, rc.group = lower, upper, group
  rc.bvh = wp.Bvh(rc.lower, rc.upper, groups=rc.group, constructor="sah")
  rc.bvh_id = rc.bvh.id

  single_root = wp.empty(1, dtype=int)
  wp.launch(kernel=bvh.compute_bvh_group_roots, dim=1,
            inputs=[rc.bvh_id], outputs=[single_root])
  root_value = int(single_root.numpy()[0])
  rc.group_root = wp.full(nworld, root_value, dtype=int)

  saved = max(nworld - 1, 0) * rc.bvh_ngeom * 28
  print(f"[memory] terrain BVH shared across {nworld} worlds: leaves={rc.bvh_ngeom} "
        f"(avoids {saved/1024**3:.2f} GiB leaf arrays, excluding tree/scratch), "
        "static refit disabled", flush=True)
  return None
