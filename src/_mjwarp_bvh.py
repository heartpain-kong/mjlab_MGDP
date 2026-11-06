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

"""Build MJWarp 3.5.0 initial BVH bounds without tiled geometry staging.

Upstream starts every render world from the same single MuJoCo MjData. Compute
those bounds once and broadcast them, retaining independent per-world BVH leaves,
groups and roots. Runtime refit and rays continue to use the native implementation.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _broadcast_bounds(
  lower_in: wp.array(dtype=wp.vec3),
  upper_in: wp.array(dtype=wp.vec3),
  ngeom: int,
  lower_out: wp.array(dtype=wp.vec3),
  upper_out: wp.array(dtype=wp.vec3),
  group_out: wp.array(dtype=int),
):
  world, geom = wp.tid()
  index = world * ngeom + geom
  lower_out[index] = lower_in[geom]
  upper_out[index] = upper_in[geom]
  group_out[index] = world


def build_scene_bvh(mjm, mjd, rc, nworld):
  from mujoco_warp._src import bvh
  avoided = max(nworld-1, 0) * mjm.ngeom * 60
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
    inputs=[lower, upper, rc.bvh_ngeom],
    outputs=[rc.lower, rc.upper, rc.group],
  )
  rc.bvh = wp.Bvh(rc.lower, rc.upper, groups=rc.group, constructor="sah")
  rc.bvh_id = rc.bvh.id
  wp.launch(kernel=bvh.compute_bvh_group_roots, dim=nworld,
            inputs=[rc.bvh_id], outputs=[rc.group_root])
