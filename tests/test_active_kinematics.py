from unittest.mock import patch

import mujoco
import mujoco_warp as mw
import numpy as np
import warp as wp

from src import _mjwarp_active_kinematics as active
from src.sim_memory import compact_model_allocations


def test_active_grid_matches_native_including_mocap_visual_geoms_and_expanded_geometry():
  boxes=''.join(f'<geom type="box" size=".1 .1 .1" pos="{10+i} 0 0"/>' for i in range(260))
  xml=f'''<mujoco><worldbody>{boxes}<geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/>
      <geom type="sphere" size=".02" contype="0" conaffinity="0" pos=".2 0 0"/>
      <site pos=".1 0 0"/></body>
    <body mocap="true" pos="4 0 0"><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
  host=mujoco.MjModel.from_xml_string(xml)
  with wp.ScopedDevice("cpu"):
    with compact_model_allocations(collision_mode="sap",active_kinematics=False):
      native=mw.put_model(host); reference=mw.make_data(host,nworld=3,nconmax=20,njmax=40)
    with compact_model_allocations(collision_mode="sap"):
      fast=mw.put_model(host); data=mw.make_data(host,nworld=3,nconmax=20,njmax=40)
    assert active.available(fast) and not active.available(native)
    assert fast._lgdog_active_geoms.size == 3
    for expand in (False,True):
      if expand:
        for model in (native,fast):
          values=np.repeat(model.geom_pos.numpy(),3,axis=0)
          values[:,261,1]=[0.,.1,.2]
          model.geom_pos=wp.array(values,dtype=wp.vec3)
      for d in (reference,data):
        q=d.qpos.numpy();q[:,0]=[-.2,.1,.3];d.qpos.assign(q)
        mocap=d.mocap_pos.numpy();mocap[:,0,1]=[0.,.2,.4];d.mocap_pos.assign(mocap)
      mw.forward(native,reference);mw.forward(fast,data)
      for name in ("geom_xpos","geom_xmat","site_xpos","qacc"):
        np.testing.assert_allclose(getattr(data,name).numpy(),getattr(reference,name).numpy(),atol=1e-6,rtol=1e-5)
    # Replaced topology goes through the native path.
    fast.geom_bodyid=wp.array(fast.geom_bodyid.numpy(),dtype=int)
    assert not active.available(fast)


def test_unknown_sources_and_small_models_keep_native_grid():
  with patch.object(active,"version",return_value="3.6.0"):
    assert active.factory.__wrapped__() is None
  with wp.ScopedDevice("cpu"):
    host=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
    with compact_model_allocations():
      model=mw.put_model(host)
    assert not active.available(model)
