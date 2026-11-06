"""Shared initial BVH staging must preserve per-world sensing and refit."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp
from mujoco_warp._src import bvh
from mujoco_warp._src.ray import rays
from mujoco_warp._src.types import vec6
from mjlab.sensor.sensor_context import SensorContext

from src._mjwarp_bvh import build_scene_bvh
from src.sim_memory import compact_model_allocations, _shared_bvh_factory, _nonvisual_recreate_factory


XML = """<mujoco>
  <asset>
    <mesh name="tetra" vertex="0 0 0  .3 0 0  0 .3 0  0 0 .3"/>
    <hfield name="hill" nrow="3" ncol="3" size="1 1 .3 .1"
      elevation="0 0 0  0 1 0  0 0 0"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 .1" group="0"/>
    <geom type="box" size=".3 .2 .1" pos="1 0 .4" euler="20 30 40" group="1"/>
    <geom type="capsule" size=".1 .2" pos="-1 0 .5" euler="30 20 10"/>
    <geom type="cylinder" size=".1 .2" pos="0 1 .5"/>
    <geom type="ellipsoid" size=".1 .2 .3" pos="0 -1 .5"/>
    <geom type="mesh" mesh="tetra" pos="2 0 .2"/>
    <geom type="hfield" hfield="hill" pos="-2 0 0"/>
    <geom type="sphere" size=".2" pos="0 0 3" group="5"/>
    <body pos="0 0 1"><freejoint/><geom type="sphere" size=".2"/></body>
    <camera name="cam" pos="0 -3 2" xyaxes="1 0 0 0 .6 .8" resolution="4 4"/>
  </worldbody>
</mujoco>"""


def context(model, worlds, cameras=False):
  return mjwarp.create_render_context(model,nworld=worlds,use_textures=False,
    enabled_geom_groups=[0,1],cam_active=[cameras])


class SharedBVHTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if _shared_bvh_factory() is None:
      raise unittest.SkipTest("shared staging requires verified mujoco-warp 3.5.0")

  def test_initial_bounds_match_for_mixed_geoms_filtered_groups_and_cameras(self):
    with wp.ScopedDevice("cpu"):
      model=mujoco.MjModel.from_xml_string(XML)
      for worlds,cameras in ((1,False),(3,False),(7,True)):
        with self.subTest(worlds=worlds,cameras=cameras):
          native=context(model,worlds,cameras)
          with patch.object(bvh,"build_scene_bvh",build_scene_bvh):
            shared=context(model,worlds,cameras)
          for field in ("lower","upper","group","enabled_geom_ids","rgb_adr","depth_adr"):
            np.testing.assert_array_equal(getattr(native,field).numpy(),getattr(shared,field).numpy(),err_msg=field)
          self.assertEqual(shared.bvh_ngeom,native.bvh_ngeom)
          self.assertEqual(shared.group_root.shape,(worlds,))

  def test_build_uses_only_one_world_of_geometry_staging(self):
    with wp.ScopedDevice("cpu"):
      model=mujoco.MjModel.from_xml_string(XML)
      data=mujoco.MjData(model);mujoco.mj_forward(model,data)
      launch=wp.launch
      calls=[]
      def record(*args,**kwargs):
        if kwargs.get("kernel") is bvh._compute_bvh_bounds:
          calls.append((kwargs["dim"],[x.shape for x in kwargs["inputs"][2:5]]))
        return launch(*args,**kwargs)
      with patch.object(wp,"launch",side_effect=record):
        rc=context(model,7)
        with patch.object(np,"tile",side_effect=AssertionError("per-world tile forbidden")):
          build_scene_bvh(model,data,rc,7)
      self.assertEqual(calls,[((7,rc.bvh_ngeom),[(7,model.ngeom)]*3),
                              ((1,rc.bvh_ngeom),[(1,model.ngeom)]*3)])

  def test_runtime_refit_and_rays_match_after_distinct_world_motion_and_sizes(self):
    with wp.ScopedDevice("cpu"):
      model=mujoco.MjModel.from_xml_string(XML)
      native=context(model,3)
      with patch.object(bvh,"build_scene_bvh",build_scene_bvh):shared=context(model,3)
      wm=mjwarp.put_model(model)
      wd=mjwarp.make_data(model,nworld=3,nconmax=35,njmax=100)
      qpos=wd.qpos.numpy();qpos[:,0]=np.array([-.4,0,.7]);wd.qpos.assign(qpos)
      sizes=np.repeat(wm.geom_size.numpy(),3,axis=0)
      sizes*=np.array([.9,1,1.1],dtype=np.float32)[:,None,None]
      wm.geom_size=wp.array(sizes,dtype=wp.vec3)
      mjwarp.forward(wm,wd)
      bvh.refit_scene_bvh(wm,wd,native);bvh.refit_scene_bvh(wm,wd,shared)
      for field in ("lower","upper"):
        np.testing.assert_array_equal(getattr(native,field).numpy(),getattr(shared,field).numpy())
      self.assertFalse(np.array_equal(shared.lower.numpy()[:shared.bvh_ngeom],
                                     shared.lower.numpy()[shared.bvh_ngeom:2*shared.bvh_ngeom]))
      pnt=wp.array(np.tile(np.array([[0,0,3],[1,0,3],[2.05,.05,3],[-2,0,3],[9,9,3]],dtype=np.float32)[None],(3,1,1)),dtype=wp.vec3)
      vec=wp.array(np.tile(np.array([[[0,0,-1]]],dtype=np.float32),(3,5,1)),dtype=wp.vec3)
      def cast(rc):
        output=(wp.empty((3,5),dtype=float),wp.empty((3,5),dtype=int),wp.empty((3,5),dtype=wp.vec3))
        rays(wm,wd,pnt,vec,vec6(1,1,0,0,0,0),True,
             wp.array(-np.ones(5,dtype=np.int32),dtype=int),*output,rc=rc)
        return output
      before=cast(native)
      after=cast(shared)
      for a,b in zip(before,after):np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-6,atol=1e-6)
      self.assertTrue((after[0].numpy()[:,:4]>0).all())

  def test_scope_restores_bvh_factory_after_failure_and_disable_skips_it(self):
    original=bvh.build_scene_bvh
    original_recreate=SensorContext.recreate
    def factory(*_args):return "built"
    with patch("src.sim_memory._shared_bvh_factory",return_value=factory), \
         patch("src.sim_memory._unused_pair_table_safe",return_value=False):
      with self.assertRaisesRegex(RuntimeError,"later failure"):
        with compact_model_allocations() as stats:
          self.assertEqual(bvh.build_scene_bvh(NS(ngeom=100),None,None,2048),"built")
          self.assertEqual(stats["bvh_staging_bytes_avoided"],2047*100*60)
          raise RuntimeError("later failure")
      self.assertIs(bvh.build_scene_bvh,original)
      self.assertIs(SensorContext.recreate,original_recreate)
    with compact_model_allocations(False):self.assertIs(bvh.build_scene_bvh,original)

  def test_unverified_source_and_version_are_rejected(self):
    with patch("src.sim_memory.version",return_value="3.6.0"):
      self.assertIsNone(_shared_bvh_factory.__wrapped__())
    with patch("src.sim_memory.inspect.getsource",return_value="def changed(): pass"):
      self.assertIsNone(_shared_bvh_factory.__wrapped__())

  def test_nonvisual_expansion_reuses_existing_context_and_visual_fields_rebuild(self):
    recreate=_nonvisual_recreate_factory()
    if recreate is None:self.skipTest("reuse requires verified mjlab 1.2.0")
    from unittest.mock import Mock
    ctx=NS(_ctx=object(),_disable_precomputed_rays=False,_create_context=Mock())
    existing=ctx._ctx
    recreate(ctx,None,{"body_mass","body_inertia","body_ipos","geom_friction",
                        "actuator_gainprm","actuator_biasprm","body_subtreemass",
                        "body_invweight0","dof_invweight0","tendon_length0","tendon_invweight0"})
    self.assertIs(ctx._ctx,existing)
    ctx._create_context.assert_not_called()
    recreate(ctx,"model",{"cam_fovy"})
    self.assertTrue(ctx._disable_precomputed_rays)
    ctx._create_context.assert_called_once_with("model")
    ctx._create_context.reset_mock()
    recreate(ctx,"geometry",{"geom_size"})
    self.assertFalse(ctx._disable_precomputed_rays)
    ctx._create_context.assert_called_once_with("geometry")
    ctx._create_context.reset_mock()
    recreate(ctx,"unknown")
    ctx._create_context.assert_called_once_with("unknown")


if __name__=="__main__":unittest.main()
