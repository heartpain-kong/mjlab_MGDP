"""Compare bounded construction against native contact/sensor filtering."""

import dataclasses
import inspect
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import io, types

from src._mjwarp_pairs import build_collision_pairs
from src.sim_memory import _replace_pair_block, _streaming_model_factory, _unused_pair_table_safe


XML = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <geom name="static" type="box" size=".1 .1 .1" pos="1 0 .1"/>
    <body name="base" pos="0 0 1"><freejoint/>
      <geom name="a" type="sphere" size=".1" contype="1" conaffinity="2"/>
      <geom name="b" type="capsule" size=".05 .1" pos=".2 0 0"/>
      <body name="child" pos="0 0 .3"><joint type="hinge"/>
        <geom name="c" type="box" size=".1 .1 .1" contype="2" conaffinity="1"/>
        <body name="welded" pos="0 0 .2">
          <geom name="d" type="sphere" size=".05" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="other" pos="0 1 1"><freejoint/>
      <geom name="e" type="sphere" size=".15"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="base" body2="other"/>
    <pair geom1="a" geom2="d"/>
    <pair geom1="floor" geom2="a"/>
  </contact>
  <sensor>
    <distance name="ab" geom1="a" geom2="b" cutoff="2"/>
    <normal name="ba" geom1="b" geom2="a" cutoff="2"/>
    <fromto name="body" body1="base" body2="child" cutoff="2"/>
    <distance name="ignored" geom1="d" geom2="static" cutoff="2"/>
  </sensor>
</mujoco>"""


class StreamingPairTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not _unused_pair_table_safe():
      raise unittest.SkipTest("streaming implementation requires verified mujoco-warp 3.5.0")

  def assert_pair_fields_equal(self, actual, expected):
    for field in ("nxn_geom_pair_filtered", "nxn_pairid", "nxn_pairid_filtered",
                  "sensor_collision_start_adr"):
      np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field).numpy(),err_msg=field)
    self.assertEqual(actual.nsensorcollision,expected.nsensorcollision)
    self.assertEqual(actual.geom_pair_type_count,expected.geom_pair_type_count)
    self.assertEqual(actual.nxn_geom_pair.shape,(0,2))

  def test_chunk_boundaries_contacts_exclusions_parent_filter_and_sensor_order(self):
    with wp.ScopedDevice("cpu"):
      for disable_parent in (False, True):
        model=mujoco.MjModel.from_xml_string(XML)
        if disable_parent:model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        native=io.put_model(model)
        for chunk_size in (1,2,3,7,262144):
          with self.subTest(disable_parent=disable_parent,chunk_size=chunk_size):
            fields,opt=NS(),NS()
            build_collision_pairs(model,fields,opt,types,chunk_size=chunk_size)
            self.assert_pair_fields_equal(fields,native)
            self.assertEqual(opt.broadphase,native.opt.broadphase)

  def test_single_geom_empty_pairs_and_no_global_triangular_index_allocation(self):
    with wp.ScopedDevice("cpu"):
      for xml in ("<mujoco/>","<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>",XML):
        model=mujoco.MjModel.from_xml_string(xml)
        native=io.put_model(model)
        fields,opt=NS(),NS()
        with patch.object(np,"triu_indices",side_effect=AssertionError("dense indices forbidden")):
          build_collision_pairs(model,fields,opt,types,chunk_size=2)
        self.assert_pair_fields_equal(fields,native)

  def test_compiled_factory_preserves_all_other_model_fields(self):
    def compare(a,b,path):
      if isinstance(a,wp.array):np.testing.assert_array_equal(a.numpy(),b.numpy(),err_msg=path)
      elif dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
          if field.name!="nxn_geom_pair":compare(getattr(a,field.name),getattr(b,field.name),path+"."+field.name)
      elif isinstance(a,(tuple,list)):
        self.assertEqual(len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+str(i))
      elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
      else:self.assertEqual(a,b,path)
    with wp.ScopedDevice("cpu"):
      model=mujoco.MjModel.from_xml_string(XML)
      native=io.put_model(model)
      with patch.object(np,"triu_indices",side_effect=AssertionError("dense indices forbidden")):
        streamed=_streaming_model_factory()(model)
      self.assertEqual(streamed.nxn_geom_pair.size,0)
      compare(native,streamed,"model")

  def test_modified_backend_pair_block_is_rejected(self):
    source=inspect.getsource(io.put_model)
    _replace_pair_block(source)
    changed=source.replace("np.triu_indices(mjm.ngeom, k=1)","np.triu_indices(mjm.ngeom, k=0)")
    with self.assertRaisesRegex(ValueError,"source changed"):_replace_pair_block(changed)

  def test_sparse_filtering_never_allocates_dense_lookup_and_preserves_sensor_order(self):
    with wp.ScopedDevice("cpu"):
      model = mujoco.MjModel.from_xml_string(XML)
      native = io.put_model(model)
      count = model.ngeom * (model.ngeom-1)//2
      allocate = np.empty
      def bounded(shape, *args, **kwargs):
        if shape == (count, 2):
          raise AssertionError("dense pair lookup allocated")
        return allocate(shape, *args, **kwargs)
      fields, opt = NS(), NS()
      with patch.object(np, "empty", side_effect=bounded):
        build_collision_pairs(model, fields, opt, types, chunk_size=2, sparse=True)
      for name in ("nxn_geom_pair_filtered", "nxn_pairid_filtered", "sensor_collision_start_adr"):
        np.testing.assert_array_equal(getattr(fields, name), getattr(native, name).numpy())
      self.assertEqual(fields.nxn_pairid.shape, (1,2))
      self.assertEqual(fields.nsensorcollision, native.nsensorcollision)
      state = fields._lgdog_sparse_pairs
      for key, value in zip(state["np_keys"], state["np_vals"]):
        i, j = divmod(int(key), model.ngeom)
        index = i * (2*model.ngeom-i-3)//2 + j-1
        np.testing.assert_array_equal(value, native.nxn_pairid.numpy()[index])

  def test_static_pair_pruning_retains_explicit_and_sensor_only_static_pairs(self):
    xml = '''<mujoco><worldbody>''' + ''.join(
      f'<geom name="s{i}" type="sphere" size=".1" pos="{i*.3} 0 0"/>'
      for i in range(40)) + '''
      <body pos="0 0 .1"><freejoint/><geom name="robot" size=".1"/></body>
      </worldbody><contact><pair geom1="s0" geom2="s1"/></contact>
      <sensor><distance geom1="s2" geom2="s3" cutoff="2"/>
      <normal geom1="s3" geom2="s2" cutoff="2"/></sensor></mujoco>'''
    with wp.ScopedDevice("cpu"):
      model = mujoco.MjModel.from_xml_string(xml)
      native = io.put_model(model)
      for chunk in (1, 7, 262144):
        fields, opt = NS(), NS()
        build_collision_pairs(model, fields, opt, types, chunk_size=chunk, sparse=True)
        for name in ("nxn_geom_pair_filtered", "nxn_pairid_filtered", "sensor_collision_start_adr"):
          np.testing.assert_array_equal(getattr(fields, name), getattr(native, name).numpy())
        self.assertEqual(fields.nsensorcollision, native.nsensorcollision)

  def test_all_broadphase_choices_match_native_pair_construction(self):
    import ast
    function=ast.parse(inspect.getsource(io.put_model)).body[0]
    start=next(i for i,x in enumerate(function.body) if isinstance(x,ast.Assign)
      and any(isinstance(t,ast.Name) and t.id=="filterparent" for t in x.targets))
    end=next(i+1 for i,x in enumerate(function.body) if isinstance(x,ast.Assign)
      and any(isinstance(t,ast.Attribute) and t.attr=="geom_pair_type_count" for t in x.targets))
    reference=compile(ast.Module(body=function.body[start:end],type_ignores=[]),"<native-pair-block>","exec")
    for n,expected_phase in ((20,types.BroadphaseType.NXN),(800,types.BroadphaseType.SAP_TILE),
                             (1100,types.BroadphaseType.SAP_SEGMENTED)):
      with self.subTest(n=n):
        model=NS(ngeom=n,npair=0,nsensor=0,geom_bodyid=np.arange(1,n+1,dtype=np.int32),
          body_weldid=np.arange(n+1,dtype=np.int32),body_parentid=np.zeros(n+1,dtype=np.int32),
          geom_contype=np.ones(n,dtype=np.int32),geom_conaffinity=np.ones(n,dtype=np.int32),
          geom_type=np.arange(n,dtype=np.int32)%len(types.GeomType),
          exclude_signature=np.array([],dtype=np.int32),sensor_type=np.array([],dtype=np.int32),
          opt=NS(disableflags=0))
        native,native_opt=NS(),NS()
        exec(reference,dict(np=np,types=types,mjm=model,m=native,opt=native_opt,
                           is_collision_sensor=np.array([],dtype=bool)))
        fields,opt=NS(),NS()
        build_collision_pairs(model,fields,opt,types,chunk_size=4096)
        self.assertEqual(opt.broadphase,expected_phase)
        self.assertEqual(opt.broadphase,native_opt.broadphase)
        for name in ("nxn_geom_pair_filtered","nxn_pairid","nxn_pairid_filtered","sensor_collision_start_adr"):
          np.testing.assert_array_equal(getattr(fields,name),getattr(native,name),err_msg=name)
        self.assertEqual(fields.geom_pair_type_count,native.geom_pair_type_count)


if __name__=="__main__":unittest.main()
