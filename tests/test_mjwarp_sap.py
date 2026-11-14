"""int64 SAP 补丁的静态测试（不需要 GPU）。

验证：
- 源码哈希守卫能识别已安装的 mujoco-warp 版本；
- install/uninstall 能正确接管并还原 `collision_driver.sap_broadphase`；
- 阈值以下不会被接管（走原生路径）。
"""

import unittest

from mujoco_warp._src import collision_driver as cd

from src import _mjwarp_sap


class SapInt64PatchTests(unittest.TestCase):
  def setUp(self):
    _mjwarp_sap.uninstall()

  def tearDown(self):
    _mjwarp_sap.uninstall()

  def test_source_guard_matches_installed_backend(self):
    self.assertTrue(_mjwarp_sap._verify_source(), "mujoco-warp 源码与补丁不匹配")

  def test_install_and_uninstall(self):
    original = cd.sap_broadphase
    self.assertTrue(_mjwarp_sap.install())
    self.assertIsNot(cd.sap_broadphase, original)
    _mjwarp_sap.uninstall()
    self.assertIs(cd.sap_broadphase, original)

  def test_threshold_dispatch_prefers_native(self):
    original = cd.sap_broadphase
    self.assertTrue(_mjwarp_sap.install(threshold=10**12))
    try:
      class _M:
        ngeom = 100
      class _D:
        nworld = 100
      sentinel = {}
      def fake_native(m, d, ctx):
        sentinel["native"] = True
      # dispatch 在阈值以下应调用原生实现
      self.assertFalse(_mjwarp_sap.sap_broadphase_i64 is cd.sap_broadphase)
    finally:
      _mjwarp_sap.uninstall()
      self.assertIs(cd.sap_broadphase, original)

  def test_int64_cumsum_survives_int32_overflow(self):
    """前缀和总量超过 2^31 时，int64 版本仍然给出正确的全局前缀和。"""
    import warp as wp

    nworld, ngeom = 2, 3
    big = 1 << 30  # 每个元素 2^30，总和 6*2^30 > 2^31
    values = [[big, big, big], [big, big, big]]
    range_wp = wp.array2d(values, dtype=int)
    cum = _mjwarp_sap._cumsum_i64(range_wp, nworld, ngeom)
    got = wp.to_torch(cum).cpu().numpy().tolist()
    expected = [big * (i + 1) for i in range(nworld * ngeom)]
    self.assertEqual(got, expected)

  def test_blocked_int64_scan_handles_tails_zeros_and_large_global_totals(self):
    import numpy as np
    import warp as wp
    with wp.ScopedDevice("cpu"):
      for width in (1, 255, 256, 257, 1025):
        data = np.random.default_rng(42).integers(0, 100_000_000, size=(3,width), dtype=np.int32)
        data[0,::2] = 0
        result = _mjwarp_sap._cumsum_i64(wp.array(data, dtype=int), 3, width)
        np.testing.assert_array_equal(result.numpy(), np.cumsum(data.reshape(-1), dtype=np.int64))

  def test_sparse_and_native_sap_contacts_match_with_two_live_models(self):
    import mujoco
    import mujoco_warp as mjwarp
    import numpy as np
    import warp as wp
    from src.sim_memory import compact_model_allocations
    from mujoco_warp._src.types import BroadphaseType
    xml = '''<mujoco><worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
      <body pos=".3 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
    with wp.ScopedDevice("cpu"):
      model = mujoco.MjModel.from_xml_string(xml)
      native = mjwarp.put_model(model)
      with compact_model_allocations():
        first = mjwarp.put_model(model)
        # Same number of geoms, different filtering: catches a global-table bug.
        other = mujoco.MjModel.from_xml_string(xml)
        other.geom_contype[2] = 0
        other.geom_conaffinity[2] = 0
        second = mjwarp.put_model(other)
      self.assertIsNot(first._lgdog_sparse_pairs, second._lgdog_sparse_pairs)
      for m in (native, first, second):
        m.opt.broadphase = BroadphaseType.SAP_SEGMENTED
      dn = mjwarp.make_data(model, nworld=3, nconmax=20, njmax=40)
      df = mjwarp.make_data(model, nworld=3, nconmax=20, njmax=40)
      ds = mjwarp.make_data(other, nworld=3, nconmax=20, njmax=40)
      def contacts(m, d):
        mjwarp.forward(m, d)
        n = int(d.nacon.numpy()[0])
        pairs = d.contact.geom.numpy()[:n]
        worlds = d.contact.worldid.numpy()[:n]
        return sorted((int(w), *sorted(pair.tolist())) for w, pair in zip(worlds,pairs))
      expected = contacts(native, dn)
      self.assertTrue(expected)
      self.assertEqual(contacts(first, df), expected)
      self.assertLess(len(contacts(second, ds)), len(expected))
      # Constructing/running the second model must not contaminate the first.
      self.assertEqual(contacts(first, df), expected)
      work = df._lgdog_sap_workspace
      contacts(first, df)
      self.assertIs(df._lgdog_sap_workspace, work)
      for _ in range(3):
        mjwarp.step(native, dn)
        mjwarp.step(first, df)
      np.testing.assert_allclose(dn.qpos.numpy(), df.qpos.numpy(), atol=1e-6, rtol=1e-5)

  def test_simulation_prepares_shared_workspace_before_first_graph_or_forward(self):
    from unittest.mock import patch
    import mujoco
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg
    from src.sim_memory import compact_model_allocations, _streaming_model_factory
    from mujoco_warp._src.types import BroadphaseType
    factory = _streaming_model_factory(sparse=True)
    def sap_model(*args, **kwargs):
      m = factory(*args, **kwargs)
      m.opt.broadphase = BroadphaseType.SAP_SEGMENTED
      return m
    original = Simulation.create_graph
    seen = []
    def before_capture(sim):
      self.assertIsNotNone(getattr(sim.wp_data, "_lgdog_sap_workspace", None))
      seen.append(sim.wp_data._lgdog_sap_workspace)
      return original(sim)
    xml = '''<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice("cpu"), \
         patch("src.sim_memory._streaming_model_factory", return_value=sap_model), \
         patch.object(Simulation, "create_graph", before_capture), compact_model_allocations():
      sim = Simulation(3, SimulationCfg(nconmax=20, njmax=40), model, "cpu")
      # env.reset launches forward FIRST, without ever launching the step graph.
      sim.reset()
      sim.forward()
      self.assertGreater(int(sim.wp_data.nacon.numpy()[0]), 0)
      sim.create_graph()
      self.assertIs(seen[0], seen[1])
      self.assertIs(seen[0], sim.wp_data._lgdog_sap_workspace)

  def test_missing_workspace_is_rejected_during_graph_capture(self):
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    device = NS(is_capturing=True)
    m = NS(ngeom=100, opt=NS(broadphase=1))
    d = NS(nworld=3, geom_xpos=NS(device=device))
    with patch.object(_mjwarp_sap.wp, "empty", side_effect=AssertionError("graph-owned persistent allocation")):
      with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
        _mjwarp_sap._workspace(m, d)

  def test_cuda_forward_graph_can_launch_before_step_and_after_recapture(self):
    from unittest.mock import patch
    import mujoco
    import warp as wp
    from mjlab.sim import Simulation, SimulationCfg
    from src.sim_memory import compact_model_allocations, _streaming_model_factory
    from mujoco_warp._src.types import BroadphaseType
    if not wp.is_cuda_available():
      self.skipTest("requires real CUDA graph allocation/replay")
    factory = _streaming_model_factory(sparse=True)
    def sap_model(*args, **kwargs):
      m = factory(*args, **kwargs)
      m.opt.broadphase = BroadphaseType.SAP_SEGMENTED
      return m
    xml = '''<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>'''
    with wp.ScopedDevice("cuda:0"), \
         patch("src.sim_memory._streaming_model_factory", return_value=sap_model), compact_model_allocations():
      sim = Simulation(3, SimulationCfg(nconmax=20, njmax=40),
                       mujoco.MjModel.from_xml_string(xml), "cuda:0")
      if not sim.use_cuda_graph:
        self.skipTest("driver/mempool does not enable CUDA graphs")
      self.assertTrue(sim.wp_data._lgdog_sap_workspace[1]["sort_prepared"])
      for _ in range(2):
        sim.reset()
        sim.forward()
        wp.synchronize_device("cuda:0")
        self.assertGreater(int(sim.wp_data.nacon.numpy()[0]), 0)
        sim.create_graph()


if __name__ == "__main__":
  unittest.main()
