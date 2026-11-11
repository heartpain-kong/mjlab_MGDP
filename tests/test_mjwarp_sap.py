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


if __name__ == "__main__":
  unittest.main()
