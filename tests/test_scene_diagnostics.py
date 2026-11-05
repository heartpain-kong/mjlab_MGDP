"""Scene budget reporting must preserve hooks and serialized configuration."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from src.scene_diagnostics import report_scene_initialization, scene_memory_estimates


class SceneDiagnosticsTests(unittest.TestCase):
  def test_pair_estimate_is_quadratic_and_world_pose_estimate_is_linear(self):
    sizes=scene_memory_estimates(41736,2048)
    self.assertGreater(sizes["pair_index_bytes"],25*1024**3)
    reduced=scene_memory_estimates(6956,256)
    self.assertLess(reduced["pair_index_bytes"],1024**3)
    twice=scene_memory_estimates(6956,512)
    self.assertEqual(reduced["pair_index_bytes"],twice["pair_index_bytes"])
    self.assertEqual(2*reduced["geom_pose_bytes"],twice["geom_pose_bytes"])

  def test_hook_composition_and_restoration_on_native_failure(self):
    def original(spec):spec.geoms.append("added")
    cfg=NS(scene=NS(spec_fn=original,num_envs=64))
    spec=NS(geoms=["floor"])
    output=StringIO()
    with self.assertRaisesRegex(RuntimeError,"native initialization"),redirect_stdout(output):
      with report_scene_initialization(cfg):
        cfg.scene.spec_fn(spec)
        raise RuntimeError("native initialization")
    self.assertIs(cfg.scene.spec_fn,original)
    self.assertEqual(len(spec.geoms),2)
    self.assertIn("geoms=2, envs=64",output.getvalue())

  def test_train_applies_compact_grid_before_physics_and_rejects_missing_types(self):
    from scripts import train
    class StopBeforePhysics(Exception):pass
    cfg=replace(train.TrainConfig.from_task("LgDog1-Race-MGDP"),terrain_num_rows=4,terrain_num_cols=10)
    with patch.dict("os.environ",{"CUDA_VISIBLE_DEVICES":""}), \
         patch.object(train,"configure_torch_backends"), \
         patch.object(train,"ManagerBasedRlEnv",side_effect=StopBeforePhysics):
      with self.assertRaises(StopBeforePhysics):
        train.run_train("LgDog1-Race-MGDP",cfg,Path("unused"))
      grid=cfg.env.scene.terrain.terrain_generator
      self.assertEqual((grid.num_rows,grid.num_cols),(4,10))
      self.assertEqual(len(grid.sub_terrains),10)
      self.assertIsNone(cfg.env.scene.spec_fn)
      with self.assertRaisesRegex(ValueError,"保留所有障碍类型"):
        train.run_train("LgDog1-Race-MGDP",replace(cfg,terrain_num_cols=9),Path("unused"))


if __name__=="__main__":unittest.main()
