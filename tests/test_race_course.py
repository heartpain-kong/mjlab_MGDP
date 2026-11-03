"""2027 比赛场地接入的回归测试：地形定义、任务注册、转换后的场景资产。

不需要 GPU / MuJoCo 仿真；只检查配置和文件，真场地跑通靠
`python scripts/sim2sim.py ... --map race`（CPU 也能跑，秒级）。
"""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from src.dashboard.catalog import SIM2SIM_MAPS, TASKS, resolve_task
from src.tasks.velocity.config.lgdog1.race_depth_terrain_cfg import (
  RACE_DEPTH_PROPORTIONS,
  BoxClearanceFrameTerrainCfg,
  TireRingTerrainCfg,
  lgdog1_race_depth_terrain_cfg,
)
from src.tasks.velocity.config.lgdog1.race_terrain_cfg import (
  RACE_PROPORTIONS,
  lgdog1_race_terrain_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
TERRAINS = ROOT / "src" / "assets" / "terrains"


class RaceTerrainTests(unittest.TestCase):
  def test_proportions_sum_to_one(self):
    self.assertAlmostEqual(sum(RACE_PROPORTIONS.values()), 1.0, places=6)

  def test_terrain_uses_race_blocks(self):
    cfg = lgdog1_race_terrain_cfg()
    self.assertEqual(set(cfg.sub_terrains), set(RACE_PROPORTIONS))
    for name, proportion in RACE_PROPORTIONS.items():
      self.assertAlmostEqual(cfg.sub_terrains[name].proportion, proportion)
    # 尺寸照规则：梅花桩 0.2×0.2×0.1，单木桥宽 0.4
    stones = cfg.sub_terrains["stepping_stones"]
    self.assertEqual(stones.stone_size_range, (0.2, 0.2))
    self.assertEqual(stones.stone_height, 0.1)
    beams = cfg.sub_terrains["balance_beam"]
    self.assertEqual(beams.beam_width_range, (0.4, 0.4))
    self.assertEqual(beams.beam_height, 0.1)
    self.assertTrue(cfg.curriculum)


class RaceTaskTests(unittest.TestCase):
  def test_task_is_registered_with_race_terrain(self):
    import src.tasks  # noqa: F401  触发注册
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    self.assertIn("LgDog1-Race-Blind-HIM", list_tasks())
    cfg = load_env_cfg("LgDog1-Race-Blind-HIM")
    self.assertIsNotNone(cfg.scene.terrain.terrain_generator)
    self.assertEqual(set(cfg.scene.terrain.terrain_generator.sub_terrains), set(RACE_PROPORTIONS))
    # 观测/奖励契约与标准 HIM 一致：同一份 checkpoint 可以在两边互相 play
    self.assertEqual(cfg.observations["actor"].history_length, 3)
    self.assertIn("feet_air_time", cfg.rewards)

  def test_catalog_entry_matches_registry(self):
    alias, task = resolve_task("race")
    self.assertEqual(alias, "race")
    self.assertEqual(task["id"], "LgDog1-Race-Blind-HIM")
    self.assertEqual(task["scene"], "race")
    self.assertEqual(task["profile"], "him_standard")
    self.assertEqual(TASKS["race"]["experiment"], "lgdog1_velocity_race_him_v1")


class RaceAssetTests(unittest.TestCase):
  def test_converted_courses_exist_and_are_radian_quat(self):
    for name, geoms in (("race_obstacle_2027.xml", 984), ("race_task_2027.xml", 105)):
      path = TERRAINS / name
      self.assertTrue(path.is_file(), path)
      tree = ET.parse(path)
      root = tree.getroot()
      compiler = root.find("compiler")
      self.assertIsNotNone(compiler)
      self.assertEqual(compiler.get("angle"), "radian")
      # 关键：不能留下 option/default（会污染机器人场景的默认值和积分器）
      for tag in ("option", "default", "visual", "statistic"):
        self.assertIsNone(root.find(tag), f"{name} 不该保留 <{tag}>")
      self.assertEqual(sum(1 for _ in root.iter("geom")), geoms)
      # 姿态必须全部烘成 quat，不能留 euler（角度单位会错位）
      self.assertEqual(sum(1 for node in root.iter("geom") if node.get("euler")), 0)

  def test_task_course_textures_are_local(self):
    labels = TERRAINS / "race_task_labels"
    for name in ("food.png", "tool.png", "instrument.png", "medicine.png"):
      self.assertTrue((labels / name).is_file(), name)
    text = (TERRAINS / "race_task_2027.xml").read_text(encoding="utf-8")
    self.assertNotIn("../assets/task_labels", text)
    self.assertIn("race_task_labels/food.png", text)

  def test_sim2sim_catalog_lists_courses(self):
    self.assertIn("race", SIM2SIM_MAPS)
    self.assertIn("race_task", SIM2SIM_MAPS)
    self.assertEqual(SIM2SIM_MAPS["race"]["geoms"], 984)

  def test_sim2sim_map_names_resolve_to_files(self):
    """sim2sim 的 --map race 必须真的有文件可挂。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
      "lgdog_sim2sim", ROOT / "scripts" / "sim2sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    self.assertEqual(set(module.COURSE_MAPS), {"race", "race_task"})
    for kind, filename in module.COURSE_MAPS.items():
      self.assertTrue((module.TERRAIN_DIR / filename).is_file(), kind)


class RaceDepthTerrainTests(unittest.TestCase):
  """深度感知用的比赛地形：多了限高横梁和轮胎环。"""

  def test_proportions_sum_to_one(self):
    self.assertAlmostEqual(sum(RACE_DEPTH_PROPORTIONS.values()), 1.0, places=6)

  def test_depth_terrain_contains_overhang_and_tires(self):
    cfg = lgdog1_race_depth_terrain_cfg()
    self.assertEqual(set(cfg.sub_terrains), set(RACE_DEPTH_PROPORTIONS))
    self.assertIsInstance(
      cfg.sub_terrains["clearance_frame"], BoxClearanceFrameTerrainCfg)
    self.assertIsInstance(cfg.sub_terrains["tire_ring"], TireRingTerrainCfg)

  def test_overhang_geometry_is_real_3d(self):
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = BoxClearanceFrameTerrainCfg(proportion=1.0)
    cfg.size = (8.0, 8.0)
    output = cfg.function(0.5, spec, np.random.default_rng(0))
    # plane + 2 posts + beam
    self.assertEqual(len(output.geometries), 4)
    beam = output.geometries[-1].geom
    self.assertGreater(float(beam.pos[2]), 0.2)
    self.assertLess(float(beam.pos[2]), 0.4)

  def test_tire_ring_has_segments(self):
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = TireRingTerrainCfg(proportion=1.0)
    cfg.size = (8.0, 8.0)
    output = cfg.function(0.5, spec, np.random.default_rng(0))
    self.assertEqual(len(output.geometries), 1 + cfg.segments)


if __name__ == "__main__":
  unittest.main()
