"""Depth-perception variant of the 2027 race terrain.

`race_terrain_cfg.py` is built for the HIM height-scan contract and therefore
only contains obstacles that a heightfield can represent.  A depth camera can
also see **overhangs and head-level geometry**, so this variant adds the two
obstacles that were missing, using the *competition* dimensions taken from
`src/assets/terrains/race_obstacle_2027.xml`:

* 限高框架 ``clearance_frame``: 0.8 x 0.8 m cage, 6 legs + 3 crossbars +
  2 longbars, beam underside 0.34 -> 0.28 m depending on difficulty.
  Two cages per cell so the policy sees the obstacle repeatedly.
* 轮胎阵 ``tire_array``: 2 x 4 tyres, centre spacing 0.55 m, outer radius
  0.2744 m, inner radius 0.1646 m, height 0.165 m, lying flat - exactly the
  competition layout.

Only the MGDP (depth) task uses this terrain.  The HIM task keeps the
heightfield-only variant unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxNarrowBeamsTerrainCfg,
  BoxOpenStairsTerrainCfg,
  BoxRandomSpreadTerrainCfg,
  BoxSteppingStonesTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane

RACE_DEPTH_PROPORTIONS = {
  "start_flat": 0.06,
  "stepping_stones": 0.12,
  "balance_beam": 0.10,
  "gap_steps": 0.10,
  "brick_pit": 0.10,
  "slalom_posts": 0.10,
  "high_wall": 0.08,
  "clearance_frame": 0.18,
  "tire_array": 0.12,
  "rough": 0.04,
}


@dataclass(kw_only=True)
class BoxClearanceFrameTerrainCfg(SubTerrainCfg):
  """0.8 x 0.8 m cage (6 legs + 3 crossbars + 2 longbars), two per cell."""

  frame_size: float = 0.8
  frame_count: int = 2
  frame_spacing: float = 2.2
  underside_range: tuple[float, float] = (0.34, 0.28)
  bar_half: float = 0.01
  leg_half: float = 0.01

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    base = make_plane(body, self.size, 0.0, center_zero=False)
    geoms = [TerrainGeometry(geom=base[0], color=(0.55, 0.55, 0.55, 1.0))]

    underside = self.underside_range[0] + difficulty * (
      self.underside_range[1] - self.underside_range[0]
    )
    half = self.frame_size / 2.0
    side_half = half + self.leg_half  # 0.8m INNER width, matching the source XML.
    cx, cy = self.size[0] / 2.0, self.size[1] / 2.0
    bar_center = underside + self.bar_half

    for index in range(self.frame_count):
      ox = cx + (index - (self.frame_count - 1) / 2.0) * self.frame_spacing
      # 6 legs: x = -half, 0, +half; y = +-half.
      for lx in (-half, 0.0, half):
        for ly in (-side_half, side_half):
          leg = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.leg_half, self.leg_half, underside / 2.0),
            pos=(ox + lx, cy + ly, underside / 2.0),
            rgba=(0.35, 0.62, 0.20, 1.0),
            contype=1, conaffinity=1,
          )
          geoms.append(TerrainGeometry(geom=leg))
      # 3 crossbars across y at x = -half, 0, +half.
      for lx in (-half, 0.0, half):
        bar = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(self.bar_half, side_half + self.leg_half, self.bar_half),
          pos=(ox + lx, cy, bar_center),
          rgba=(0.82, 0.34, 0.22, 1.0),
          contype=1, conaffinity=1,
        )
        geoms.append(TerrainGeometry(geom=bar))
      # 2 longbars across x at y = +-half.
      for ly in (-side_half, side_half):
        bar = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(half + self.bar_half, self.bar_half, self.bar_half),
          pos=(ox, cy + ly, bar_center),
          rgba=(0.82, 0.34, 0.22, 1.0),
          contype=1, conaffinity=1,
        )
        geoms.append(TerrainGeometry(geom=bar))

    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


@dataclass(kw_only=True)
class TireArrayTerrainCfg(SubTerrainCfg):
  """Competition tyre array: 2 x 4 tyres, 0.55 m spacing, lying flat."""

  rows: int = 2
  cols: int = 4
  spacing: float = 0.55
  outer_radius: float = 0.2744
  inner_radius: float = 0.1646
  height: float = 0.165
  segments: int = 16

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng, difficulty
    body = spec.body("terrain")
    base = make_plane(body, self.size, 0.0, center_zero=False)
    geoms = [TerrainGeometry(geom=base[0], color=(0.55, 0.55, 0.55, 1.0))]

    mid_radius = (self.outer_radius + self.inner_radius) / 2.0
    radial_half = (self.outer_radius - self.inner_radius) / 2.0
    seg_half = math.pi * mid_radius / self.segments * 1.08
    cx, cy = self.size[0] / 2.0, self.size[1] / 2.0

    for row in range(self.rows):
      for col in range(self.cols):
        tx = cx + (col - (self.cols - 1) / 2.0) * self.spacing
        ty = cy + (row - (self.rows - 1) / 2.0) * self.spacing
        for index in range(self.segments):
          angle = 2.0 * math.pi * index / self.segments
          quat = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
          geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(radial_half, seg_half, self.height / 2.0),
            pos=(
              tx + mid_radius * math.cos(angle),
              ty + mid_radius * math.sin(angle),
              self.height / 2.0,
            ),
            quat=quat,
            rgba=(0.12, 0.12, 0.12, 1.0),
            contype=1, conaffinity=1,
          )
          geoms.append(TerrainGeometry(geom=geom))
    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


def lgdog1_race_depth_terrain_cfg() -> TerrainGeneratorCfg:
  """2027 race terrain with competition-sized overhang and tyre obstacles."""
  cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=12,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={},
  )
  cfg.sub_terrains = {
    "start_flat": BoxFlatTerrainCfg(proportion=RACE_DEPTH_PROPORTIONS["start_flat"]),
    "stepping_stones": BoxSteppingStonesTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["stepping_stones"],
      stone_size_range=(0.2, 0.2),
      stone_distance_range=(0.2, 0.4),
      stone_height=0.1,
      stone_height_variation=0.0,
      stone_size_variation=0.0,
      displacement_range=0.0,
      floor_depth=1.5,
      platform_width=0.8,
      border_width=0.25,
    ),
    "balance_beam": BoxNarrowBeamsTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["balance_beam"],
      num_beams=3,
      beam_width_range=(0.4, 0.4),
      beam_height=0.1,
      spacing=0.9,
      platform_width=0.8,
      border_width=0.25,
      floor_depth=1.5,
    ),
    "gap_steps": BoxOpenStairsTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["gap_steps"],
      step_height_range=(0.10, 0.14),
      step_width_range=(0.30, 0.36),
      platform_width=0.8,
      border_width=0.25,
      step_thickness=0.05,
      inverted=False,
    ),
    "brick_pit": BoxRandomSpreadTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["brick_pit"],
      num_boxes=36,
      box_width_range=(0.10, 0.20),
      box_length_range=(0.10, 0.20),
      box_height_range=(0.0, 0.06),
      box_yaw_range=(0.0, 360.0),
      add_floor=True,
      platform_width=0.8,
      border_width=0.25,
    ),
    "slalom_posts": BoxRandomSpreadTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["slalom_posts"],
      num_boxes=7,
      box_width_range=(0.05, 0.06),
      box_length_range=(0.05, 0.06),
      box_height_range=(0.45, 0.55),
      box_yaw_range=(0.0, 360.0),
      add_floor=True,
      platform_width=0.8,
      border_width=0.25,
    ),
    "high_wall": BoxRandomSpreadTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["high_wall"],
      num_boxes=3,
      box_width_range=(0.5, 0.8),
      box_length_range=(0.04, 0.04),
      box_height_range=(0.28, 0.30),
      box_yaw_range=(0.0, 0.0),
      add_floor=True,
      platform_width=0.8,
      border_width=0.25,
    ),
    "clearance_frame": BoxClearanceFrameTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["clearance_frame"]
    ),
    "tire_array": TireArrayTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["tire_array"]
    ),
    "rough": HfRandomUniformTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["rough"],
      noise_range=(0.02, 0.06),
      noise_step=0.02,
      border_width=0.25,
    ),
  }
  return cfg
