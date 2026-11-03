"""Depth-perception variant of the 2027 race terrain.

`race_terrain_cfg.py` is built for the HIM height-scan contract and therefore
only contains obstacles that a heightfield can represent.  A depth camera can
also see **overhangs and head-level geometry**, so this variant adds the two
obstacles that were missing:

* 限高框架 ``clearance_frame``: two posts + an overhead beam (0.36 -> 0.28 m,
  difficulty-dependent).  The robot has to duck / crawl under it.
* 轮胎阵 ``tire_ring``: a ring of tyre blocks lying on the ground; the depth
  camera sees the hole and the rim, the policy can step into / over it.

Only the MGDP (depth) task uses this terrain.  The HIM task keeps the
heightfield-only variant unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

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
  "start_flat": 0.08,
  "stepping_stones": 0.14,
  "balance_beam": 0.12,
  "gap_steps": 0.12,
  "brick_pit": 0.12,
  "slalom_posts": 0.10,
  "high_wall": 0.08,
  "clearance_frame": 0.12,
  "tire_ring": 0.08,
  "rough": 0.04,
}


@dataclass(kw_only=True)
class BoxClearanceFrameTerrainCfg(SubTerrainCfg):
  """Flat cell with two posts and an overhead beam (限高框架)."""

  beam_height_range: tuple[float, float] = (0.36, 0.28)
  """Beam underside height, interpolated from easy to hard difficulty."""

  beam_length: float = 1.40
  beam_width: float = 0.16
  beam_thickness: float = 0.04
  post_size: float = 0.04

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    base = make_plane(body, self.size, 0.0, center_zero=False)
    geoms = [TerrainGeometry(geom=base[0], color=(0.55, 0.55, 0.55, 1.0))]

    height = self.beam_height_range[0] + difficulty * (
      self.beam_height_range[1] - self.beam_height_range[0]
    )
    cx, cy = self.size[0] / 2.0, self.size[1] / 2.0

    for offset in (-self.beam_length / 2.0, self.beam_length / 2.0):
      post = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.post_size, self.post_size, height / 2.0),
        pos=(cx, cy + offset, height / 2.0),
        rgba=(0.35, 0.35, 0.35, 1.0),
        contype=1, conaffinity=1,
      )
      geoms.append(TerrainGeometry(geom=post))

    beam = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.beam_width / 2.0, self.beam_length / 2.0, self.beam_thickness / 2.0),
      pos=(cx, cy, height - self.beam_thickness / 2.0),
      rgba=(0.82, 0.34, 0.22, 1.0),
      contype=1, conaffinity=1,
    )
    geoms.append(TerrainGeometry(geom=beam))
    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


@dataclass(kw_only=True)
class TireRingTerrainCfg(SubTerrainCfg):
  """Flat cell with a tyre-like ring lying on the ground (轮胎阵)."""

  ring_radius_range: tuple[float, float] = (0.22, 0.28)
  tube: float = 0.035
  height: float = 0.10
  segments: int = 12

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    base = make_plane(body, self.size, 0.0, center_zero=False)
    geoms = [TerrainGeometry(geom=base[0], color=(0.55, 0.55, 0.55, 1.0))]

    radius = self.ring_radius_range[0] + difficulty * (
      self.ring_radius_range[1] - self.ring_radius_range[0]
    )
    cx, cy = self.size[0] / 2.0, self.size[1] / 2.0
    seg_len = 2.0 * np.pi * radius / self.segments * 1.15

    for index in range(self.segments):
      angle = 2.0 * np.pi * index / self.segments
      yaw = angle + np.pi / 2.0
      quat = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(seg_len / 2.0, self.tube, self.height / 2.0),
        pos=(cx + radius * np.cos(angle), cy + radius * np.sin(angle), self.height / 2.0),
        quat=quat,
        rgba=(0.12, 0.12, 0.12, 1.0),
        contype=1, conaffinity=1,
      )
      geoms.append(TerrainGeometry(geom=geom))
    return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


def lgdog1_race_depth_terrain_cfg() -> TerrainGeneratorCfg:
  """2027 race terrain with the two overhang/ring obstacles added."""
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
    "tire_ring": TireRingTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["tire_ring"]
    ),
    "rough": HfRandomUniformTerrainCfg(
      proportion=RACE_DEPTH_PROPORTIONS["rough"],
      noise_range=(0.02, 0.06),
      noise_step=0.02,
      border_width=0.25,
    ),
  }
  return cfg
