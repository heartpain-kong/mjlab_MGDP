"""Directed obstacle lessons: safe start, mandatory obstacle strip, safe exit."""

from dataclasses import dataclass
import numpy as np
import mujoco
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from .race_depth_terrain_cfg import BoxClearanceFrameTerrainCfg, TireArrayTerrainCfg

KINDS = ("start_flat", "stepping_stones", "balance_beam", "gap_steps", "brick_pit",
         "slalom_posts", "high_wall", "clearance_frame", "tire_array", "rough")
START_X = 1.0
GOAL_X = 5.3  # relative to start; world-local exit x=6.3, past all obstacles
LANE_HALF_WIDTH = 0.28


@dataclass(kw_only=True)
class PassageTerrainCfg(SubTerrainCfg):
  kind: str = "start_flat"

  def function(self, difficulty, spec, rng):
    body = spec.body("terrain")
    geoms = []
    cy = self.size[1] / 2
    def box(x, y, z, sx, sy, sz):
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(x, y, z),
        size=(sx, sy, sz), contype=1, conaffinity=1)
      geoms.append(TerrainGeometry(geom=geom))
    def floor(x0, x1, top=0):
      box((x0+x1)/2, cy, top-.1, (x1-x0)/2, self.size[1]/2, .1)

    d = float(np.clip(difficulty, 0, 1))
    if self.kind in ("stepping_stones", "balance_beam", "gap_steps"):
      floor(0, 2.3)
      floor(2.3, 5.6, -.45)
      floor(5.6, self.size[0])
      if self.kind == "stepping_stones":
        gap = .08 + .32*d
        count = int(np.ceil(3.1/(.2+gap)))+1
        for x in np.linspace(2.4, 5.5, count):
          box(float(x), cy, .05, .1, .1, .05)
      elif self.kind == "balance_beam":
        box(3.95, cy, .05, 1.65, .3-.1*d, .05)
      else:
        # Open treads: progressive height/gap, with a safe final landing.
        for i, x in enumerate(np.arange(2.4, 5.6, .4)):
          height = (.04+.08*d) * (1 + i % 2)
          box(float(x), cy, height/2, .18-.04*d, .5, height/2)
    else:
      if self.kind not in ("clearance_frame", "tire_array"):
        floor(0, self.size[0])
      if self.kind in ("brick_pit", "rough"):
        for x in np.arange(2.5, 5.5, .25):
          for y in np.arange(cy-.5, cy+.51, .25):
            height = rng.uniform(.005, .02+.04*d)
            box(float(x), float(y), height/2, .12, .12, height/2)
      elif self.kind == "slalom_posts":
        for i, x in enumerate((2.8, 3.8, 4.8)):
          box(x, cy + (-1)**i*.08, .15+.1*d, .04, .04, .15+.1*d)
      elif self.kind == "high_wall":
        height = .04+.26*d
        box(4, cy, height/2, .04, .6, height/2)
      elif self.kind in ("clearance_frame", "tire_array"):
        # Reuse actual cage/ring geometry; the returned central origin is not
        # used. Our spawn remains on the clear entry pad at x=1.
        if self.kind == "clearance_frame":
          source = BoxClearanceFrameTerrainCfg(size=self.size, frame_spacing=1.7,
                                               underside_range=(.40, .28))
        else:
          source = TireArrayTerrainCfg(size=self.size, height=.05+.115*d)
        geoms = list(source.function(d, spec, rng).geometries)
      elif self.kind != "start_flat":
        raise ValueError(f"Unknown passage obstacle {self.kind!r}")
    return TerrainOutput(origin=np.array([START_X, cy, 0.]), geometries=geoms)


def lgdog1_passage_terrain_cfg():
  return TerrainGeneratorCfg(size=(8., 8.), num_rows=10, num_cols=20,
    border_width=10., curriculum=True, difficulty_range=(0., 1.),
    sub_terrains={kind: PassageTerrainCfg(kind=kind, proportion=.1) for kind in KINDS})
