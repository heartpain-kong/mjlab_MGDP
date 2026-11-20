"""Eight-stage obstacle course: stage six is the bundled race reference.

Reference units may be repeated across the training tile. The two synthetic
heightfields have a training reference, rather than a competition counterpart.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from .passage_terrain_cfg import (
  PassageTerrainCfg, WallSkillTerrainCfg, CombinedGapTerrainCfg, START_X,
)
from .race_depth_terrain_cfg import BoxClearanceFrameTerrainCfg

COMPETITION_LEVEL = 5
STAGES = 8
BEAM_WIDTHS = (.70,.64,.58,.52,.46,.40,.35,.30)
BEAM_HEIGHTS = (.02,.03,.04,.06,.08,.10,.12,.15)
FRAME_CLEARANCES = (.44,.41,.38,.35,.32,.29,.27,.25)
TIRE_HEIGHTS = (.025,.05,.08,.11,.14,.165,.19,.22)
# Shared by terrain construction and optional foot-hole guidance (cell frame).
TIRE_CENTER_X = 3.75
TIRE_SPACING = .55
TIRE_COLUMNS = 4
TIRE_INNER_APOTHEM = .164647
PIT_HEIGHT_SCALES = (.20,.35,.50,.65,.85,1.,1.20,1.40)
HEIGHTFIELD_SCALES = (1.,1.,1.,1.,1.,1.,1.25,1.50)


def stage_index(difficulty):
  return min(7,int(float(np.clip(difficulty,0.,1.))*STAGES))


@lru_cache(maxsize=1)
def race_reference():
  return ET.parse(Path(__file__).resolve().parents[4]/'assets/terrains/race_obstacle_2027.xml').getroot()


def vector(element,name,default):
  return np.fromstring(element.get(name,default),sep=' ')


@dataclass(kw_only=True)
class UnifiedWallTerrainCfg(WallSkillTerrainCfg):
  # 2026-11-20：第 1 档从 10 cm 降到 6 cm，给策略一个真正可学的起点
  #（10 cm 时 3.8 万回合 0 成功、进度只有 0.23）；第 6 档仍是比赛 30 cm。
  heights: tuple[float,...] = (.06,.08,.11,.15,.21,.30,.35,.40)
  skill_version: int = 4


@dataclass(kw_only=True)
class UnifiedGapTerrainCfg(CombinedGapTerrainCfg):
  skill_version: int = 4
  heights: tuple[float,...] = (.06,.10,.16,.22,.30,.36,.42,.48)
  gaps: tuple[float,...] = (.04,.08,.14,.22,.30,.36,.42,.48)

  def gap_parameters(self,difficulty):
    s=stage_index(difficulty)
    # The bundled 20-degree wedge is 0.989092m long. Hard stages raise the
    # platform and widen the gap, while keeping both sequences inside 12m.
    length=.8+.189092*min(s/COMPETITION_LEVEL,1.)
    tread=(.32,)*6+(.30,.28)
    return self.heights[s],length,self.gaps[s],tread[s],tread[s]


@dataclass(kw_only=True)
class UnifiedObstacleTerrainCfg(PassageTerrainCfg):
  course_version: int = 4
  post_radii: tuple[float,...] = (.025,.03,.035,.04,.045,.05,.0575,.065)
  post_heights: tuple[float,...] = (.20,.23,.26,.29,.31,.34,.37,.40)
  # ★ 2026-11-19：比赛验收用官方 2×4 阵（2.2×1.1 m、高 16.5 cm），
  #   训练保持铺满整格（12×4），避免机器人从阵侧绕过去。
  competition_tires: bool = False

  def _floor(self,spec,rng):
    return list(PassageTerrainCfg(kind='start_flat',size=self.size).function(0.,spec,rng).geometries)

  def _pit(self,s,spec,rng):
    geoms=self._floor(spec,rng);body=spec.body('terrain')
    prefix=f'v4_pit_{len(body.geoms)}_';scale=PIT_HEIGHT_SCALES[s]
    template=race_reference().find(".//body[@name='brick_bark_pit']")
    for ref in template.findall('geom'):
      if ref.get('contype','1')=='0':continue
      assert ref.get('type','box')=='box'
      pos=vector(ref,'pos','0 0 0');size=vector(ref,'size','0 0 0')
      pos[:2]+=np.array([2.5,self.size[1]/2-1.]);pos[2]*=scale;size[2]*=scale
      g=body.add_geom(name=prefix+ref.get('name'),type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,size=size,quat=vector(ref,'quat','1 0 0 0'),
        rgba=vector(ref,'rgba','.5 .3 .18 1'),contype=1,conaffinity=1)
      geoms.append(TerrainGeometry(geom=g))
    return TerrainOutput(origin=np.array([START_X,self.size[1]/2,0.]),geometries=geoms)

  def _tires(self,s,spec,rng):
    geoms=self._floor(spec,rng);body=spec.body('terrain');cy=self.size[1]/2
    prefix=f'v4_tires_{len(body.geoms)}_';height=.165 if self.competition_tires else TIRE_HEIGHTS[s]
    # Reuse the exact convex sectors from the bundled model, preserving the
    # annular hole. A single convex mesh would incorrectly fill the hole.
    root=race_reference();refmesh=root.find(".//asset/mesh[@name='tire_sector']")
    vertices=vector(refmesh,'vertex','').reshape(-1,3);vertices[:,2]*=height/.165
    mesh=spec.add_mesh(name=prefix+'sector',uservert=vertices.ravel().tolist())
    sectors=root.find(".//body[@name='tire_0_0']").findall('geom')
    rows=int(np.floor((self.size[1]-.8-2*.275)/.55))+1;rows-=rows%2
    if self.competition_tires:
      # 官方规则 V2.0 表1：8 个轮胎 2×4，外径 550、内径 330、胎面宽 165，
      # 两胎侧短边 1100、四胎侧长边 2200，高度固定 165 mm。
      rows=2
    for row in range(rows):
      for col in range(4):
        center=np.array([TIRE_CENTER_X+(col-(TIRE_COLUMNS-1)/2)*TIRE_SPACING,
                         cy+(row-(rows-1)/2)*TIRE_SPACING,height/2])
        for i,ref in enumerate(sectors):
          g=body.add_geom(name=prefix+f'{row}_{col}_{i}',type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,pos=center+vector(ref,'pos','0 0 0'),
            quat=vector(ref,'quat','1 0 0 0'),rgba=(.045,.045,.05,1.),contype=1,conaffinity=1)
          geoms.append(TerrainGeometry(geom=g))
    return TerrainOutput(origin=np.array([START_X,cy,0.]),geometries=geoms)

  def _joined(self,difficulty,spec,rng):
    slalom=UnifiedObstacleTerrainCfg(kind='slalom_posts',size=self.size,
      post_radii=self.post_radii,post_heights=self.post_heights).function(difficulty,spec,rng)
    pit=UnifiedObstacleTerrainCfg(kind='brick_pit',size=self.size).function(difficulty,spec,rng)
    geoms=list(slalom.geometries)
    old_origin=np.array([2.5,self.size[1]/2-1.,0.]);new_origin=np.array([5.5,self.size[1]/2+2.9,0.])
    rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    quat=np.array([2**-.5,0.,0.,-2**-.5])
    for item in pit.geometries:
      if item.geom.name.endswith('lesson_floor'):
        spec.delete(item.geom);continue
      item.geom.pos=rotation@(np.array(item.geom.pos)-old_origin)+new_origin
      rotated=np.empty(4);mujoco.mju_mulQuat(rotated,quat,np.array(item.geom.quat));item.geom.quat=rotated
      geoms.append(item)
    return TerrainOutput(origin=slalom.origin,geometries=geoms)

  def function(self,difficulty,spec,rng):
    s=stage_index(difficulty);d=min(s/COMPETITION_LEVEL,1.)
    if self.kind=='slalom_brick':return self._joined(difficulty,spec,rng)
    if self.kind=='brick_pit':return self._pit(s,spec,rng)
    if self.kind=='tire_array':return self._tires(s,spec,rng)
    if self.kind=='clearance_frame':
      half_lanes=int(np.floor((self.size[1]/2-.82)/1.))
      result=BoxClearanceFrameTerrainCfg(size=self.size,frame_count=4,frame_spacing=1.,
        center_x=3.6,lane_count=2*half_lanes+1,lane_spacing=1.,
        underside_range=(FRAME_CLEARANCES[s],)*2).function(0.,spec,rng)
      return TerrainOutput(origin=np.array([START_X,self.size[1]/2,0.]),geometries=result.geometries)
    result=super().function(d,spec,rng)
    if self.kind=='balance_beam':
      for item in result.geometries:
        if 'beam_lane_' in item.geom.name:
          item.geom.size[1]=BEAM_WIDTHS[s]/2;item.geom.size[2]=BEAM_HEIGHTS[s]/2
          item.geom.pos[2]=BEAM_HEIGHTS[s]/2
    elif self.kind=='slalom_posts':
      for item in result.geometries:
        if 'slalom_pole_' in item.geom.name:
          item.geom.size[0]=self.post_radii[s];item.geom.size[1]=self.post_heights[s]/2
          item.geom.pos[2]=self.post_heights[s]/2
    elif self.kind in ('random_hills','mjlab_rough'):
      for item in result.geometries:
        if item.hfield is not None:item.hfield.size[2]*=HEIGHTFIELD_SCALES[s]
    return result
