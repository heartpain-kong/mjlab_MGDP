"""2027 obstacle lessons with supported geometry and graded dimensions.

Hard endpoints follow the bundled race_obstacle_2027.xml. These isolated
skills and joined courses are training approximations, not competition scoring.
"""

from dataclasses import dataclass
import numpy as np
import mujoco
from mjlab.terrains import TerrainGeneratorCfg, HfPerlinNoiseTerrainCfg, HfRandomUniformTerrainCfg
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from .race_depth_terrain_cfg import BoxClearanceFrameTerrainCfg, TireArrayTerrainCfg

KINDS = ("start_flat", "stepping_stones", "balance_beam", "gap_steps", "high_wall",
         "clearance_frame", "tire_array", "random_hills", "mjlab_rough", "slalom_brick")
_COMPONENT_KINDS = ("brick_pit", "slalom_posts")  # Internal parts of the joined course only.
START_X = 1.0
GOAL_X = 5.3
LANE_HALF_WIDTH = 0.28
LAYOUT_VERSION = 7  # Remove separate pit/slalom/old rough; add two heightfield lessons.
BEAM_EDGE_GAP = 0.4
WALL_CENTRES = (2.3, 3.7, 5.1)
WALL_HALF_THICKNESS = .04


def _lateral_lanes(size_y, width, gap):
  """Centred repeats with at least 35cm clearance to either cell boundary."""
  pitch = width + gap
  count = int(np.floor((size_y/2-.35-width/2+1e-8)/pitch))
  return [(i,size_y/2+i*pitch) for i in range(-count,count+1)]


def lesson_route(kind, size=(8., 8.)):
  """Return ordered waypoints relative to spawn for single and joined lessons."""
  cy = size[1]/2
  if kind in ("slalom_posts","slalom_brick"):
    points = [(1.,cy), (1.9,cy-.6), (2.4,cy-.3), (3.3,cy-.9),
              (4.2,cy-.3), (4.65,cy-.9), (5.4,cy-.9), (5.4,cy-.1),
              (4.8,cy+.3), (5.4,cy+1.2), (5.1,cy+1.9)]
    if kind == "slalom_brick":
      # Turn on the pit floor, then cross its diagonal rail while moving;
      # do not put a slowdown/turning waypoint on top of that rail.
      points += [(5.1,cy+2.4),(6.4,cy+2.4),(7.,cy+1.4),(7.,cy+.15)]
  elif kind == "stepping_stones":
    points = [(START_X,cy),(7.5,cy)]
  elif kind == "brick_pit":
    points = [(1.,cy), (2.,cy-.9), (3.,cy-1.4), (3.,cy+.5),
              (4.,cy+.5), (5.6,cy+.5)]
  else:
    points = [(START_X,cy), (START_X+GOAL_X,cy)]
  return tuple((x-START_X, y-cy) for x,y in points)


@dataclass(kw_only=True)
class PassageTerrainCfg(SubTerrainCfg):
  kind: str = "start_flat"
  # Included in checkpoint fingerprints, even when the dataclass shape and
  # terrain dimensions otherwise look the same as the old implementation.
  layout_version: int = LAYOUT_VERSION

  def wall_parameters(self, difficulty):
    return WALL_CENTRES, .04 + .26 * difficulty

  def gap_entry_style(self):
    return "ramp"

  def stone_stagger(self, difficulty):
    return .3*difficulty

  def stone_parameters(self, difficulty):
    return .02+.08*difficulty, .36-.16*difficulty

  def _heightfield(self, difficulty, spec, rng):
    """Use native mjlab generators, adding passage entry/exit landing pads."""
    d = float(np.clip(difficulty, 0., 1.))
    if self.kind == "random_hills":
      source = HfPerlinNoiseTerrainCfg(size=self.size, height_range=(.025,.20),
        octaves=3, persistence=.4, scale=4., horizontal_scale=.1, resolution=.1, border_width=.2)
    else:
      source = HfRandomUniformTerrainCfg(size=self.size, noise_range=(.02,.10),
        noise_step=.02, horizontal_scale=.1, vertical_scale=.005, border_width=.25)
    output = source.function(d,spec,rng)
    item = output.geometries[0]
    field = item.hfield
    assert field is not None and item.geom is not None
    heights = np.array(field.userdata).reshape(field.nrow,field.ncol)
    # MuJoCo heightfield columns run along physical X, rows along Y.
    x = np.linspace(0.,self.size[0],field.ncol)
    y = np.linspace(0.,self.size[1],field.nrow)
    def smooth_ramp(values):
      u = np.clip(values,0.,1.)
      return u*u*(3.-2.*u)
    forward = smooth_ramp(np.minimum((x-1.7)/.5,(5.5-x)/.5))
    lateral = smooth_ramp(np.minimum((y-.35)/.35,(self.size[1]-.35-y)/.35))
    heights *= lateral[:,None]*forward[None,:]
    heights /= max(float(heights.max()),1e-12)
    field.userdata = heights.ravel().tolist()
    if self.kind == "mjlab_rough":
      # Native random-uniform ignores difficulty: grade its physical amplitude.
      field.size[2] *= .2+.8*d
    item.geom.name = f"lesson_{self.kind}_{len(spec.body('terrain').geoms)}_field"
    item.geom.material = color_by_height(spec,heights,field.name+"_passage",heights)
    return TerrainOutput(origin=np.array([START_X,self.size[1]/2,0.]),geometries=output.geometries)

  def function(self, difficulty, spec, rng):
    if self.kind not in (*KINDS,*_COMPONENT_KINDS) or self.layout_version != LAYOUT_VERSION:
      raise ValueError("Unknown passage kind/layout version")
    if self.size[0] < 8 or self.size[1] < 6:
      raise ValueError("2027 lessons require a cell of at least 8 x 6 m")
    if self.kind in ("random_hills","mjlab_rough"):
      return self._heightfield(difficulty,spec,rng)
    if self.kind == "slalom_brick":
      if self.size[1] < 8:
        raise ValueError("Joined slalom/brick lesson requires at least 8 x 8 m")
      slalom=PassageTerrainCfg(kind="slalom_posts",size=self.size).function(difficulty,spec,rng)
      pit=PassageTerrainCfg(kind="brick_pit",size=self.size).function(difficulty,spec,rng)
      geoms=list(slalom.geometries)
      old_origin=np.array([2.5,self.size[1]/2-1.,0.])
      new_origin=np.array([5.5,self.size[1]/2+2.9,0.])
      rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
      quat=np.array([2**-.5,0.,0.,-2**-.5])
      for item in pit.geometries:
        if item.geom.name.endswith("lesson_floor"):
          spec.delete(item.geom)  # Keep one shared floor for the whole course.
          continue
        item.geom.pos = rotation@(np.array(item.geom.pos)-old_origin)+new_origin
        rotated=np.empty(4)
        mujoco.mju_mulQuat(rotated,quat,np.array(item.geom.quat))
        item.geom.quat=rotated
        geoms.append(item)
      return TerrainOutput(origin=slalom.origin,geometries=geoms)
    body, geoms = spec.body("terrain"), []
    prefix = f"lesson_{self.kind}_{len(body.geoms)}_"
    cy = self.size[1]/2
    d = float(np.clip(difficulty, 0, 1))
    green = (.35,.62,.20,1.)
    def box(name, x, y, sx, sy, top, bottom=0., color=green, yaw=0.):
      geom = body.add_geom(name=prefix+name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x,y,(top+bottom)/2), size=(sx,sy,(top-bottom)/2),
        quat=(np.cos(yaw/2),0.,0.,np.sin(yaw/2)), rgba=color, contype=1, conaffinity=1)
      geoms.append(TerrainGeometry(geom=geom))

    # Actual obstacles stand on the arena floor, not an artificial 45cm pit.
    if self.kind not in ("clearance_frame", "tire_array"):
      box("lesson_floor", self.size[0]/2,cy,self.size[0]/2,self.size[1]/2,
          0.,-.1,color=(.55,.55,.55,1.))
    if self.kind == "stepping_stones":
      height, width = self.stone_parameters(d)
      for lane,y in _lateral_lanes(self.size[1],.8,.2):
        box(f"stones_entry_{lane}",1.8,y,.4,.4,height)
        box(f"stones_exit_{lane}",6.3,y,.4,.4,height)
        for row in range(2):
          # Repeat the reference unit along X, then repeat paired lanes along Y.
          for i,x in enumerate(2.4+row*self.stone_stagger(d)+np.arange(6)*.6):
            box(f"stone_{lane}_{row}_{i}",float(x),y+(1 if row==0 else -1)*.2,
                width/2,width/2,height)
    elif self.kind == "balance_beam":
      width = .7-.3*d
      for lane,y in _lateral_lanes(self.size[1],width,BEAM_EDGE_GAP):
        box(f"beam_lane_{lane}",4.,y,1.5,width/2,.02+.08*d)
    elif self.kind == "gap_steps":
      height, ramp_length, gap = .06+.30*d, .8+.189092*d, .04+.32*d
      # Same platforms/gap/descent for ramp and three-step ascent variants.
      x0, half_width = 2., .4
      style = self.gap_entry_style()
      if style not in ("ramp", "stairs"):
        raise ValueError("Gap entry style must be ramp or stairs")
      if style == "ramp":
        mesh = spec.add_mesh(name=prefix+"ramp", uservert=[
          x0,-half_width,0, x0,half_width,0,
          x0+ramp_length,-half_width,0, x0+ramp_length,half_width,0,
          x0+ramp_length,-half_width,height, x0+ramp_length,half_width,height])
      platform0, platform1 = x0+ramp_length, x0+ramp_length+.8+gap
      for lane,y in _lateral_lanes(self.size[1],.8,0.):
        # Touching side faces form a continuous-width obstacle; share one mesh.
        if style == "ramp":
          ramp = body.add_geom(name=prefix+f"gap_ramp_{lane}",type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,pos=(0,y,0),rgba=green,contype=1,conaffinity=1)
          geoms.append(TerrainGeometry(geom=ramp))
        else:
          tread = ramp_length/3
          for i in range(3):
            box(f"gap_rise_{lane}_{i}",x0+(i+.5)*tread,y,tread/2,.4,height*(i+1)/3)
        box(f"gap_platform0_{lane}",platform0+.4,y,.4,.4,height)
        box(f"gap_platform1_{lane}",platform1+.4,y,.4,.4,height)
        for i in range(2):
          box(f"gap_tread_{lane}_{i}",platform1+.8+.16+i*.32,y,.16,.4,height*(2-i)/3)
    elif self.kind == "brick_pit":
      # Same 2x2 L footprint and boundary/diagonal rails as the supplied model.
      ox, oy = 2.5, cy-1.
      base, rail = .01+.02*d, .025+.125*d
      box("pit_base_vertical",ox+.5,oy+1.,.5,1.,base,color=(.36,.24,.12,1.))
      box("pit_base_horizontal",ox+1.5,oy+1.5,.5,.5,base,color=(.55,.28,.2,1.))
      for i,(x,y,sx,sy) in enumerate(((.5,0,.5,.01),(1,.5,.01,.5),
          (1.5,1,.5,.01),(2,1.5,.01,.5),(1,2,1,.01),(0,1,.01,1))):
        box(f"pit_rail_{i}",ox+x,oy+y,sx,sy,rail)
      box("pit_diagonal",ox+.5,oy+1.5,2**-.5,.01,rail,yaw=-np.pi/4)
      for x in np.arange(.1,2.,.2):
        for y in np.arange(.1,2.,.2):
          if x > 1 and y < 1:
            continue
          h = rng.uniform(.003,.008+.012*d)
          box(f"pit_fill_{len(geoms)}",ox+x,oy+y,.065,.065,base+h,base,
              color=(.5,.3,.18,1.),yaw=float(rng.uniform(-np.pi,np.pi)))
    elif self.kind == "slalom_posts":
      # Reference: six 10cm-high cylinders in an L, not 30-50cm-high walls.
      for i,(x,y) in enumerate([(2.4+i*.9,cy-.6) for i in range(4)]+
                               [(5.1,cy+.3),(5.1,cy+1.2)]):
        height, radius = .03+.07*d, .025+.025*d
        geom = body.add_geom(name=prefix+f"slalom_pole_{i}",type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          pos=(x,y,height/2),size=(radius,height/2,0),rgba=green,contype=1,conaffinity=1)
        geoms.append(TerrainGeometry(geom=geom))
    elif self.kind == "high_wall":
      # Three hurdles with room to land between them; repeat the reference
      # 0.8m wall across the cell rather than leaving an empty lateral field.
      half_lanes = int(np.floor((cy-.8)/.8+1e-8))
      centres, height = self.wall_parameters(d)
      for i,x in enumerate(centres):
        for j in range(-half_lanes,half_lanes+1):
          box(f"high_wall_{i}_{j}",x,cy+j*.8,WALL_HALF_THICKNESS,.4,height)
    elif self.kind in ("clearance_frame", "tire_array"):
      if self.kind == "clearance_frame":
        half_lanes = int(np.floor((cy-.82)/1.))
        source = BoxClearanceFrameTerrainCfg(size=self.size,frame_count=4,
          frame_spacing=1.,center_x=3.6,lane_count=2*half_lanes+1,lane_spacing=1.,
          underside_range=(.44,.29))
      else:
        # Even row count keeps the centre lane between two rows, as in the
        # reference 2x4 array. One floor for the whole field, not per tyre.
        rows = int(np.floor((self.size[1]-.8-2*.2744)/.55))+1
        rows -= rows%2
        source = TireArrayTerrainCfg(size=self.size,rows=rows,cols=6,center_x=3.75,
                                     height=.025+.14*d)
      geoms = list(source.function(d,spec,rng).geometries)
    return TerrainOutput(origin=np.array([START_X,cy,0.]),geometries=geoms)


@dataclass(kw_only=True)
class WallSkillTerrainCfg(PassageTerrainCfg):
  """Eight auxiliary training stages; the original terrain stays unchanged."""
  kind: str = "high_wall"
  heights: tuple[float, ...] = (.10, .12, .15, .18, .21, .25, .30, .35)
  counts: tuple[int, ...] = (1, 1, 1, 2, 2, 3, 3, 3)
  skill_version: int = 3

  def wall_parameters(self, difficulty):
    if (self.kind != "high_wall" or len(self.heights) != 8 or len(self.counts) != 8
        or any(h <= 0 for h in self.heights) or any(n not in (1,2,3) for n in self.counts)):
      raise ValueError("Invalid eight-stage high-wall training course")
    stage = min(7, int(float(np.clip(difficulty, 0., 1.)) * 8))
    return WALL_CENTRES[:self.counts[stage]], self.heights[stage]


@dataclass(kw_only=True)
class GapSkillTerrainCfg(PassageTerrainCfg):
  """Paired same-row gap dimensions; only the ascent shape differs."""
  kind: str = "gap_steps"
  entry_style: str = "ramp"
  skill_version: int = 2

  def gap_entry_style(self):
    return self.entry_style

  def function(self, difficulty, spec, rng):
    # The generator jitters difficulty independently for every cell. Quantizing
    # both variants to the same eight stages keeps SAME-ROW height/gap identical.
    stage = min(7, int(float(np.clip(difficulty, 0., 1.))*8))
    return super().function(stage/7, spec, rng)


@dataclass(kw_only=True)
class StrictStonesTerrainCfg(PassageTerrainCfg):
  """Competition dimensions at stage six, then two harder stages."""
  kind: str = "stepping_stones"
  skill_version: int = 4
  widths: tuple[float, ...] = (.36, .32, .28, .25, .22, .20, .18, .16)
  # ★ 2026-11-19：按比赛规则统一为 100 mm（石头 200×200×100、平台 800×800×100），
  #   难度只由石面尺寸（.36→.16）和错位决定。
  heights: tuple[float, ...] = (.10,)*8

  def stone_parameters(self, difficulty):
    if len(self.widths) != 8 or len(self.heights) != 8:
      raise ValueError("The stone curriculum requires eight width/height stages")
    # Ignore row jitter: stage n-2 is the fixed competition reference, while
    # the last two rows provide smaller landing areas and higher steps.
    stage = min(7, int(float(np.clip(difficulty, 0., 1.))*8))
    return self.heights[stage], self.widths[stage]

  def stone_stagger(self, difficulty):
    # Half of the 60cm longitudinal pitch, even at the easiest stage.
    return .3


@dataclass(kw_only=True)
class CombinedGapTerrainCfg(PassageTerrainCfg):
  """ONE cell: stairs-gap-downslope, connector, upslope-gap-downstairs."""
  kind: str = "gap_steps"
  skill_version: int = 3
  connector_length: float = .8

  def gap_parameters(self, difficulty):
    stage=min(7,int(float(np.clip(difficulty,0.,1.))*8));d=stage/7
    height,length,gap=.06+.30*d,.8+.189092*d,.04+.32*d
    return height,length,gap,length/3,.32

  def function(self, difficulty, spec, rng):
    if self.size[0] < 12 or self.size[1] < 6:
      raise ValueError("The combined gap lesson requires a cell of at least 12x6m")
    height,length,gap,up_tread,down_tread=self.gap_parameters(difficulty)
    body=spec.body('terrain');geoms=[];prefix=f'combined_gap_{len(body.geoms)}_'
    cy=self.size[1]/2;green=(.35,.62,.20,1.)
    def box(name,x,y,sx,sy,top,bottom=0.):
      g=body.add_geom(name=prefix+name,type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x,y,(top+bottom)/2),size=(sx,sy,(top-bottom)/2),rgba=green,contype=1,conaffinity=1)
      geoms.append(TerrainGeometry(geom=g))
    box('floor',self.size[0]/2,cy,self.size[0]/2,self.size[1]/2,0.,-.1)
    def ramp(name,x,y,down=False):
      top_x=x if down else x+length
      mesh=spec.add_mesh(name=prefix+name+'_mesh',uservert=[
        x,-.4,0.,x,.4,0.,x+length,-.4,0.,x+length,.4,0.,top_x,-.4,height,top_x,.4,height])
      g=body.add_geom(name=prefix+name,type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,pos=(0.,y,0.),rgba=green,contype=1,conaffinity=1)
      geoms.append(TerrainGeometry(geom=g))
    first=2.;first_top=first+3*up_tread;first_after=first_top+.8+gap
    first_end=first_after+.8+length;second=first_end+self.connector_length
    second_top=second+length;second_after=second_top+.8+gap
    for lane,y in _lateral_lanes(self.size[1],.8,0.):
      for i in range(3):
        box(f'first_rise_{lane}_{i}',first+(i+.5)*up_tread,y,up_tread/2,.4,height*(i+1)/3)
      for name,x in [('first_platform0',first_top),('first_platform1',first_after),
                     ('second_platform0',second_top),('second_platform1',second_after)]:
        box(f'{name}_{lane}',x+.4,y,.4,.4,height)
      ramp(f'first_downslope_{lane}',first_after+.8,y,down=True)
      ramp(f'second_upslope_{lane}',second,y)
      for i in range(2):
        box(f'second_tread_{lane}_{i}',second_after+.8+(i+.5)*down_tread,y,down_tread/2,.4,height*(2-i)/3)
    return TerrainOutput(origin=np.array([START_X,cy,0.]),geometries=geoms)


def lgdog1_passage_terrain_cfg():
  return TerrainGeneratorCfg(size=(8.,8.),num_rows=10,num_cols=2*len(KINDS),
    border_width=10.,curriculum=True,difficulty_range=(0.,1.),add_lights=True,
    sub_terrains={kind:PassageTerrainCfg(kind=kind,proportion=1./len(KINDS)) for kind in KINDS})
