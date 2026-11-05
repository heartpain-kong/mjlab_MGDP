"""Render reproducible MuJoCo previews of the 2027 skill-training terrain."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg, KINDS, lesson_route, LAYOUT_VERSION


def build_lesson(kind, difficulty, *, route=False):
  spec=mujoco.MjSpec();spec.worldbody.add_body(name="terrain")
  PassageTerrainCfg(kind=kind,size=(8.,8.)).function(difficulty,spec,np.random.default_rng(2027))
  spec.worldbody.add_light(pos=(3,2,6),dir=(0,0,-1),diffuse=(.8,.8,.8),ambient=(.4,.4,.4))
  if route:
    points=np.array(lesson_route(kind))+[1.,4.]
    for a,b in zip(points[:-1],points[1:]):
      spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(*a,.55,*b,.55),size=(.015,0,0),rgba=(1,.35,.05,1),contype=0,conaffinity=0)
  model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
  return model,data


def render(model,data,top=False,kind=None):
  camera=mujoco.MjvCamera();camera.lookat[:]=(3.7,4.,.1)
  camera.distance=6.8;camera.azimuth=90 if top else 135;camera.elevation=-90 if top else -45
  centres=data.geom_xpos[np.min(model.geom_size[:,:2],axis=1)<3.]
  dense=len(centres)>0 and np.ptp(centres[:,1])>5.
  if dense:
    camera.lookat[:]=(3.6,4.,.1);camera.distance=9.;camera.azimuth=0;camera.elevation=-60
  if top:
    camera.lookat[:]=(3.4,4.1,0.);camera.distance=5.2
    if dense:
      camera.lookat[:]=(3.6,4.,0.);camera.distance=8.;camera.azimuth=0
    camera.elevation=-90
  if kind in ("stepping_stones","slalom_brick"):
    camera.lookat[:]=(4.2,4.6 if kind=="slalom_brick" else 4.,0.)
    camera.distance=8.5 if top else 9.
    camera.azimuth=90 if top else 135
    camera.elevation=-90 if top else -55
  with mujoco.Renderer(model,height=280,width=420,max_geom=max(10000,model.ngeom+20)) as renderer:
    renderer.update_scene(data,camera=camera)
    if top:
      renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    return renderer.render().copy()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output",type=Path,default=Path(".artifacts/2027-training"))
  args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  kinds=[k for k in KINDS if k not in ("start_flat","rough")]
  fig,axes=plt.subplots(len(kinds),3,figsize=(12,2.8*len(kinds)),layout="constrained")
  metrics={"layout_version":LAYOUT_VERSION}
  for row,kind in enumerate(kinds):
    for col,difficulty in enumerate((0.,.5,1.)):
      model,data=build_lesson(kind,difficulty)
      axes[row,col].imshow(render(model,data,kind=kind));axes[row,col].axis("off")
      axes[row,col].set_title(f"{kind} | difficulty {difficulty:g}",fontsize=10)
      metrics[f"{kind}/{difficulty:g}"]={"geoms":model.ngeom}
    metrics[kind]={"route":lesson_route(kind),"length_m":float(np.linalg.norm(
      np.diff(np.array(lesson_route(kind)),axis=0),axis=1).sum())}
  fig.savefig(args.output/"lessons.png",dpi=150);plt.close(fig)
  fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
  for ax,kind in zip(axes,("slalom_posts","brick_pit","slalom_brick")):
    model,data=build_lesson(kind,1.,route=True)
    ax.imshow(render(model,data,top=True,kind=kind));ax.axis("off")
    ax.set_title(kind+" | orange: ordered route (illustration)")
  fig.savefig(args.output/"routes.png",dpi=150);plt.close(fig)
  fig,axes=plt.subplots(3,2,figsize=(12,9),layout="constrained")
  for row,kind in enumerate(("high_wall","clearance_frame","tire_array")):
    model,data=build_lesson(kind,1.)
    for col,top in enumerate((False,True)):
      axes[row,col].imshow(render(model,data,top=top));axes[row,col].axis("off")
      axes[row,col].set_title(kind+" | "+("top view" if top else "dense target layout"))
  fig.savefig(args.output/"dense_lessons.png",dpi=150);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(13,8.6),layout="constrained")
  for row,kind in enumerate(("slalom_brick","stepping_stones")):
    for col,top in enumerate((False,True)):
      model,data=build_lesson(kind,1.,route=top)
      axes[row,col].imshow(render(model,data,top=top,kind=kind));axes[row,col].axis("off")
      label="Slalom -> brick pit: ONE continuous lesson" if row==0 else "Extended stones: 2 rows x 6 = 12 stones"
      axes[row,col].set_title(label+(" | ordered route" if top else " | target geometry"),fontsize=11)
  fig.suptitle("2027 training layout 5 | orange = commanded route, not measured policy trajectory",fontsize=13)
  fig.savefig(args.output/"joined_and_stones.png",dpi=160);plt.close(fig)
  (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
  print(args.output.resolve())


if __name__=="__main__":
  main()
