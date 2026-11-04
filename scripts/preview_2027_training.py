"""Render reproducible MuJoCo previews of the 2027 skill-training terrain."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg, KINDS, lesson_route


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


def render(model,data,top=False):
  camera=mujoco.MjvCamera();camera.lookat[:]=(3.7,4.,.1)
  camera.distance=6.8;camera.azimuth=90 if top else 135;camera.elevation=-90 if top else -45
  if top:
    camera.lookat[:]=(3.4,4.1,0.);camera.distance=5.2
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
  metrics={}
  for row,kind in enumerate(kinds):
    for col,difficulty in enumerate((0.,.5,1.)):
      model,data=build_lesson(kind,difficulty)
      axes[row,col].imshow(render(model,data));axes[row,col].axis("off")
      axes[row,col].set_title(f"{kind} | difficulty {difficulty:g}",fontsize=10)
      metrics[f"{kind}/{difficulty:g}"]={"geoms":model.ngeom}
    metrics[kind]={"route":lesson_route(kind),"length_m":float(np.linalg.norm(
      np.diff(np.array(lesson_route(kind)),axis=0),axis=1).sum())}
  fig.savefig(args.output/"lessons.png",dpi=150);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
  for ax,kind in zip(axes,("slalom_posts","brick_pit")):
    model,data=build_lesson(kind,1.,route=True)
    ax.imshow(render(model,data,top=True));ax.axis("off")
    ax.set_title(kind+" | orange: ordered route (illustration)")
  fig.savefig(args.output/"routes.png",dpi=150);plt.close(fig)
  (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
  print(args.output.resolve())


if __name__=="__main__":
  main()
