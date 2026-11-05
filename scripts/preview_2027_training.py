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
  if kind in ("stepping_stones","gap_steps","balance_beam","random_hills","mjlab_rough"):
    camera.lookat[:]=(4.,4.,0.)
    camera.distance=9.5
    camera.azimuth=90 if top else 135
    camera.elevation=-90 if top else -55
  with mujoco.Renderer(model,height=280,width=420,max_geom=max(10000,model.ngeom+20)) as renderer:
    renderer.update_scene(data,camera=camera)
    if top or kind in ("random_hills","mjlab_rough"):
      renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    return renderer.render().copy()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output",type=Path,default=Path(".artifacts/2027-training"))
  args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  kinds=[k for k in KINDS if k != "start_flat"]
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
  for ax,kind in zip(axes,("slalom_brick","random_hills","mjlab_rough")):
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
      label="Slalom -> brick pit: ONE continuous lesson" if row==0 else "Dense stones: 7 paired courses x 12 = 84 stones"
      axes[row,col].set_title(label+(" | ordered route" if top else " | target geometry"),fontsize=11)
  fig.suptitle(f"2027 training layout {LAYOUT_VERSION} | orange = commanded route, not measured policy trajectory",fontsize=13)
  fig.savefig(args.output/"joined_and_stones.png",dpi=160);plt.close(fig)
  fig,axes=plt.subplots(3,2,figsize=(13,12.5),layout="constrained")
  labels={"stepping_stones":"Stones: 7 paired courses / 84 stones",
          "gap_steps":"Gap course: 9 adjacent units / 7.2m wide",
          "balance_beam":"Beams: 400mm EDGE gap / 9 lanes at target difficulty"}
  for row,kind in enumerate(labels):
    for col,top in enumerate((False,True)):
      model,data=build_lesson(kind,1.,route=top)
      axes[row,col].imshow(render(model,data,top=top,kind=kind));axes[row,col].axis("off")
      axes[row,col].set_title(labels[kind]+(" | top" if top else " | perspective"),fontsize=11)
  fig.suptitle(f"2027 dense training layout {LAYOUT_VERSION} | orange = central commanded route",fontsize=14)
  fig.savefig(args.output/"dense_skills.png",dpi=150);plt.close(fig)
  from matplotlib.patches import Rectangle
  fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
  for ax,difficulty in zip(axes,(0.,.5,1.)):
    model,_=build_lesson("balance_beam",difficulty)
    beams=[model.geom(i) for i in range(model.ngeom) if "beam_lane_" in model.geom(i).name]
    width=beams[0].size[1]*2;pitch=width+.4
    for geom in beams:
      if abs(geom.pos[1]-4.)>pitch+1e-6: continue
      ax.add_patch(Rectangle(geom.pos[:2]-geom.size[:2],*(2*geom.size[:2]),
        facecolor="#72ae44",edgecolor="#375b23"))
    low=4.+width/2;high=4.+pitch-width/2
    ax.annotate("",xy=(3.7,high),xytext=(3.7,low),arrowprops={"arrowstyle":"<->","color":"#b44711"})
    ax.text(3.85,(low+high)/2,"400 mm",color="#b44711",va="center",fontsize=11)
    ax.set(xlim=(2.3,5.7),ylim=(2.3,5.7),aspect="equal",xlabel="forward X (m)",ylabel="lateral Y (m)",
      title=f"difficulty {difficulty:g}: width {width*1000:.0f}mm")
    ax.grid(alpha=.15)
  fig.suptitle("Bridge edges stay 400mm apart at EVERY difficulty (actual geometry)",fontsize=13)
  fig.savefig(args.output/"beam_gap.png",dpi=150,bbox_inches="tight",pad_inches=.12);plt.close(fig)
  fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
  for row,kind in enumerate(("random_hills","mjlab_rough")):
    model,data=build_lesson(kind,1.)
    for col,top in enumerate((False,True)):
      axes[row,col].imshow(render(model,data,top=top,kind=kind));axes[row,col].axis("off")
      axes[row,col].set_title(kind+(" | top" if top else " | actual height"))
    xs=np.linspace(0.,8.,321)
    heights=[1.-mujoco.mj_ray(model,data,np.array([x,4.,1.]),np.array([0.,0.,-1.]),None,1,-1,None) for x in xs]
    axes[row,2].plot(xs,heights,color="#177a6b");axes[row,2].fill_between(xs,heights,alpha=.15,color="#177a6b")
    axes[row,2].axvspan(0.,1.6,alpha=.15,color="#aaa");axes[row,2].axvspan(5.6,8.,alpha=.15,color="#aaa")
    axes[row,2].set(xlim=(0,8),ylim=(0,.22),xlabel="forward X (m)",ylabel="height (m)",title="centre-line collision surface")
    axes[row,2].grid(alpha=.2)
  fig.suptitle(f"Layout {LAYOUT_VERSION}: smooth random hills + native mjlab random rough heightfield")
  fig.savefig(args.output/"heightfields.png",dpi=160,bbox_inches="tight");plt.close(fig)
  (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
  print(args.output.resolve())


if __name__=="__main__":
  main()
