"""Draw the complete curriculum top view from each cell's compiled geometry.

Compiling cells separately preserves the generator's RNG order and coordinates
without constructing one huge MjSpec. This is a geometry plot, with heightfields
coloured by their actual elevation, rather than a screenshot of a live run.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import (
  LAYOUT_VERSION, lesson_route, lgdog1_passage_terrain_cfg)

LABELS={"start_flat":"平地","stepping_stones":"梅花桩\n84桩",
  "balance_beam":"单木桥\n净距400mm","gap_steps":"沟壑台阶\n铺宽7.2m",
  "high_wall":"高墙","clearance_frame":"匍匐框架","tire_array":"轮胎阵",
  "random_hills":"随机高低起伏","mjlab_rough":"mjlab原生\n随机粗糙高度场",
  "slalom_brick":"绕桩→砖木坑\n连续路线"}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output",type=Path,default=Path(".artifacts/2027-overview"))
  parser.add_argument("--seed",type=int,default=42)
  args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.collections import PatchCollection
  from matplotlib.patches import Polygon,Circle,Rectangle
  from matplotlib.colors import LinearSegmentedColormap
  from matplotlib.font_manager import FontProperties,findfont
  font_path=Path("C:/Windows/Fonts/msyh.ttc")
  if font_path.exists():
    plt.rcParams["font.family"]=FontProperties(fname=str(font_path)).get_name()
  else:
    font=findfont(FontProperties(family=["Noto Sans CJK SC","WenQuanYi Micro Hei","DejaVu Sans"]))
    plt.rcParams["font.family"]=FontProperties(fname=font).get_name()
  plt.rcParams["axes.unicode_minus"]=False
  cfg=lgdog1_passage_terrain_cfg();rng=np.random.default_rng(args.seed)
  names=list(cfg.sub_terrains);sub_cfgs=list(cfg.sub_terrains.values())
  for sub in sub_cfgs: sub.size=cfg.size
  proportions=np.array([sub.proportion for sub in sub_cfgs]);cdf=np.cumsum(proportions/proportions.sum())
  indices=np.searchsorted(cdf,np.arange(cfg.num_cols)/cfg.num_cols+.001,side="right")
  sx,sy=cfg.size;extent_x=cfg.num_rows*sx;extent_y=cfg.num_cols*sy
  fig,ax=plt.subplots(figsize=(23,13),layout="constrained")
  ax.set_facecolor("#e5e9eb")
  patches=[];colours=[];tops=[];records=[];total_geoms=0
  cmap=LinearSegmentedColormap.from_list("terrain_height",["#d9eeee","#66ad85","#d3bd7d","#b47952"])
  for col,sub_index in enumerate(indices):
    kind=names[sub_index];sub=sub_cfgs[sub_index]
    for row in range(cfg.num_rows):
      difficulty=cfg.difficulty_range[0]+(row+rng.uniform())/cfg.num_rows*(cfg.difficulty_range[1]-cfg.difficulty_range[0])
      corner=np.array([row*sx-extent_x/2,col*sy-extent_y/2,0.])
      spec=mujoco.MjSpec();spec.worldbody.add_body(name="terrain")
      result=sub.function(difficulty,spec,rng)
      model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
      total_geoms+=model.ngeom
      for i in range(model.ngeom):
        position=data.geom_xpos[i]+corner;rotation=data.geom_xmat[i].reshape(3,3)
        geom_type=model.geom_type[i];size=model.geom_size[i]
        if geom_type==mujoco.mjtGeom.mjGEOM_HFIELD:
          hf=model.geom_dataid[i];nr=model.hfield_nrow[hf];nc=model.hfield_ncol[hf]
          address=model.hfield_adr[hf]
          h=model.hfield_data[address:address+nr*nc].reshape(nr,nc)*model.hfield_size[hf,2]+position[2]
          hx,hy=model.hfield_size[hf,:2]
          ax.imshow(h.T,origin="upper",extent=(position[1]-hy,position[1]+hy,position[0]+hx,position[0]-hx),
            cmap=cmap,vmin=0.,vmax=.2,zorder=1,interpolation="bilinear")
          continue
        if geom_type==mujoco.mjtGeom.mjGEOM_BOX:
          local=np.array([[-size[0],-size[1],size[2]],[size[0],-size[1],size[2]],
            [size[0],size[1],size[2]],[-size[0],size[1],size[2]]])
          world=local@rotation.T+position
          patch=Polygon(world[:,[1,0]],closed=True)
          top=position[2]+np.abs(rotation[2])@size
        elif geom_type==mujoco.mjtGeom.mjGEOM_CYLINDER:
          patch=Circle(position[[1,0]],size[0]);top=position[2]+size[1]
        elif geom_type==mujoco.mjtGeom.mjGEOM_MESH:
          mesh=model.geom_dataid[i];adr=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
          vertices=model.mesh_vert[adr:adr+n]@rotation.T+position
          xy=vertices[:,[1,0]];patch=Polygon(xy[ConvexHull(xy).vertices],closed=True);top=vertices[:,2].max()
        else:
          raise ValueError(f"Unsupported terrain geometry type {geom_type}")
        patches.append(patch);colours.append(model.geom_rgba[i]);tops.append(top)
      points=np.array(lesson_route(kind))+result.origin[:2]+corner[:2]
      ax.plot(points[:,1],points[:,0],color="#df7520",lw=.55,zorder=4)
      ax.scatter(points[0,1],points[0,0],s=3,c="#008f71",zorder=5)
      ax.scatter(points[-1,1],points[-1,0],s=3,c="#d94d57",zorder=5)
      records.append(dict(row=row,col=col,kind=kind,difficulty=float(difficulty),geoms=model.ngeom))
    print(f"[overview] column {col+1}/{cfg.num_cols}: {kind}",flush=True)
  order=np.argsort(tops,kind="stable")
  ax.add_collection(PatchCollection([patches[i] for i in order],facecolors=np.array(colours)[order],
    edgecolors="none",zorder=2))
  for col in range(cfg.num_cols+1):
    y=col*sy-extent_y/2
    ax.plot([y,y],[-extent_x/2,extent_x/2],color="#647789",alpha=.6 if col%2==0 else .28,lw=.6,zorder=6)
  for row in range(cfg.num_rows+1):
    x=row*sx-extent_x/2
    ax.plot([-extent_y/2,extent_y/2],[x,x],color="#647789",alpha=.3,lw=.5,zorder=6)
  ax.add_patch(Rectangle((-extent_y/2,-extent_x/2),extent_y,sx,fill=False,edgecolor="#00a6c7",lw=1.7,zorder=7))
  centers=[];labels=[]
  for i,kind in enumerate(names):
    cols=np.where(indices==i)[0];centers.append((cols.mean()+.5)*sy-extent_y/2);labels.append(LABELS[kind])
  ax.set_xticks(centers,labels,fontsize=11);ax.xaxis.tick_top();ax.tick_params(axis="x",length=0,pad=10)
  ax.set_yticks((np.arange(cfg.num_rows)+.5)*sx-extent_x/2,
    [f"第{r+1}行\n{r/cfg.num_rows:.1f}–{(r+1)/cfg.num_rows:.1f}" for r in range(cfg.num_rows)],fontsize=9)
  ax.set(xlim=(-extent_y/2-1,extent_y/2+1),ylim=(extent_x/2+1,-extent_x/2-1))
  ax.set_title(f"2027 MGDP 完整训练俯视排布 · 布局{LAYOUT_VERSION}\n{cfg.num_rows}行 × {cfg.num_cols}列 = {len(records)}块 ｜ 每块8×8m ｜ 按当前碰撞几何绘制",fontsize=18,pad=64)
  ax.set_ylabel("课程难度向下递增 ↓",fontsize=13)
  ax.set_xlabel("青框为初始难度行；橙线为各回合指令路线，绿点起点、红点终点。高度场颜色表示实际高度。\n砖木坑和绕桩仅保留最右侧连续课；外围平地未画出。seed="+str(args.seed),fontsize=11,labelpad=12)
  fig.savefig(args.output/"full_topdown.png",dpi=180,bbox_inches="tight",pad_inches=.15);plt.close(fig)
  assert len(records)==cfg.num_rows*cfg.num_cols
  metadata=dict(layout_version=LAYOUT_VERSION,seed=args.seed,rows=cfg.num_rows,cols=cfg.num_cols,
    geoms=total_geoms,records=records)
  (args.output/"full_topdown_metrics.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
  print(args.output.resolve(),flush=True)


if __name__=="__main__":
  main()
