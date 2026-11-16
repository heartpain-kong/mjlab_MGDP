"""Draw tire targets and a nominal FK pose; this is not a trained-policy rollout."""
from pathlib import Path
import json
import mujoco,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Wedge
from src.tasks.velocity.mdp.tire_training import hole_centres
from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import TIRE_INNER_APOTHEM


def main():
  root=Path(__file__).resolve().parents[1]
  out=root/'doc/analysis/tire_hole_guidance';out.mkdir(parents=True,exist_ok=True)
  plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':12,'text.color':'#20334d','axes.labelcolor':'#20334d'})
  m=mujoco.MjModel.from_xml_path(str(root/'src/assets/robots/lgdog1/xmls/lgdog1.xml'))
  d=mujoco.MjData(m)
  for i in range(m.njnt):
    name=m.joint(i).name
    if name and name.endswith('_joint'):d.qpos[m.jnt_qposadr[i]]=.2 if '_calf_' in name else 0.
  d.qpos[:7]=[2.475,0.,.312,1.,0.,0.,0.]
  mujoco.mj_forward(m,d)
  centres=hole_centres().numpy();radius=.012;safe=TIRE_INNER_APOTHEM-radius-.01
  fig,ax=plt.subplots(figsize=(15,8));fig.subplots_adjust(left=.07,right=.98,bottom=.29,top=.81)
  for y in (-.275,.275):
    for x in centres[0,:,0]:
      ax.add_patch(Wedge((x,y),.275,0,360,width=.275-TIRE_INNER_APOTHEM,facecolor='#353b45',edgecolor='white'))
      ax.add_patch(Circle((x,y),safe,color='#42b99a',alpha=.25))
      ax.add_patch(Circle((x,y),safe,fill=False,edgecolor='#128569',linestyle='--',linewidth=1.6))
      ax.plot(x,y,'+',color='#128569',ms=8)
  ax.annotate('',xy=(4.05,0),xytext=(1.45,0),arrowprops={'arrowstyle':'->','color':'#4c82b4','lw':2,'linestyle':'--'})
  ax.text(3.9,.055,'前进',color='#4c82b4',ha='center')
  trunk=m.geom('trunk_collision');p=d.geom_xpos[trunk.id];size=m.geom_size[trunk.id]
  ax.add_patch(Rectangle(p[:2]-size[:2],size[0]*2,size[1]*2,facecolor='#dae7ef',edgecolor='#4b6c83',linewidth=2,alpha=.8))
  records=[]
  for f,name in enumerate(('FR','FL','RR','RL')):
    g=m.geom(name+'_calf1_collision');p=d.geom_xpos[g.id]
    dist=np.linalg.norm(centres[f]-p[:2],axis=1);c=int(dist.argmin())
    ax.add_patch(Circle(p[:2],radius,facecolor='#d86f30',edgecolor='white',linewidth=1,zorder=5))
    ax.annotate(name,xy=p[:2],xytext=(p[0],p[1]+(.105 if f%2 else -.105)),ha='center',va='center',color='#b45120',weight='bold',arrowprops={'arrowstyle':'-','color':'#b45120'})
    records.append({'foot':name,'sphere_center':p.tolist(),'nearest_column_1_based':c+1,'distance_to_hole_center':float(dist[c]),'inside_safe_disk':bool(dist[c]<=safe)})
  ax.text(1.38,.62,'左脚 FL / RL → 左列内孔',color='#128569',va='center')
  ax.text(1.38,-.62,'右脚 FR / RR → 右列内孔',color='#128569',va='center')
  ax.set(xlim=(1.35,4.12),ylim=(-.7,.7),xlabel='沿路线距离（m，相对出生点）',ylabel='横向位置（m）',aspect='equal')
  ax.set_yticks([-.55,-.275,0,.275,.55]);ax.spines[['top','right']].set_visible(False)
  fig.suptitle('轮胎阵：身体沿中线，左右脚分别落进对应内孔',fontsize=22,weight='bold',y=.97)
  fig.text(.5,.865,'黑环 = 实际尺寸近似外形   绿色 = 留安全余量后的落脚区域   橙点 = 机器人标称关节姿态的脚心投影',ha='center',fontsize=12)
  fig.text(.07,.115,'尺寸：内孔约 Ø33 cm · 两列中心距 55 cm · 前后轮胎中心距 55 cm\n脚球半径 1.2 cm + 余量 1 cm → 安全落脚半径约 14.3 cm',fontsize=12,linespacing=1.8)
  fig.text(.56,.115,'训练反馈：每脚每孔每回合首次有效落地 +0.25\n轮胎顶部、腾空、原地重复抬脚不加分；不要求四脚同时入孔',fontsize=12,linespacing=1.8)
  fig.text(.5,.035,'几何与正向运动学示意，不是策略回放，也不代表已经学会通过。图中只显示路线两侧的中间两列轮胎。',ha='center',fontsize=11,color='#687b90')
  fig.savefig(out/'tire_hole_guidance.png',dpi=150,facecolor='white');plt.close(fig)
  report={'safe_radius_m':safe,'pose_type':'nominal_forward_kinematics_not_rollout','feet':records}
  (out/'nominal_pose.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  print(out/'tire_hole_guidance.png')


if __name__=='__main__':main()
