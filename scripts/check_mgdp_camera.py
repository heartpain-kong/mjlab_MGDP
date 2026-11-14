"""Static MuJoCo wall visibility sweep; does not measure perception or policy skill."""
import argparse, csv, json, math
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.sensor import PinholeCameraPatternCfg
from src.mgdp_depth import D435iPatternCfg, pattern_for


def run(output, camera_height=.395):
    torch.set_num_threads(2)
    patterns={'legacy16':PinholeCameraPatternCfg(width=16,height=16,fovy=87),
              'corrected16':D435iPatternCfg(width=16,height=16),
              'd435i_edge16':pattern_for('d435i_edge16'),
              'd435i_native32':pattern_for('d435i_native32')}
    # Exact rotation of the task's existing mgdp_depth site.
    q=np.array([.7071067812,0.,-.7071067812,0.]);rot=np.zeros(9)
    mujoco.mju_quat2Mat(rot,q);rot=rot.reshape(3,3)
    rows=[]
    for height in (.10,.18,.30,.40):
      model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <geom name="floor" type="plane" size="10 10 .1"/>
        <geom name="wall" type="box" size=".04 .75 {height/2}" pos="1.24 0 {height/2}"/>
      </worldbody></mujoco>''')
      data=mujoco.MjData(model);wall=model.geom('wall').id
      for distance in (.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.):
       mujoco.mj_forward(model,data)
       for pitch in (-10.,-5.,0.,5.,10.):
        a=math.radians(pitch);tilt=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        for name,pattern in patterns.items():
          local=pattern.generate_rays(None,'cpu')[1].numpy().astype(np.float64)
          directions=np.ascontiguousarray(local@rot.T@tilt.T)
          ids=np.empty(len(local),dtype=np.int32);depth=np.empty(len(local),dtype=np.float64)
          mujoco.mj_multiRay(model,data,np.array([1.2-distance,0.,camera_height]),directions.reshape(-1),None,1,-1,
                             ids,depth,None,len(local),10.)
          hits=(ids==wall).reshape(pattern.height,pattern.width)
          pixels=int(hits.sum())
          if name=='d435i_edge16':pixels=int(hits.reshape(16,2,16,2).any(axis=(1,3)).sum())
          rows.append({'profile':name,'wall_height_m':height,'distance_m':distance,'pitch_deg':pitch,
                       'rays':len(local),'wall_pixels':pixels,'visible':bool(pixels)})
    summary={name:{'poses':len(v:=[r for r in rows if r['profile']==name]),
                  'visible_poses':sum(r['visible'] for r in v),
                  'visibility_fraction':sum(r['visible'] for r in v)/len(v)} for name in patterns}
    result={'kind':'ideal_static_raycast_visibility_only','camera_height_m':camera_height,
            'wall_thickness_m':.08,'wall_width_m':1.5,'summary':summary,
            'at_1_2m_pitch0':[r for r in rows if r['distance_m']==1.2 and r['pitch_deg']==0.],
            'limits':['No stereo matching, texture, noise, latency, or motion blur simulated',
                      'Not a learned encoder test or a policy success rate',
                      'Corrected16 changes FOV, pixel orientation and centre sampling together']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('MGDP_CAMERA '+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='.artifacts/mgdp-camera/visibility.json')
    p.add_argument('--camera-height',type=float,default=.395);args=p.parse_args()
    if not .1 <= args.camera_height <= 1.5:p.error('camera-height must be in [0.1,1.5] m')
    run(args.output,args.camera_height)
