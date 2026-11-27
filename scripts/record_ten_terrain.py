"""Record actual MGDP inference, one honest trial per lesson including high wall.

Playback is 25 FPS from a 50 Hz controller sampled every two steps (1x).
No motion interpolation, state teleporting, or synthetic success is used.
"""

# ============================================================
# 本脚本生成 release/media 下的十段回放视频与 results.json。
# 运行前请切到与策略匹配的代码版本:  git checkout policy-v1
# 用法:
#   python scripts/record_ten_terrain.py \
#       --checkpoint release/policy/model_12400.pt \
#       --perception release/policy/perception_3cca*.pt \
#       --output out/videos --max-seconds 20 --speed 0.6
# 依赖: imageio-ffmpeg Pillow mujoco torch；整片由 ffmpeg 合成。
# ============================================================

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tensordict import TensorDict

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.sim_memory import compact_model_allocations
from src.mgdp_checkpoint import bind_checkpoint_encoder, encoder_fingerprint
from src.tasks.velocity.mdp.mgdp_observations import reset_perception_state
from src.tasks.velocity.rl.action_monitor import ActionMonitorVecEnvWrapper
from src.camera_inset import find_camera

LABELS={"high_wall":"高墙","start_flat":"平地对照","stepping_stones":"梅花桩","balance_beam":"单木桥",
        "gap_steps":"沟壑台阶","clearance_frame":"限高笼","tire_array":"轮胎阵",
        "random_hills":"随机起伏","mjlab_rough":"随机粗糙","slalom_brick":"绕桩接砖木坑"}
REASONS={"illegal_contact":"机身触地","sustained_leg_contact":"腿部持续/严重碰撞",
         "left_lane":"离开通行路线","time_out":"超时","fell_over":"摔倒"}
_FONT_CANDIDATES=(Path("C:/Windows/Fonts/msyh.ttc"),
                   Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                   Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
FONT=next((p for p in _FONT_CANDIDATES if p.is_file()),None)
def _font(size):
  return ImageFont.truetype(str(FONT),size) if FONT is not None else ImageFont.load_default()


def draw_hud(frame, item, t, speed, progress, total, outcome=None, inset=None):
    img=Image.fromarray(frame).convert("RGBA")
    layer=Image.new("RGBA",img.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
    title=_font(26);body=_font(20)
    draw.rectangle((0,0,img.width,78),fill=(15,25,40,215))
    draw.text((18,8),f"{item['name_zh']}  |  第{item['level_index']+1}/8行  |  目标 0.60 m/s",font=title,fill="white")
    draw.text((18,46),"model_12400 + 匹配感知权重  ·  真实策略推理  ·  1倍速",font=body,fill=(190,208,230))
    draw.rectangle((0,img.height-70,img.width,img.height),fill=(15,25,40,215))
    draw.text((18,img.height-64),f"t={t:5.2f}s   实际平面速度 {speed:.2f} m/s   路线进度 {progress:.2f}/{total:.2f}m",font=body,fill="white")
    frac=float(np.clip(progress/max(total,1e-6),0,1))
    draw.rectangle((18,img.height-25,img.width-18,img.height-16),fill=(65,78,95))
    draw.rectangle((18,img.height-25,18+int((img.width-36)*frac),img.height-16),fill=(38,182,160))
    if outcome:
        large=_font(25)
        detail=_font(18)
        fine=_font(16)
        draw.rectangle((18,img.height-160,img.width-18,img.height-80),fill=(15,25,40,235))
        color=(84,231,174) if outcome[0] else (255,194,104)
        draw.text((img.width//2,img.height-143),outcome[1],font=large,fill=color,anchor="mm")
        draw.text((img.width//2,img.height-117),outcome[2],font=detail,fill="white",anchor="mm")
        draw.text((img.width//2,img.height-94),"指令速度不等于实测速度；本片未加速播放",font=fine,fill=(190,208,230),anchor="mm")
    result=Image.alpha_composite(img,layer).convert("RGB")
    if inset is not None:
        # Actual fixed RGB camera rendered from the same native pose/time.
        ih,iw=inset.shape[:2];x=result.width-iw-14;y=90
        pic=Image.fromarray(inset);result.paste(pic,(x,y+24))
        d=ImageDraw.Draw(result)
        d.rectangle((x-3,y-3,x+iw+2,y+ih+26),outline=(25,190,175),width=3)
        d.rectangle((x,y,x+iw-1,y+23),fill=(15,25,40))
        small=_font(16)
        d.text((x+7,y+2),"D435i 模拟前视 · 下俯20°",font=small,fill="white")
    return np.asarray(result)


def render_world(env, renderer, inset_renderer, native_data, camera, front_camera, index, groups):
    model=env.sim.mj_model
    native_data.qpos[:]=env.sim.data.qpos[index].cpu().numpy()
    native_data.qvel[:]=env.sim.data.qvel[index].cpu().numpy()
    # Only transforms are needed for rendering, not a second physics solve.
    mujoco.mj_kinematics(model,native_data)
    mujoco.mj_comPos(model,native_data)
    mujoco.mj_camlight(model,native_data)
    root=env.scene["robot"].data.root_link_pos_w[index].cpu().numpy()
    camera.lookat[:]=root+np.array([.40,0.,.12])
    camera.distance=2.9;camera.elevation=-26.;camera.azimuth=125.
    # Visual culling only; the Warp model/sensors/physics are untouched.
    radius=np.asarray(model.geom_rbound)
    near=np.linalg.norm(native_data.geom_xpos[:,:2]-root[:2],axis=1)<(7.+radius)
    near |= np.asarray(model.geom_type)==int(mujoco.mjtGeom.mjGEOM_PLANE)
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[:3]=1
    try:
        model.geom_group[:]=np.where(near,groups,5)
        renderer.update_scene(native_data,camera=camera,scene_option=option)
        main=renderer.render().copy()
        inset_renderer.update_scene(native_data,camera=front_camera,scene_option=option)
        front=inset_renderer.render().copy()
    finally:
        model.geom_group[:]=groups
    return main,front


def write_report(items,args,checkpoint,output,wall):
    data={"checkpoint":str(args.checkpoint.resolve()),"iteration":checkpoint["iter"],"policy_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),"perception":str(args.perception.resolve()),"perception_sha256":hashlib.sha256(args.perception.read_bytes()).hexdigest(),"seed":args.seed,"command_speed_mps":args.speed,"controller_hz":50,"video_fps":25,"wall_seconds":wall,"trials":items,"notes":["Skill V5, action scale 0.5, wall/slalom local speed limits 0.35/0.60 m/s", "One trial per terrain at modal checkpoint difficulty, deterministic seed 42, terrain rebuilt", "Single-row compact playback: termination/route calibrated to selected stone/slalom levels", "Play randomization differs from training; reset pose fixed; not a statistical evaluation", "Training stall termination disabled in play; 8s no-progress clip cutoff is not task failure", "RGB inset is simulated camera, not depth tensor"]}
    (output/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# model_12400 单次回放','', '0.6 m/s 为巡航目标，实际速度以画面显示为准；高墙/绕桩局部限速 0.35/0.60 m/s。', '技能V5、动作映射0.5；每地形选择检查点内最多环境的档位，种子42重新生成地形。右上为模拟前视RGB，不是深度输入。', '回放使用播放模式随机化，固定初始姿态且无训练停滞终止；连续8秒无5cm进展则截断录像，截断不等于环境终止。每类只有一次演示，不能当作成功率。','','|地形|档位|结果|时间s|平均速度m/s|路线进度m|原因|','|---|---:|---|---:|---:|---:|---|']
    for i in items:lines.append(f"|{i['name_zh']}|{i['level_index']+1}|{'通过' if i['passed'] else '未完成'}|{i['sim_seconds']:.2f}|{i['mean_xy_speed_mps']:.3f}|{i['route_progress_m']:.2f}/{i['route_length_m']:.2f}|{i['reason_zh']}|")
    (output/'视频实测说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=Path('E:/mjlab/输出/v/model_12400.pt'))
    p.add_argument("--perception",type=Path,default=Path('E:/mjlab/输出/v/perception_3cca6497b8e9f392ad5e47e61ea1a6e03e324d80979d8cca1a4518ad79e46e23.pt'))
    p.add_argument("--output",type=Path,default=Path('E:/mjlab/输出/model_12400_分析与回放/视频'))
    p.add_argument("--max-seconds",type=float,default=20.)
    p.add_argument("--speed",type=float,default=.6)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--width",type=int,default=960);p.add_argument("--height",type=int,default=540)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    state=checkpoint["infos"]["mgdp_passage_curriculum"]
    rows,cols=state["grid"]
    trials=[]
    for col,kind in enumerate(state["names"]):
        levels=state["levels"][state["types"]==col]
        level=int(torch.bincount(levels,minlength=rows).argmax())
        trials.append({"kind":kind,"name_zh":LABELS[kind],"column":col,"level_index":level})
    task="LgDog1-Race-MGDP-WallTraining-D435i20"
    # Generate one tile per lesson, at its saved modal 8-stage difficulty.
    # This preserves the original V4 lesson geometry and avoids compiling
    # 80 tiles on CPU. No tire guidance reward is added.
    from mjlab.terrains.terrain_generator import TerrainGenerator
    original_create=TerrainGenerator._create_terrain_geom
    selected_levels={t['kind']:t['level_index'] for t in trials}
    def selected_create(self,spec,world_position,difficulty,terrain_cfg,sub_row,sub_col):
        kind=terrain_cfg.kind
        return original_create(self,spec,world_position,(selected_levels[kind]+.5)/8,terrain_cfg,sub_row,sub_col)
    TerrainGenerator._create_terrain_geom=selected_create
    cfg=load_env_cfg(task,play=True);cfg.scene.num_envs=len(trials);cfg.seed=args.seed
    cfg.scene.terrain.terrain_generator.num_rows=rows;cfg.scene.terrain.terrain_generator.num_cols=cols
    cfg.scene.terrain.terrain_generator.sub_terrains["high_wall"].counts=(1,1,1,2,2,3,3,3)
    cfg.commands["twist"].ranges.lin_vel_x=(args.speed,args.speed)
    cfg.episode_length_s=args.max_seconds
    # Reward values do not feed actor observations; omit training-only wall reward managers in the compact one-row playback.
    from src.config.mgdp_course_contract import restore_for_play
    from src.config.mgdp_skill_training import restore_skill_for_play,configure_skill_policy
    restore_skill_for_play(cfg,checkpoint)
    restore_for_play(cfg,checkpoint)
    assert cfg.actions["joint_pos"].scale==.5
    saved_limits=tuple(checkpoint["infos"]["mgdp_obstacle_speed"]["limits"])
    assert tuple(cfg.commands["twist"].obstacle_speed_limits)==saved_limits, (
        cfg.commands["twist"].obstacle_speed_limits, saved_limits)
    print(f"[info] obstacle speed limits restored: {saved_limits}",flush=True)
    assert cfg.terminations["stone_ground_contact"].params["min_feet"]==2
    exits=cfg.terminations['stone_ground_contact'].params['exit_by_level']
    cfg.terminations['stone_ground_contact'].params['exit_by_level']=(exits[selected_levels['stepping_stones']],)+tuple(exits[1:])
    cfg.rewards={}
    cfg.events["reset_base"].params["pose_range"]={"x":(0.,0.),"y":(0.,0.),"z":(0.,0.),"yaw":(0.,0.)}
    bind_checkpoint_encoder(cfg,args.checkpoint,str(args.perception))
    wall=time.perf_counter()
    with compact_model_allocations():
        env=ManagerBasedRlEnv(cfg,device="cpu")
    renderer=None;inset_renderer=None;writers=[]
    try:
        wrapped=ActionMonitorVecEnvWrapper(env,clip_actions=1.)
        agent=load_rl_cfg(task)
        configure_skill_policy(agent,cfg)
        runner=(load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped,asdict(agent),device="cpu")
        runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"],strict=True)
        policy=runner.get_inference_policy(device="cpu")
        terrain=env.scene.terrain
        terrain.terrain_types[:]=torch.tensor([t["column"] for t in trials])
        terrain.terrain_levels[:]=torch.tensor([t["level_index"] for t in trials])
        projection=env.termination_manager.get_term_cfg("slalom_body_projection").func
        projection.radii[:]=projection.radii[selected_levels["slalom_brick"]].clone()
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        command=env.command_manager.get_term("twist")
        # Retain the task's own approach slowdown and slalom heading alignment.
        constant_update=command._update_command
        reset_perception_state(env)
        obs,_=wrapped.reset()
        # Native reset resamples the route but updates its command on the next
        # compute(). Refresh t=0 observations after applying the requested speed.
        constant_update()
        env.observation_manager.reset(torch.arange(len(trials)))
        reset_perception_state(env)
        env.obs_buf=env.observation_manager.compute(update_history=True)
        obs=TensorDict(env.obs_buf,batch_size=[len(trials)])
        assert torch.allclose(command.speed,torch.full((len(trials),),args.speed),atol=1e-6)
        assert encoder_fingerprint(env._mgdp_perception_state["module"])==checkpoint["infos"]["mgdp_perception"]["fingerprint"]
        # Preserve the first terminal state for the video instead of teleporting
        # back to spawn. Criteria themselves remain the original manager terms.
        original_reset=env._reset_idx
        def deferred_reset(ids=None):
            if ids is None:
                return original_reset(ids)
            # Active trials get their terminal frame first. Previously ended
            # worlds may reset normally; they never re-enter their video.
            finished=ids[torch.from_numpy(~active)[ids]]
            if len(finished):original_reset(finished)
        env._reset_idx=deferred_reset
        model=env.sim.mj_model;model.vis.global_.offwidth=args.width;model.vis.global_.offheight=args.height
        model.stat.extent=8.
        # The full terrain has many scene lights. A single camera headlight
        # avoids clipping the white robot/floor; this affects rendering only.
        model.light_active[:]=False
        model.vis.headlight.active=1
        model.vis.headlight.diffuse[:]=(.7,.7,.7)
        model.vis.headlight.ambient[:]=(.45,.45,.45)
        model.vis.headlight.specular[:]=(.05,.05,.05)
        model.vis.map.znear=min(model.vis.map.znear,.01/model.stat.extent)
        front_camera=find_camera(model)
        from src.config.mgdp_camera import tilt_quaternion
        from src.config.lgdog1 import FRONT_CAMERA_QUAT
        model.cam_quat[front_camera]=tilt_quaternion(FRONT_CAMERA_QUAT,20.)
        model.cam_fovy[front_camera]=58.
        buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
        mujoco.mj_saveModel(model,filename=None,buffer=buffer)
        (args.output/"render_model.mjb").write_bytes(buffer.tobytes())
        renderer=mujoco.Renderer(model,height=args.height,width=args.width,max_geom=max(20000,model.ngeom+1000))
        inset_renderer=mujoco.Renderer(model,height=180,width=320,max_geom=max(20000,model.ngeom+1000))
        native_data=mujoco.MjData(model);camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
        groups=model.geom_group.copy()
        for idx,item in enumerate(trials):
            item["file"]=f"{idx+1:02d}_{item['kind']}_cmd0p6mps_front.mp4"
            writer=imageio_ffmpeg.write_frames(str(args.output/item["file"]),(args.width,args.height),
                    fps=25,codec="libx264",pix_fmt_out="yuv420p",macro_block_size=2,
                    output_params=["-preset","fast","-crf","20"])
            writer.send(None);writers.append(writer)
        active=np.ones(len(trials),dtype=bool)
        speeds=[[] for _ in trials];smoothed=[deque(maxlen=10) for _ in trials]
        last_frames=[None for _ in trials];last_raw=[None for _ in trials]
        last_inset=[None for _ in trials]
        poses=[[] for _ in trials]
        records=[]
        best_progress=np.zeros(len(trials));last_progress=np.zeros(len(trials))
        dt=env.step_dt
        assert abs(dt-.02)<1e-8
        total_seconds=min(args.max_seconds,2.) if args.smoke else args.max_seconds
        def frame(index,t,velocity,progress,total):
            raw,front=render_world(env,renderer,inset_renderer,native_data,camera,front_camera,index,groups)
            smoothed[index].append(velocity)
            decorated=draw_hud(raw,trials[index],t,float(np.mean(smoothed[index])),progress,total,inset=front)
            writers[index].send(decorated);last_frames[index]=decorated;last_raw[index]=raw
            last_inset[index]=front
            poses[index].append((t,env.sim.data.qpos[index].numpy().copy(),
                                  env.sim.data.qvel[index].numpy().copy(),velocity,progress,total))
            if len(poses[index])==1:
                Image.fromarray(decorated).save(args.output/f"{index+1:02d}_{trials[index]['kind']}_preview.jpg",quality=92)
            return decorated
        with torch.inference_mode():
            totals=command.cumulative[terrain.terrain_types,command.last_waypoint[terrain.terrain_types]].numpy()
            for i in range(len(trials)):
                frame(i,0.,0.,0.,float(totals[i]))
            for step in range(1,int(round(total_seconds/dt))+1):
                action=policy(obs)
                action[torch.from_numpy(~active)]=0.
                obs,_,done,_=wrapped.step(action)
                t=step*dt
                velocity=env.scene["robot"].data.root_link_lin_vel_b[:,:2].norm(dim=1).numpy()
                progress=command.route_state()[3].numpy()
                done=done.numpy().astype(bool)
                newly_finished=[]
                for i in np.flatnonzero(active):
                    speeds[i].append(float(velocity[i]))
                    if float(progress[i]) >= best_progress[i] + .05:
                        best_progress[i]=float(progress[i]);last_progress[i]=t
                    stopped_for_video=bool(t-last_progress[i]>=8. and not done[i])
                    terminal=bool(done[i]) or step==int(round(total_seconds/dt)) or stopped_for_video
                    if step%2==0 or terminal:
                        frame(i,t,float(velocity[i]),float(progress[i]),float(totals[i]))
                    if terminal:
                        reasons=[name for name in env.termination_manager.active_terms
                                 if bool(env.termination_manager.get_term(name)[i])]
                        failed=[name for name in reasons if name not in ("passage_success","time_out")]
                        passed="passage_success" in reasons and not failed
                        reason="成功到达出口" if passed else "、".join(REASONS.get(r,r) for r in failed) or "时限内未到达出口"
                        if stopped_for_video:
                            reason="录像停止：8秒无新增5cm进展，原任务尚未终止"
                        mean=float(np.mean(speeds[i]));later=speeds[i][50:]
                        item={**trials[i],"passed":passed,"reason_zh":reason,"termination_terms":reasons,
                              "sim_seconds":t,"video_stagnation_cut":stopped_for_video,"mean_xy_speed_mps":mean,
                              "mean_xy_speed_after_1s_mps":float(np.mean(later)) if later else mean,
                              "route_progress_m":float(progress[i]),"route_length_m":float(totals[i]),
                              "actual_mean_within_10pct_of_command":bool(.9*args.speed<=mean<=1.1*args.speed)}
                        records.append(item)
                        np.savez_compressed(args.output/f"{i+1:02d}_{item['kind']}_trajectory.npz",
                            time=np.array([p[0] for p in poses[i]]),qpos=np.stack([p[1] for p in poses[i]]),
                            qvel=np.stack([p[2] for p in poses[i]]),speed=np.array([p[3] for p in poses[i]]),
                            progress=np.array([p[4] for p in poses[i]]))
                        final=draw_hud(last_raw[i],item,t,float(np.mean(smoothed[i])),float(progress[i]),float(totals[i]),
                                       (passed,"通过（0.60m/s指令）" if passed else "未完成（0.60m/s指令）",f"{reason}  |  实际全程均速 {mean:.2f} m/s"),inset=last_inset[i])
                        for _ in range(38):writers[i].send(final)
                        writers[i].close();writers[i]=None;active[i]=False
                        newly_finished.append(i)
                        Image.fromarray(final).save(args.output/f"{i+1:02d}_{item['kind']}_result.jpg",quality=92)
                        print("RESULT "+json.dumps(item,ensure_ascii=False),flush=True)
                if newly_finished:
                    original_reset(torch.tensor(newly_finished,dtype=torch.long))
                    env.scene.write_data_to_sim()
                if step%100==0:
                    print(f"PROGRESS simulated={t:.1f}s, active={int(active.sum())}, wall={time.perf_counter()-wall:.1f}s",flush=True)
                if not active.any():break
        records.sort(key=lambda r:[x["kind"] for x in trials].index(r["kind"]))
        write_report(records,args,checkpoint,args.output,time.perf_counter()-wall)
        listing=args.output/"concat.txt"
        listing.write_text("\n".join("file '"+r["file"]+"'" for r in records)+"\n",encoding="utf-8")
        combined=args.output/"model_12400_十类地形_0p6mps_同步前视.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","warning","-y",
                        "-f","concat","-safe","0","-i",str(listing),"-c","copy",str(combined)],check=True)
        print("COMPLETE "+str(combined),flush=True)
    finally:
        for writer in writers:
            if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        if inset_renderer is not None:inset_renderer.close()
        env.close()


if __name__=="__main__":main()
