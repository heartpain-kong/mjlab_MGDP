"""Zero-action physical probe for wall reward farming, not policy acceptance.

The old score is a counterfactual evaluation of the original shaping functions
on the SAME physical states. This does not compare two trained policies.
"""
import argparse,csv,json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from src.sim_memory import compact_model_allocations
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg
from src.tasks.velocity.mdp.wall_training import WallProgressReward


def run(args):
  cfg=lgdog1_wall_training_env_cfg(with_perception=False,checkpoint='')
  cfg.scene.num_envs=args.envs;cfg.seed=args.seed;cfg.curriculum={}
  generator=cfg.scene.terrain.terrain_generator;generator.num_cols=10
  for name,terrain in generator.sub_terrains.items():terrain.proportion=float(name=='high_wall')
  cfg.commands['twist'].ranges.lin_vel_x=(.3,.3)
  cfg.events['reset_base'].params['pose_range'].update(x=(1.,1.),y=(0.,0.),yaw=(0.,0.))
  output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
  env=None
  try:
    with compact_model_allocations(True,'auto',True):env=ManagerBasedRlEnv(cfg,device=args.device)
    env.reset();manager=env.reward_manager;names=manager.active_terms
    shaping=[name for name in names if isinstance(manager.get_term_cfg(name).func,WallProgressReward)
             and manager.get_term_cfg(name).weight>0]
    first=manager.get_term_cfg(shaping[0]).func
    actions=torch.zeros((args.envs,env.action_space.shape[-1]),device=args.device)
    rows=[];idle_old=[];idle_new=[];finite=True
    with torch.inference_mode():
      for step in range(args.steps):
        record=first.highwater.clone();ready=first.ready.clone()
        _,_,terminated,truncated,_=env.step(actions)
        done=terminated|truncated
        data=env.scene['robot'].data
        local_x=data.root_link_pos_w[:,0]-env.scene.env_origins[:,0]
        speed=data.root_link_lin_vel_b[:,:2].norm(dim=1)
        old=torch.zeros(args.envs,device=args.device);new=torch.zeros_like(old)
        for name in shaping:
          term=manager.get_term_cfg(name)
          params={k:v for k,v in term.params.items() if k not in
                  ('base_func','progress_reference_mps','min_command_mps')}
          old+=term.func.base(env,**params)*term.weight
          new+=manager._step_reward[:,names.index(name)]
        waiting=manager._step_reward[:,names.index('wall_waiting')]
        idle=(~done)&ready&(speed<.03)&(local_x>=.55)&(local_x<1.69)&((first.highwater-record)<=1e-7)
        if idle.any():
          idle_old.extend(old[idle].cpu().tolist());idle_new.extend(new[idle].cpu().tolist())
        finite=finite and bool(torch.isfinite(data.root_link_pos_w).all())
        rows.append({'seconds':(step+1)*env.step_dt,'old_counterfactual_shaping_rate':old.mean().item(),
          'new_shaping_rate':new.mean().item(),'waiting_reward_rate':waiting.mean().item(),
          'mean_local_root_x_m':local_x.mean().item(),'mean_speed_mps':speed.mean().item(),
          'stationary_no_new_progress_samples':int(idle.sum()),'done_envs':int(done.sum())})
    with output.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'passed':bool(idle_new) and max(idle_new)<=1e-6 and sum(idle_old)>0 and finite,
      'device':args.device,'envs':args.envs,'steps':args.steps,'step_dt':env.step_dt,
      'fixture':'zero action, wall-only terrain, stage 1, spawn 1m along route',
      'stationary_no_new_progress_samples':len(idle_new),
      'old_counterfactual_idle_shaping_rate_mean':sum(idle_old)/max(len(idle_old),1),
      'new_idle_shaping_rate_max':max(idle_new,default=None),'finite_physics':finite,
      'counterfactual_on_same_states':True,'trained_policy_success_validated':False,
      'csv':str(output.with_suffix('.csv'))}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('MGDP_WALL_REWARD_CHECK '+json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Wall reward physical probe did not satisfy its checks')
    return result
  finally:
    if env is not None:env.close()


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--device',default='cpu');p.add_argument('--envs',type=int,default=8)
  p.add_argument('--steps',type=int,default=250);p.add_argument('--seed',type=int,default=42)
  p.add_argument('--output',default='.artifacts/mgdp-wall-rewards/check.json')
  args=p.parse_args()
  if args.envs<1 or args.steps<1:p.error('--envs and --steps must be positive')
  if args.device=='cpu':torch.set_num_threads(4)
  run(args)


if __name__=='__main__':main()
