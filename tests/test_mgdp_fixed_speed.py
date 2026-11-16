"""A requested 1m/s cruise must reach the CLI and survive curriculum/resume."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,math,shutil,subprocess,sys,tempfile,unittest
import torch
from mjlab.terrains import TerrainGeneratorCfg
from src.config.passage_speed import (configure_passage_fixed_speed,validate_fixed_speed,
    validate_passage_speed_resume,restore_fixed_speed_for_play)
from src.config.mgdp_camera import D435I20_ALIAS,D435I20_TASK
from src.dashboard.catalog import TASKS
from src.dashboard.commands import build_command
from src.tasks.velocity.mdp import passage
from src.tasks.velocity.config.lgdog1.passage_terrain_cfg import PassageTerrainCfg

ROOT=Path(__file__).resolve().parents[1]


def fake_env(fixed=True):
  xy=torch.zeros(2,2);failure=torch.zeros(2,dtype=torch.bool)
  cmd=NS(local_xy=lambda:xy,cfg=NS(goal_x=5.3,lane_half_width=.28,
      ranges=NS(lin_vel_x=(1.,1.) if fixed else (.25,.35))))
  terrain=NS(terrain_types=torch.arange(2),terrain_levels=torch.zeros(2,dtype=torch.long),
    max_terrain_level=3,terrain_origins=torch.zeros(3,2,3),env_origins=torch.zeros(2,3),
    cfg=NS(terrain_generator=TerrainGeneratorCfg(size=(8.,8.),num_cols=2,num_rows=3,
      sub_terrains={'flat':PassageTerrainCfg(kind='start_flat',proportion=.5),
                    'wall':PassageTerrainCfg(kind='high_wall',proportion=.5)})))
  def update(ids,up,down):
    terrain.terrain_levels[ids]=(terrain.terrain_levels[ids]+up.long()-down.long()).clamp(0,2)
  terrain.update_env_origins=update
  env=NS(num_envs=2,device='cpu',step_dt=.02,scene=NS(terrain=terrain),
    command_manager=NS(get_term=lambda name:cmd),episode_length_buf=torch.full((2,),100),
    reset_buf=torch.ones(2,dtype=torch.bool),cfg=NS(terminations={'impact':NS(time_out=False)}),
    termination_manager=NS(get_term=lambda name:failure),metrics_manager=NS(
      _step_count=torch.full((2,),100),_episode_sums={'velocity_error_xy_mps':torch.ones(2)*10}))
  params={'min_episodes':1,'speed_limit':1.}
  if fixed:params['fixed_speed']=1.
  return env,xy,failure,params


class FixedSpeedTests(unittest.TestCase):
  def test_workbench_preview_parses_as_explicit_one_meter_per_second(self):
    import mjlab,tyro
    from scripts.train import TrainConfig
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);encoder=root/TASKS[D435I20_ALIAS].pretrain_checkpoint
      encoder.parent.mkdir(parents=True);encoder.touch()
      argv=build_command(root,sys.executable,{'mode':'train','task':D435I20_ALIAS,
        'passage_fixed_speed':'1.0','envs':1024,'terrain_num_rows':8,'terrain_num_cols':10})
      self.assertIn('--passage-fixed-speed=1.0',argv)
      self.assertFalse(any(x.startswith('--velocity-') for x in argv))
      cfg=tyro.cli(TrainConfig,default=TrainConfig.from_task(D435I20_TASK),args=argv[3:],config=mjlab.TYRO_FLAGS)
      self.assertEqual(cfg.passage_fixed_speed,1.)
      self.assertEqual(cfg.env.scene.num_envs,1024)
      self.assertEqual((cfg.terrain_num_rows,cfg.terrain_num_cols),(8,10))
      for invalid in ({'passage_speed_mode':'fixed','passage_fixed_speed':''},
                      {'passage_speed_mode':'curriculum','passage_fixed_speed':1.},
                      {'passage_speed_mode':'unknown'}):
        with self.subTest(invalid=invalid),self.assertRaises(ValueError):
          build_command(root,sys.executable,{'mode':'train','task':D435I20_ALIAS,**invalid})

  def test_original_course_remains_available_without_fixed_flag(self):
    from scripts.train import TrainConfig
    self.assertIsNone(TrainConfig.from_task(D435I20_TASK).passage_fixed_speed)
    self.assertEqual(TrainConfig.from_task(D435I20_TASK).env.commands['twist'].ranges.lin_vel_x,(.6,.6))
    self.assertIsNone(TASKS[D435I20_ALIAS].passage_fixed_speed_default)
    self.assertIsNone(TASKS['race_mgdp'].passage_fixed_speed_default)

  def test_fixed_speed_keeps_camera_rewards_and_terrain_curriculum(self):
    from scripts.train import TrainConfig
    from src.mgdp_depth import camera_contract
    cfg=TrainConfig.from_task(D435I20_TASK).env
    camera=camera_contract(cfg);weights={n:r.weight for n,r in cfg.rewards.items()}
    configure_passage_fixed_speed(cfg,1.)
    self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x,(1.,1.))
    self.assertEqual(cfg.curriculum['passage'].params['fixed_speed'],1.)
    self.assertEqual(camera_contract(cfg),camera)
    self.assertEqual({n:r.weight for n,r in cfg.rewards.items()},weights)

  def test_invalid_or_nonpassage_speed_is_rejected(self):
    from scripts.train import TrainConfig
    for value in (0.,-.1,float('nan'),float('inf'),3.01,True):
      with self.subTest(value=value),self.assertRaises((ValueError,TypeError)):validate_fixed_speed(value)
    with self.assertRaisesRegex(ValueError,'定向越障'):
      configure_passage_fixed_speed(TrainConfig.from_task('LgDog1-Rough-MGDP').env,1.)
    with self.assertRaisesRegex(ValueError,'定向越障'):
      build_command(ROOT,sys.executable,{'mode':'train','task':'standard','passage_fixed_speed':1.})

  def test_success_and_failure_never_change_fixed_cruise_but_do_change_levels(self):
    env,xy,failure,params=fake_env();term=passage.PassageCurriculum(NS(params=params),env)
    xy[:,0]=5.4
    for _ in range(6):
      logs=term(env,torch.arange(2),**params)
      self.assertEqual(term.command.cfg.ranges.lin_vel_x,(1.,1.))
      self.assertEqual(float(logs['speed_min']),1.);self.assertEqual(float(logs['speed_max']),1.)
      self.assertEqual(float(logs['speed_fixed']),1.)
    self.assertTrue((env.scene.terrain.terrain_levels==2).all())
    failure[:]=True
    term(env,torch.arange(2),**params)
    self.assertTrue((env.scene.terrain.terrain_levels==1).all())
    self.assertEqual(term.command.cfg.ranges.lin_vel_x,(1.,1.))

  def test_fixed_curriculum_resume_preserves_speed_and_rejects_tampering(self):
    env,xy,_,params=fake_env();term=passage.PassageCurriculum(NS(params=params),env)
    xy[:,0]=5.4;term(env,torch.arange(2),**params)
    state=term.state_dict()
    fresh,_,_,_=fake_env();restored=passage.PassageCurriculum(NS(params=params),fresh)
    restored.load_state_dict(state)
    self.assertEqual(restored.command.cfg.ranges.lin_vel_x,(1.,1.))
    torch.testing.assert_close(fresh.scene.terrain.terrain_levels,env.scene.terrain.terrain_levels)
    with self.assertRaisesRegex(ValueError,'Invalid passage speed'):
      restored.load_state_dict({**state,'ranges':(.25,.35)})
    adaptive,_,_,adaptive_params=fake_env(False)
    with self.assertRaisesRegex(ValueError,'Incompatible passage'):
      passage.PassageCurriculum(NS(params=adaptive_params),adaptive).load_state_dict(state)

  def test_resume_mode_guard_and_play_restore_use_checkpoint(self):
    from scripts.train import TrainConfig
    from mjlab.tasks.registry import load_env_cfg
    saved={'infos':{'mgdp_passage_curriculum':{'params':{'fixed_speed':1.}}}}
    cfg=TrainConfig.from_task(D435I20_TASK).env
    saved['infos']['mgdp_passage_curriculum']['params'].update(
      {key:cfg.curriculum['passage'].params[key]
       for key in ('terrain_max_error','level_up_streak','level_down_streak')})
    with self.assertRaisesRegex(ValueError,'速度模式'):validate_passage_speed_resume(cfg,saved)
    configure_passage_fixed_speed(cfg,1.);validate_passage_speed_resume(cfg,saved)
    with self.assertRaisesRegex(ValueError,'速度模式'):validate_passage_speed_resume(cfg,{'infos':{}})
    play=load_env_cfg(D435I20_TASK,play=True)
    self.assertEqual(restore_fixed_speed_for_play(play,saved),1.)
    self.assertEqual(play.commands['twist'].ranges.lin_vel_x,(1.,1.))
    self.assertNotIn('passage',play.curriculum)

  @unittest.skipUnless(shutil.which('node'),'node is required for frontend wiring')
  def test_frontend_submits_fixed_speed_and_reports_actual_run_metrics(self):
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
function node(){return {value:'',disabled:false,checked:false,textContent:'',
 classList:{toggle(){},add(){},remove(){}},querySelectorAll:()=>[],append(){},replaceChildren(){}};}
const nodes=new Proxy({}, {get(t,k){return t[k]??=(node());}});
const ctx={document:{getElementById:id=>nodes[id],createElement:()=>node()},
 window:{LgDogFormat:{metricValue:(d,k)=>d?.metrics?.[k]?.value??null}}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace(/\nstart\(\);\s*$/,''),ctx);
nodes.task.value='new';nodes['terrain-curriculum'].value='manual';
nodes['velocity-min-abs'].value=nodes['velocity-max-abs'].value='1.0';
vm.runInContext("state={tasks:[{alias:'new',scene:'race_depth',directed_passage:true,passage_fixed_speed_default:1},{alias:'standard',scene:'rough'}]};updateCurriculumOptions();globalThis.payload=collect();",ctx);
assert.equal(nodes['passage-speed-mode'].value,'fixed');assert.equal(nodes['passage-fixed-speed'].disabled,false);
assert.equal(ctx.payload.passage_fixed_speed,'1');assert.equal(ctx.payload.velocity_min_abs,'');
assert.equal(ctx.payload.passage_speed_mode,'fixed');
vm.runInContext("renderPassage({metrics:{passage_speed_max:{value:.35},passage_speed_fixed:{value:0}}});",ctx);
assert.match(nodes['passage-speed'].textContent,/0.250–0.350/);
vm.runInContext("renderPassage({metrics:{passage_speed_max:{value:1},passage_speed_min:{value:1},passage_speed_fixed:{value:1}}});",ctx);
assert.match(nodes['passage-speed'].textContent,/固定巡航目标 1.000/);
nodes['passage-speed-mode'].value='curriculum';vm.runInContext('updateCurriculumOptions();globalThis.payload=collect();',ctx);
assert.equal(ctx.payload.passage_fixed_speed,'');assert.equal(nodes['passage-fixed-speed'].disabled,true);
nodes.task.value='standard';vm.runInContext('updateCurriculumOptions();globalThis.payload=collect();',ctx);
assert.equal(nodes['velocity-min-abs'].disabled,false);assert.equal(ctx.payload.velocity_min_abs,'1.0');
'''
    subprocess.run(['node','-e',script,str(ROOT/'src/dashboard/web/app.js')],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
