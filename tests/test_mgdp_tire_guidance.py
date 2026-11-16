"""Grounded foot placement, anti-farming, geometry and training/UI contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,sys,tempfile,unittest,subprocess,shutil
import mujoco,numpy as np,torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.config.mgdp_camera import D435I20_TASK,D435I20_ALIAS
from src.config.tire_guidance import configure_tire_hole_guidance,guidance_contract,validate_tire_guidance_resume
from src.tasks.velocity.mdp.tire_training import TireHoleTouchdown,hole_centres
from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import UnifiedObstacleTerrainCfg
from src.dashboard.commands import build_command
from src.dashboard.catalog import TASKS

ROOT=Path(__file__).resolve().parents[1]


def fixture():
  cfg=load_env_cfg(D435I20_TASK);configure_tire_hole_guidance(cfg)
  g=cfg.scene.terrain.terrain_generator;g.num_cols=10
  class Scene(dict):pass
  n=2;origin=torch.tensor([[100.,200.,0.],[300.,400.,0.]])
  data=NS(found_history=torch.zeros(n,8,4),force_history=torch.zeros(n,8,4,3),
          pos_history=torch.zeros(n,8,4,3),normal_history=torch.zeros(n,8,4,3))
  # Deliberately scramble primaries to ensure no reliance on XML ordering.
  names=['RL','FR','RR','FL']
  sensor=NS(data=data,cfg=NS(num_slots=2),_slots=[NS(primary_name=f'{name}_calf1_collision',field_name='found') for name in names])
  scene=Scene(stone_support_contact=sensor)
  scene.terrain=NS(cfg=NS(terrain_generator=g),env_origins=origin,terrain_types=torch.tensor([6,0]))
  model=mujoco.MjModel.from_xml_path(str(ROOT/'src/assets/robots/lgdog1/xmls/lgdog1.xml'))
  # Runtime model has the robot namespace; this fixture wraps just the lookup.
  simmodel=NS(geom=lambda name:model.geom(name.removeprefix('robot/')),geom_size=model.geom_size,geom_type=model.geom_type)
  command=NS(cfg=NS(goal_x=5.3,lane_half_width=.28),local_xy=lambda:torch.tensor([[2.,0.],[2.,0.]]))
  fail=torch.zeros(n,dtype=torch.bool)
  env=NS(cfg=cfg,num_envs=n,device='cpu',step_dt=.02,scene=scene,sim=NS(mj_model=simmodel),
    command_manager=NS(get_term=lambda _:command),termination_manager=NS(get_term=lambda _:fail))
  termcfg=cfg.rewards['tire_hole_touchdown'];term=TireHoleTouchdown(termcfg,env)
  def land(foot=0,col=0,dx=0.,dy=0.,z=0.,normal=1.,force=2.,history=0):
    name=('FR','FL','RR','RL')[foot];slot=names.index(name)*2
    data.found_history[:,slot,history]=1
    data.force_history[:,slot,history,2]=force
    xy=hole_centres()[foot,col]+torch.tensor([dx,dy])
    data.pos_history[:,slot,history,:2]=origin[:,:2]+xy
    data.pos_history[:,slot,history,2]=z
    data.normal_history[:,slot,history,2]=normal
  def clear():
    for v in vars(data).values():v.zero_()
  def call():return term(env,**termcfg.params)*env.step_dt*termcfg.weight
  return env,term,land,clear,call,fail


class TireGuidanceTests(unittest.TestCase):
  def test_new_hole_pays_once_stance_and_retouch_do_not_pay(self):
    env,term,land,clear,call,_=fixture();call();land()
    torch.testing.assert_close(call(),torch.tensor([.25,0.]))
    self.assertEqual(call().sum(),0.)
    clear();call();land();self.assertEqual(call().sum(),0.)
    clear();call();land(col=1)
    self.assertEqual(call()[0],.25)
    self.assertEqual(term.paid[0].sum(),2)

  def test_each_leg_has_its_own_holes_and_partial_reset_is_clean(self):
    env,term,land,clear,call,_=fixture();call()
    for f in range(4):land(foot=f)
    self.assertEqual(call()[0],1.)
    term.reset(torch.tensor([0]));self.assertEqual(term.paid[0].sum(),0)
    self.assertEqual(call().sum(),0.) # Seed new episode, no free spawn reward.
    clear();call();land();self.assertEqual(call()[0],.25)

  def test_tire_top_side_wrong_row_air_pad_and_unloaded_contacts_are_rejected(self):
    for change in ({'z':.025},{'normal':.1},{'dy':.55},{'dx':-.5},{'force':.1},{'dx':.15}):
      with self.subTest(change=change):
        _,_,land,_,call,_=fixture();call();land(**change);self.assertEqual(call().sum(),0.)

  def test_early_physics_substep_landing_is_kept_and_failures_are_excluded(self):
    env,term,land,clear,call,fail=fixture();call();land(history=3)
    self.assertEqual(call()[0],.25)
    self.assertFalse(term.previous[0,0]) # Contact ended before the final substep.
    clear();land(col=1);fail[0]=True;self.assertEqual(call().sum(),0.)

  def test_radius_subtracts_real_foot_size_and_margin(self):
    _,term,land,_,call,_=fixture()
    torch.testing.assert_close(term.safe_radius,torch.full((4,),.164647-.012-.01))
    call();land(dx=.14);self.assertEqual(call()[0],.25)

  def test_sliding_supported_foot_into_next_hole_is_not_a_new_touchdown(self):
    env,term,land,clear,call,_=fixture();call()
    for h in range(4):land(history=h)
    self.assertEqual(call()[0],.25)
    clear()
    for h in range(4):land(col=1,history=h)
    self.assertEqual(call().sum(),0.)
    clear();call();land(col=1);self.assertEqual(call()[0],.25)

  def test_guidance_is_opt_in_and_preserves_other_rewards_camera_and_geometry(self):
    cfg=load_env_cfg(D435I20_TASK);old=copy.deepcopy(cfg)
    self.assertIsNone(guidance_contract(cfg))
    configure_tire_hole_guidance(cfg);contract=guidance_contract(cfg)
    self.assertEqual(contract['weight'],.25)
    self.assertEqual(cfg.scene.terrain,old.scene.terrain)
    self.assertEqual(cfg.observations,old.observations)
    self.assertEqual({k:v for k,v in cfg.rewards.items() if k!='tire_hole_touchdown'},old.rewards)
    self.assertNotEqual(cfg.commands['twist'].wall_reward_contract_version,old.commands['twist'].wall_reward_contract_version)
    stamp=cfg.commands['twist'].wall_reward_contract_version;configure_tire_hole_guidance(cfg)
    self.assertEqual(stamp,cfg.commands['twist'].wall_reward_contract_version)
    for task in ('LgDog1-Rough-MGDP','LgDog1-Race-MGDP'):
      with self.subTest(task=task),self.assertRaises(ValueError):configure_tire_hole_guidance(load_env_cfg(task))

  def test_resume_requires_same_guidance_objective(self):
    cfg=load_env_cfg(D435I20_TASK);old=copy.deepcopy(cfg)
    validate_tire_guidance_resume(cfg,{'infos':{}})
    configure_tire_hole_guidance(cfg)
    saved={'infos':{'mgdp_tire_hole_guidance':guidance_contract(cfg)}}
    validate_tire_guidance_resume(cfg,saved)
    for current,checkpoint in ((cfg,{'infos':{}}),(old,saved)):
      with self.assertRaisesRegex(ValueError,'warm-start-checkpoint'):validate_tire_guidance_resume(current,checkpoint)
    from src.config.tire_guidance import restore_tire_guidance_for_play
    play=load_env_cfg(D435I20_TASK,play=True);restore_tire_guidance_for_play(play,saved)
    self.assertEqual(guidance_contract(play),guidance_contract(cfg))
    changed=copy.deepcopy(saved);changed['infos']['mgdp_tire_hole_guidance']['version']=999
    with self.assertRaises(ValueError):restore_tire_guidance_for_play(load_env_cfg(D435I20_TASK,play=True),changed)

  def test_workbench_and_cli_explicitly_enable_guidance(self):
    import mjlab,tyro
    from scripts.train import TrainConfig
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);encoder=root/TASKS[D435I20_ALIAS].pretrain_checkpoint
      encoder.parent.mkdir(parents=True);encoder.touch()
      options={'mode':'train','task':D435I20_ALIAS,'tire_hole_guidance':True}
      argv=build_command(root,sys.executable,options)
      self.assertIn('--tire-hole-guidance=True',argv)
      cfg=tyro.cli(TrainConfig,default=TrainConfig.from_task(D435I20_TASK),args=argv[3:],config=mjlab.TYRO_FLAGS)
      self.assertTrue(cfg.tire_hole_guidance)
      self.assertFalse(TrainConfig.from_task(D435I20_TASK).tire_hole_guidance)
      for bad in ('True',1,None):
        with self.subTest(value=bad),self.assertRaises(ValueError):build_command(root,sys.executable,{**options,'tire_hole_guidance':bad})
      with self.assertRaises(ValueError):build_command(root,sys.executable,{**options,'task':'standard'})

  @unittest.skipUnless(shutil.which('node'),'node is required for frontend verification')
  def test_checkbox_submits_only_for_supported_task(self):
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
function node(){return {value:'',disabled:false,checked:false,textContent:'',classList:{toggle(){},add(){},remove(){}},querySelectorAll:()=>[],append(){},replaceChildren(){}};}
const nodes=new Proxy({}, {get(t,k){return t[k]??=node();}});
const ctx={document:{getElementById:id=>nodes[id],createElement:()=>node()},window:{LgDogFormat:{}}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace(/\nstart\(\);\s*$/,''),ctx);
nodes.task.value='wall';nodes['terrain-curriculum'].value='manual';
vm.runInContext("state={tasks:[{alias:'wall',scene:'race_depth',directed_passage:true,supports_tire_hole_guidance:true},{alias:'standard',scene:'rough'}]};updateCurriculumOptions();globalThis.payload=collect();",ctx);
assert.equal(ctx.payload.tire_hole_guidance,false);assert.equal(nodes['tire-hole-guidance'].disabled,false);
nodes['tire-hole-guidance'].checked=true;vm.runInContext('globalThis.payload=collect();',ctx);
assert.equal(ctx.payload.tire_hole_guidance,true);
nodes.task.value='standard';vm.runInContext('updateCurriculumOptions();globalThis.payload=collect();',ctx);
assert.equal(ctx.payload.tire_hole_guidance,false);assert.equal(nodes['tire-hole-guidance'].disabled,true);
'''
    subprocess.run(['node','-e',script,str(ROOT/'src/dashboard/web/app.js')],check=True,capture_output=True,text=True)

  def test_real_mesh_keeps_floor_under_safe_landing_disks_at_all_stages(self):
    for row in range(8):
      spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
      UnifiedObstacleTerrainCfg(kind='tire_array',size=(12.,8.)).function((row+.1)/8,spec,np.random.default_rng(42))
      m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
      for centre in hole_centres()[0]:
        for angle in np.linspace(0,2*np.pi,12,endpoint=False):
          point=np.array([centre[0]+1+.1546*np.cos(angle),centre[1]+4+.1546*np.sin(angle),1.])
          ray=mujoco.mj_ray(m,d,point,np.array([0.,0.,-1.]),None,1,-1,None)
          self.assertAlmostEqual(ray,1.,places=5)


if __name__=='__main__':unittest.main()
