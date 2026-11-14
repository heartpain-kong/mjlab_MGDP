from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json,shutil,subprocess,sys,tempfile,unittest

from src.wall_reward_settings import (
  SOURCE_RELATIVE,WALL_REWARD_DEFAULTS,load_wall_reward_settings,read_wall_reward_settings,
  save_wall_reward_settings,validate_wall_reward_settings,wall_reward_contract,
)
from src.dashboard import routes,history
ROOT=Path(__file__).resolve().parents[1]


class WallRewardSettingsTests(unittest.TestCase):
  def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
    self.root=Path(self.tmp.name);self.file=self.root/SOURCE_RELATIVE
    self.file.parent.mkdir(parents=True);shutil.copy2(ROOT/SOURCE_RELATIVE,self.file)

  def test_save_read_backup_and_restore_defaults(self):
    before=self.file.read_bytes();data=read_wall_reward_settings(self.root)
    self.assertEqual(data['settings'],WALL_REWARD_DEFAULTS);self.assertFalse(data['changed'])
    changed=deepcopy(data['settings']);changed['wall_waiting']['grace_s']=1.2
    changed['wall_crossing']['weight']=3.;changed['no_passage_progress']['timeout_s']=5.
    saved=save_wall_reward_settings(self.root,changed)
    self.assertEqual(load_wall_reward_settings(self.file),changed)
    self.assertEqual(Path(saved['backup']).read_bytes(),before);self.assertTrue(saved['contract_changed'])
    save_wall_reward_settings(self.root,WALL_REWARD_DEFAULTS)
    self.assertFalse(read_wall_reward_settings(self.root)['changed'])
    self.assertIn('WALL_REWARD_OVERRIDES={}',self.file.read_text(encoding='utf-8'))

  def test_invalid_values_do_not_touch_source(self):
    before=self.file.read_bytes()
    for group,key,value in [('wall_waiting','weight',1.),('wall_waiting','grace_s',float('nan')),
      ('wall_waiting','ramp_s',0.),('progress_gate','enabled',1),('passage_success','weight',-1.),
      ('no_passage_progress','timeout_s','4'),('wall_crossing','weight',float('inf'))]:
      changed=deepcopy(WALL_REWARD_DEFAULTS);changed[group][key]=value
      with self.subTest(group=group,key=key),self.assertRaises(ValueError):save_wall_reward_settings(self.root,changed)
      self.assertEqual(self.file.read_bytes(),before)
    self.assertFalse(self.file.with_suffix('.py.bak').exists())

  def test_unknown_or_incomplete_payload_is_rejected(self):
    for mutate in (lambda s:s.pop('wall_waiting'),lambda s:s['wall_waiting'].update(bogus=1)):
      settings=deepcopy(WALL_REWARD_DEFAULTS);mutate(settings)
      with self.assertRaises(ValueError):validate_wall_reward_settings(settings)

  def test_routes_are_task_scoped_and_write_history(self):
    ctx=NS(root=self.root)
    self.assertFalse(routes.wall_rewards(ctx,{'task':['race_mgdp']})[1]['available'])
    with self.assertRaises(ValueError):routes.wall_reward_save(ctx,{'task':'race_mgdp','settings':WALL_REWARD_DEFAULTS})
    settings=deepcopy(WALL_REWARD_DEFAULTS);settings['passage_success']['weight']=8.
    status,data=routes.wall_reward_save(ctx,{'task':'race_mgdp_wall','settings':settings})
    self.assertEqual(status,200);self.assertEqual(data['settings']['passage_success']['weight'],8.)
    edits=history.read_history(self.root)['reward_edits']
    self.assertEqual(edits[-1]['profile'],'mgdp_wall_rules')
    self.assertEqual(edits[-1]['changes'][0]['name'],'passage_success')

  def test_editor_import_is_standard_library_only(self):
    result=subprocess.run([sys.executable,'-c',
      "import sys;import src.wall_reward_settings;assert 'torch' not in sys.modules;assert 'mujoco' not in sys.modules"],
      cwd=ROOT,capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)

  def test_default_contract_is_v5_and_modified_settings_change_it(self):
    self.assertEqual(wall_reward_contract(WALL_REWARD_DEFAULTS),1)
    changed=deepcopy(WALL_REWARD_DEFAULTS);changed['wall_waiting']['max_rate']=3.
    self.assertNotEqual(wall_reward_contract(changed),1)
    self.assertEqual(wall_reward_contract(changed),wall_reward_contract(json.loads(json.dumps(changed))))


class WallSettingsTrainingIntegrationTests(unittest.TestCase):
  def test_saved_settings_reach_real_factory_and_leave_legacy_unchanged(self):
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg,lgdog1_race_mgdp_env_cfg
    from src.tasks.velocity.mdp.wall_training import WallProgressReward
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);file=root/SOURCE_RELATIVE;file.parent.mkdir(parents=True);shutil.copy2(ROOT/SOURCE_RELATIVE,file)
      settings=deepcopy(WALL_REWARD_DEFAULTS)
      settings['wall_waiting'].update(grace_s=1.1,ramp_s=2.2,max_rate=3.,weight=-.5)
      settings['wall_crossing'].update(weight=3.,root_clearance_m=.4)
      settings['passage_success']['weight']=8.;settings['passage_progress']['weight']=1.5
      settings['no_passage_progress']['timeout_s']=6.
      settings['progress_gate']['progress_reference_mps']=.2
      save_wall_reward_settings(root,settings)
      with patch('src.wall_reward_settings.load_wall_reward_settings',side_effect=lambda:load_wall_reward_settings(file)):
        # The registry keeps an import-time snapshot. A new training process
        # constructs this factory from the saved settings at registration.
        cfg=lgdog1_wall_training_env_cfg();old=lgdog1_race_mgdp_env_cfg()
      self.assertEqual(cfg.rewards['wall_waiting'].params['grace_s'],1.1)
      self.assertEqual(cfg.rewards['wall_waiting'].weight,-.5)
      self.assertEqual(cfg.rewards['wall_waiting'].params['root_clearance_m'],.4)
      self.assertEqual(cfg.rewards['wall_crossing'].weight,3.)
      self.assertEqual(cfg.rewards['passage_success'].weight,8.)
      self.assertEqual(cfg.rewards['passage_progress'].weight,1.5)
      self.assertEqual(cfg.rewards['track_linear_velocity'].params['progress_reference_mps'],.2)
      self.assertIs(cfg.rewards['pose'].func,WallProgressReward)
      self.assertEqual(cfg.terminations['no_passage_progress'].params['timeout_s'],6.)
      self.assertNotEqual(cfg.commands['twist'].wall_reward_contract_version,1)
      self.assertEqual(old.rewards['passage_success'].weight,5.);self.assertNotIn('wall_waiting',old.rewards)

  def test_switches_disable_only_the_selected_rules(self):
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg
    from src.tasks.velocity.mdp.wall_training import WallProgressReward
    settings=deepcopy(WALL_REWARD_DEFAULTS)
    for name in ('progress_gate','wall_waiting','no_passage_progress'):settings[name]['enabled']=False
    with patch('src.wall_reward_settings.load_wall_reward_settings',return_value=settings):
      cfg=lgdog1_wall_training_env_cfg()
    self.assertIsNot(cfg.rewards['pose'].func,WallProgressReward)
    self.assertEqual(cfg.rewards['wall_waiting'].weight,0.)
    self.assertNotIn('no_passage_progress',cfg.terminations)
    self.assertEqual(cfg.rewards['wall_crossing'].weight,2.)


if __name__=='__main__':unittest.main()
