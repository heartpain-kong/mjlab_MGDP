import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from copy import deepcopy
from unittest.mock import patch
import torch
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.config.mgdp_camera import D435I20_TASK
from src.config.mgdp_skill_training import configure_skill_training,restore_skill_for_play,validate_skill_resume
from src.config.mgdp_effective_rules import resolve_wall_rules,effective_rule_snapshot
from src.config.mgdp_stages import configure_full_course_eval,pin_full_course_level
from src.wall_reward_settings import WALL_REWARD_DEFAULTS
from src.tasks.velocity.mdp.course_v3 import BodyAwarePassageCommand
from src.tasks.velocity.mdp.passage_diagnostics import PassageDiagnostics
from src.dashboard.server import read_passage_diagnostics,read_effective_wall_rules


class SkillV5Tests(unittest.TestCase):
  def fixture(self):
    from tests.test_mgdp_course_v3 import CourseV3Tests
    env,_,robot=CourseV3Tests().command_env();configure_skill_training(env.cfg,5)
    robot.find_sites=lambda *a,**k:([0,1,2,3],['FR','FL','RR','RL'])
    robot.data.site_pos_w=torch.zeros(10,4,3)
    cmd=BodyAwarePassageCommand(env.cfg.commands['twist'],env)
    env.command_manager=NS(get_term=lambda _:cmd);cmd._resample_command(torch.arange(10));cmd.speed.fill_(.6)
    env.scene.terrain.max_terrain_level=8
    return env,cmd,robot

  def test_effective_preview_matches_bound_rules_and_is_not_a_mutation(self):
    saved=deepcopy(WALL_REWARD_DEFAULTS);before=deepcopy(saved)
    preview=resolve_wall_rules(saved,5);self.assertEqual(saved,before)
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c,5)
    actual=effective_rule_snapshot(c)
    self.assertEqual(actual['settings'],preview['settings'])
    self.assertEqual(actual['max_waiting_cost_per_second'],.5)
    self.assertFalse(actual['wall_velocity_reward'])
    self.assertGreater(len(preview['changes']),0)
    disabled=deepcopy(saved);disabled['wall_waiting']['enabled']=False
    self.assertEqual(resolve_wall_rules(disabled,5)['max_waiting_cost_per_second'],0)
    self.assertEqual(resolve_wall_rules(saved,None)['settings'],saved)

  def test_mixed_rows_use_own_routes_and_wall_requires_all_feet(self):
    env,cmd,robot=self.fixture();g=env.cfg.scene.terrain.terrain_generator
    for row in range(8):
      env.scene.terrain.terrain_levels.fill_(row);cmd._resample_command(torch.arange(10))
      for kind in ('high_wall','gap_steps','slalom_brick'):
        i=list(g.sub_terrains).index(kind);bank=cmd.cfg.staged_waypoints[kind]
        expected=bank[row];key=cmd.route_keys()[i]
        self.assertEqual(cmd.last_waypoint[key].item(),len(expected)-1)
        torch.testing.assert_close(cmd.routes[key,len(expected)-1],torch.tensor(expected[-1]))
        for target in expected[1:]:
          robot.data.root_link_pos_w[i,:2]=torch.tensor(target)
          robot.data.site_pos_w[i,:,0]=robot.data.root_link_pos_w[i,0]-.35
          cmd._update_command()
        self.assertTrue(cmd.route_success()[i],(row,kind))
        if kind=='high_wall':
          robot.data.site_pos_w[i,3,0]=0.;self.assertFalse(cmd.route_success()[i])
      if row>=5:
        for kind in ('high_wall','gap_steps','slalom_brick'):
          self.assertEqual(cmd.cfg.staged_waypoints[kind][row],cmd.cfg.waypoints[kind])
    env.scene.terrain.terrain_levels[4]=0;env.scene.terrain.terrain_levels[9]=5
    self.assertEqual(cmd.last_waypoint[cmd.route_keys()[4]],1)
    self.assertEqual(cmd.last_waypoint[cmd.route_keys()[9]],18)

  def test_short_goals_do_not_complete_full_routes(self):
    env,cmd,robot=self.fixture()
    for i in (3,4,9):
      route=cmd.cfg.staged_waypoints[list(env.cfg.scene.terrain.terrain_generator.sub_terrains)[i]][0]
      for target in route[1:]:
        robot.data.root_link_pos_w[i,:2]=torch.tensor(target);robot.data.site_pos_w[i,:,0]=robot.data.root_link_pos_w[i,0]-.35
        cmd._update_command()
      self.assertTrue(cmd.route_success()[i])
    configure_full_course_eval(env.cfg)
    full=BodyAwarePassageCommand(env.cfg.commands['twist'],env);env.command_manager=NS(get_term=lambda _:full)
    full._resample_command(torch.arange(10))
    self.assertFalse(full.route_success()[[3,4,9]].any())
    wall=env.cfg.scene.terrain.terrain_generator.sub_terrains['high_wall']
    self.assertEqual(wall.counts,(3,)*8);self.assertFalse(hasattr(wall,'wall_centres'))
    stones=env.cfg.scene.terrain.terrain_generator.sub_terrains['stepping_stones']
    self.assertEqual(stones.counts,(6,)*8);self.assertEqual(stones.entry_extensions,(0.,)*8)

  def test_resume_version_and_old_routes_remain_separate(self):
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c,5)
    with self.assertRaises(ValueError):validate_skill_resume(c,{'infos':{'mgdp_skill_training':{'version':4}}})
    for version in (2,3,4,5):
      c=load_env_cfg(D435I20_TASK,play=True);restore_skill_for_play(c,{'infos':{'mgdp_skill_training':{'version':version}}})
      self.assertEqual(bool(c.commands['twist'].staged_waypoints),version==5)

  def test_acceptance_pins_one_level_and_rejects_invalid_before_reset(self):
    terrain=NS(terrain_origins=torch.arange(8*10*3).reshape(8,10,3).float(),
      terrain_levels=torch.tensor([0,7]),terrain_types=torch.tensor([3,9]),env_origins=torch.zeros(2,3))
    calls=[];env=NS(scene=NS(terrain=terrain),reset=lambda:calls.append(1))
    pin_full_course_level(env,6)
    self.assertEqual(calls,[1]);self.assertEqual(terrain.terrain_levels.tolist(),[5,5])
    torch.testing.assert_close(terrain.env_origins,terrain.terrain_origins[5,[3,9]])
    with self.assertRaises(ValueError):pin_full_course_level(env,9)
    self.assertEqual(calls,[1])

  def test_first_gap_group_finishes_after_hind_foot_landing_space(self):
    c=load_env_cfg(D435I20_TASK);configure_skill_training(c,5)
    gap=c.scene.terrain.terrain_generator.sub_terrains['gap_steps']
    for row in range(3):
      _,length,width,up,_=gap.gap_parameters((row+.5)/8)
      end=2+3*up+.8+width+.8+length
      goal=c.commands['twist'].staged_waypoints['gap_steps'][row][-1][0]+1
      self.assertAlmostEqual(goal-end,.6)
      self.assertLess(goal,end+gap.connector_length)

  def test_diagnostics_completed_counts_overlap_and_checkpoint_restore(self):
    env,cmd,robot=self.fixture();n=10
    env.episode_length_buf=torch.full((n,),100,dtype=torch.long)
    flags={name:torch.zeros(n,dtype=torch.bool) for name in env.cfg.terminations}
    flags['left_lane'][9]=True;flags['no_passage_progress'][9]=True
    flags['sustained_leg_contact'][3]=True;flags['passage_success'][4]=True
    env.termination_manager=NS(get_term=lambda name:flags[name])
    robot.data.root_link_pos_w[3,0]=2.78
    kinds=torch.arange(10);names=list(env.cfg.scene.terrain.terrain_generator.sub_terrains)
    term=PassageDiagnostics(env,names,kinds)
    ids=torch.tensor([3,4,9]);term.record(env,ids,torch.tensor([False,True,False]))
    data=term.snapshot();self.assertEqual(data['episodes'],3)
    gap=next(r for r in data['records'] if r['terrain']=='gap_steps');self.assertEqual(gap['phase'],'第一道沟')
    self.assertAlmostEqual(gap['mean_seconds'],2.)
    slalom=next(r for r in data['records'] if r['terrain']=='slalom_brick')
    self.assertEqual((slalom['reasons']['left_lane'],slalom['reasons']['no_passage_progress']),(1,1))
    self.assertEqual(sum(r['successes'] for r in data['records']),1)
    fresh=PassageDiagnostics(env,names,kinds);fresh.load_state_dict(term.state_dict())
    torch.testing.assert_close(fresh.count,term.count)
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'curriculum/passage_diagnostics.json';path.parent.mkdir();term.write(path)
      self.assertEqual(read_passage_diagnostics(tmp)['episodes'],3)
      bad=json.loads(path.read_text(encoding='utf8'));bad['records'][0]['episodes']=-1
      path.write_text(json.dumps(bad),encoding='utf8');self.assertIsNone(read_passage_diagnostics(tmp))
      path=Path(tmp)/'params/effective_wall_rules.json';path.parent.mkdir()
      path.write_text(json.dumps(effective_rule_snapshot(env.cfg)),encoding='utf8')
      self.assertEqual(read_effective_wall_rules(tmp)['skill_version'],5)

  def test_workbench_preview_uses_shared_resolver(self):
    from src.dashboard import routes
    saved={'settings':deepcopy(WALL_REWARD_DEFAULTS),'groups':{}}
    with patch.object(routes,'read_wall_reward_settings',return_value=saved):
      status,data=routes.wall_reward_preview(NS(root=Path('.')),{'task':'race_mgdp_wall_d435i20','skill_version':5})
      self.assertEqual(status,200);self.assertEqual(data['max_waiting_cost_per_second'],.5)
      with self.assertRaises(ValueError):routes.wall_reward_preview(NS(root=Path('.')),{'task':'race_mgdp_wall_d435i20','skill_version':True})


if __name__=='__main__':unittest.main()
