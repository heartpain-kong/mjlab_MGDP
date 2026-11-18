import tempfile,unittest
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace as NS
from src.dashboard.commands import build_command
from src.config.mgdp_effective_rules import resolve_wall_rules
from src.config.mgdp_stages import staged_routes
from src.wall_reward_settings import WALL_REWARD_DEFAULTS


class StageWorkbenchTests(unittest.TestCase):
  def test_acceptance_command_has_explicit_fixed_level(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);checkpoint=root/'model_0.pt';checkpoint.write_bytes(b'fixture')
      options={'mode':'play','task':'race_mgdp_wall_d435i20','checkpoint':str(checkpoint),
        'full_course':True,'full_course_level':6}
      argv=build_command(root,'python',options)
      self.assertIn('--full-course=True',argv);self.assertIn('--full-course-level=6',argv)
      options['full_course_level']=9
      with self.assertRaises(ValueError):build_command(root,'python',options)

  def test_disabled_waiting_and_gate_errors_are_visible_before_launch(self):
    settings=deepcopy(WALL_REWARD_DEFAULTS);settings['wall_waiting']['enabled']=False
    preview=resolve_wall_rules(settings,5)
    self.assertEqual(preview['max_waiting_cost_per_second'],0.)
    self.assertEqual(preview['settings']['wall_waiting']['weight'],0.)
    self.assertEqual(settings['wall_waiting']['weight'],-1.5)
    settings['progress_gate']['enabled']=False
    with self.assertRaisesRegex(ValueError,'门控'):resolve_wall_rules(settings,5)

  def test_short_first_section_has_landing_room_and_sixth_is_complete(self):
    full={'high_wall':((0.,0.),(5.3,0.)),'gap_steps':((0.,0.),(10.4,0.)),
      'slalom_brick':tuple((float(i),0.) for i in range(19))}
    gap=NS(connector_length=.8,gap_parameters=lambda d:(.06,.8,.04,.32,.32))
    bank=staged_routes(NS(waypoints=full),NS(sub_terrains={'gap_steps':gap}))
    self.assertAlmostEqual(bank['gap_steps'][0][-1][0],5.)
    for kind in full:self.assertEqual(bank[kind][5],full[kind])
    self.assertEqual(len(bank['slalom_brick'][0]),6)


if __name__=='__main__':unittest.main()
