"""Sign validation/editor recovery without simulator dependencies."""

from pathlib import Path
import shutil
import tempfile
import unittest

from src.dashboard.reward_editor import SOURCE_RELATIVE, read_rewards, save_rewards
from src.reward_semantics import NONNEGATIVE_COSTS, validate_reward_weights

SOURCE = Path(__file__).resolve().parents[1] / SOURCE_RELATIVE


class RewardSemanticsTests(unittest.TestCase):
  def setUp(self):
    self.temp=tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root=Path(self.temp.name)
    self.source=self.root/SOURCE_RELATIVE
    self.source.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE,self.source)

  def test_positive_costs_are_rejected_but_zero_and_penalties_remain_editable(self):
    for name in NONNEGATIVE_COSTS:
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, "非负代价"):
        validate_reward_weights({name:1.0})
      validate_reward_weights({name:0.0})
      validate_reward_weights({name:-0.5})
    validate_reward_weights({"feet_air_time":1.0,"track_linear_velocity":2.0})

  def test_invalid_save_does_not_change_source_or_create_backup(self):
    before=self.source.read_bytes()
    with self.assertRaisesRegex(ValueError, "foot_clearance.*非负代价"):
      save_rewards(self.root,'him_standard',[{'name':'foot_clearance','weight':1.0}])
    self.assertEqual(self.source.read_bytes(),before)
    self.assertFalse(self.source.with_suffix('.py.bak').exists())

  def test_old_positive_override_is_readable_and_can_be_fixed(self):
    source=self.source.read_text(encoding='utf-8')
    source=source.replace("'him_standard': {'weights': {}, 'params': {}}}",
                          "'him_standard': {'weights': {'foot_clearance': 1.0}, 'params': {}}}")
    self.source.write_text(source,encoding='utf-8')
    rows={r['name']:r for r in read_rewards(self.root,'him_standard')['rewards']}
    self.assertEqual(rows['foot_clearance']['kind'],'cost')
    self.assertIn('正权重',rows['foot_clearance']['warning'])
    with self.assertRaisesRegex(ValueError,'foot_clearance'):
      save_rewards(self.root,'him_standard',[{'name':'feet_air_time','weight':1.0}])
    save_rewards(self.root,'him_standard',[{'name':'foot_clearance','weight':0.0,
                                          'params':{'target_height':.1}}])
    fixed={r['name']:r for r in read_rewards(self.root,'him_standard')['rewards']}
    self.assertEqual(fixed['foot_clearance']['weight'],0.0)
    self.assertIsNone(fixed['foot_clearance']['warning'])

  def test_v19_baseline_is_explicit_and_dashboard_overrides_are_empty(self):
    rows={r['name']:r for r in read_rewards(self.root,'him_standard')['rewards']}
    for name,value in dict(foot_clearance=0.,feet_stumble=-.5,feet_air_time=.25,
                           track_linear_velocity=2.,pose=1.5,action_rate_l2=-.1,
                           soft_landing=-.002).items():
      self.assertEqual(rows[name]['weight'],value)
      self.assertFalse(rows[name]['changed'])
    self.assertEqual(rows['track_linear_velocity']['params']['relative_weight'],0.)
    self.assertEqual(read_rewards(self.root,'flat')['rewards'][0]['name'],'track_linear_velocity')


if __name__ == '__main__': unittest.main()
