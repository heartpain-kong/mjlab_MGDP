"""满级地形配额：只保留少量智能体在最高档，其余回落到低一级。"""
import unittest

import torch

from src.tasks.velocity.mdp.passage import apply_level_quota


class LevelQuotaTests(unittest.TestCase):
    def test_caps_the_top_level_and_returns_the_demoted_count(self):
        levels = torch.full((100,), 7)
        demoted = apply_level_quota(levels, start=7, fraction=.1)
        self.assertEqual(demoted, 90)
        self.assertEqual(int((levels == 7).sum()), 10)
        self.assertEqual(int((levels == 6).sum()), 90)

    def test_no_change_when_fraction_is_one_or_level_is_below_start(self):
        levels = torch.full((20,), 3)
        self.assertEqual(apply_level_quota(levels, start=7, fraction=.1), 0)
        self.assertTrue(torch.equal(levels, torch.full((20,), 3)))
        levels = torch.full((20,), 7)
        self.assertEqual(apply_level_quota(levels, start=7, fraction=1.), 0)

    def test_below_the_cap_is_untouched(self):
        levels = torch.tensor([7] * 3 + [5] * 97)
        self.assertEqual(apply_level_quota(levels, start=7, fraction=.1), 0)
        self.assertEqual(int((levels == 7).sum()), 3)

    def test_demoted_envs_keep_their_other_attributes(self):
        levels = torch.tensor([7, 7, 2, 0])
        demoted = apply_level_quota(levels, start=7, fraction=.25)
        self.assertEqual(demoted, 1)
        self.assertEqual(levels[2].item(), 2)
        self.assertEqual(levels[3].item(), 0)


if __name__ == "__main__":
    unittest.main()
