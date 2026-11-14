from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.velocity.rl.action_monitor import ActionMonitorVecEnvWrapper


def test_rollout_keeps_diagnostics_on_device_and_preserves_penalty():
  wrapper = object.__new__(ActionMonitorVecEnvWrapper)
  wrapper.env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=.02))
  wrapper.clip_actions = 1.
  wrapper.raw_action_penalty_coef = .5
  wrapper._raw_action_abs_max_seen = None
  wrapper._raw_action_abs_mean_last = None
  wrapper._action_saturation_fraction_last = None
  wrapper.action_monitor_steps = 0
  actions = torch.tensor([[2., -1.], [.2, -.4]])
  with patch.object(RslRlVecEnvWrapper, "step", return_value=(None, torch.ones(2), None, {})) as step:
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("rollout synchronized")):
      _, rewards, _, extras = wrapper.step(actions)
      wrapper.step(actions * .5)
    torch.testing.assert_close(step.call_args_list[0].args[0], actions.clamp(-1,1))
  torch.testing.assert_close(rewards, torch.tensor([.995, 1.]))
  assert wrapper.raw_action_abs_max_seen == 2.
  assert wrapper.raw_action_abs_mean_last == pytest.approx(.45)
  assert wrapper.action_saturation_fraction_last == .25
  assert wrapper.action_monitor_steps == 2
  assert extras["log"]["Metrics/raw_action_abs_mean"].ndim == 0
