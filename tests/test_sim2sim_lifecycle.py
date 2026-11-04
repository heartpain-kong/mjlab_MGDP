"""Exercise the real control loop/physics with a tiny model and a fake window."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import mujoco
import mujoco.viewer
import numpy as np
from glfw import KEY_ESCAPE
from scripts import sim2sim


class Sim2simLifecycleTests(unittest.TestCase):
  def run_session(self, *, viewer=True, duration=None, close_at=1., escape_at=None,
                  interrupt=False, fall_at=None):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.005" gravity="0 0 0"/>
      <worldbody><body><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/>
      <body><joint name="j"/><geom type="sphere" size=".02" mass=".1"/></body>
      </body></worldbody><actuator><position joint="j"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    meta = dict(env_cfg=NS(observations={"actor":NS(history_length=1)}), nq_joint=1,
                joint_names=["j"], joint_of_actuator=["j"], action_scale=.25,
                default_qpos=np.zeros(1), ctrl_joint_perm=np.array([0]), n_terrain_geom=0)
    window = Mock()
    window.is_running.side_effect = lambda: data.time < close_at
    session = Mock()
    session.get_modelmeta.return_value = NS(custom_metadata_map={})
    def infer(*_):
      if interrupt:
        raise KeyboardInterrupt
      if fall_at is not None and session.run.call_count >= fall_at:
        data.qpos[3:7] = [2**-.5, 2**-.5, 0, 0]
      return [np.zeros((1,1), dtype=np.float32)]
    session.run.side_effect = infer
    def launch(*_, key_callback, **__):
      if escape_at is not None:
        def sync():
          if window.sync.call_count == escape_at:
            key_callback(KEY_ESCAPE)
        window.sync.side_effect = sync
      return window
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
      onnx = Path(tmp)/"policy.onnx"; onnx.touch()
      args = NS(onnx=str(onnx), task="test", map="flat", map_file=None,
                vx=0., vy=0., wz=0., viewer=viewer, duration=duration,
                spawn_x=0., spawn_y=0., spawn_yaw=0., action_clip=1.,
                check_obs=False, no_keyboard=True, heading_hold=False, camera_inset=False)
      with patch.object(sim2sim, "build_model", return_value=(model,data,meta)), \
           patch.object(sim2sim, "load_onnx", return_value=(session,"obs",[1,47],[1,1])), \
           patch.object(sim2sim, "build_obs", return_value=np.zeros(47,np.float32)), \
           patch.object(mujoco.viewer, "launch_passive", side_effect=launch), \
           patch.object(sim2sim.time, "sleep"), redirect_stdout(output):
        result = sim2sim.run(args)
    if viewer:
      window.close.assert_called_once()
    return result, session.run.call_count, output.getvalue()

  def test_open_ended_viewer_exceeds_old_timer_and_keeps_bounded_tail(self):
    code, steps, text = self.run_session(close_at=120.01)
    self.assertEqual(code, 0)
    self.assertGreater(steps, 6000)
    self.assertIn("无时长限制", text)
    self.assertIn("最近 25.00 仿真秒", text)
    self.assertIn("未进行定时姿态达标判定", text)

  def test_closed_window_before_first_step_is_clean_exit(self):
    code, steps, text = self.run_session(close_at=0)
    self.assertEqual((code, steps), (0,0))
    self.assertIn("尚未推进仿真", text)

  def test_escape_works_without_global_keyboard_listener(self):
    code, steps, _ = self.run_session(close_at=100, escape_at=30)
    self.assertEqual((code, steps), (0,30))

  def test_ctrl_c_cleans_up_before_any_sample(self):
    code, _, text = self.run_session(interrupt=True)
    self.assertEqual(code, 0)
    self.assertIn("按 Ctrl+C", text)

  def test_batch_default_and_explicit_timer_remain_bounded(self):
    code, steps, text = self.run_session(viewer=False)
    self.assertEqual((code, steps), (0,1000))
    self.assertIn("定时验证 20 秒", text)
    code, steps, _ = self.run_session(viewer=False, duration=.04)
    self.assertEqual((code, steps), (0,2))

  def test_fall_in_open_ended_session_remains_failure(self):
    code, steps, text = self.run_session(fall_at=10)
    self.assertEqual((code, steps), (1,10))
    self.assertIn("未通过", text)

  def test_invalid_explicit_durations_are_rejected(self):
    for duration in ("0", "-1", "nan", "inf", ".001"):
      with self.subTest(duration=duration), redirect_stderr(io.StringIO()), \
           patch.object(sys,"argv",["sim2sim.py","test","--duration",duration]), \
           self.assertRaises(SystemExit) as error:
        sim2sim.main()
      self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
  unittest.main()
