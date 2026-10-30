"""Launcher/HTTP/process tests with no GPU or robotics imports."""

import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.dashboard.commands import build_command, discover_models
from src.dashboard.jobs import JobManager
from src.dashboard.server import create_server, read_terrain_curriculum_status


class CommandTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name).resolve()

  def checkpoint(self, experiment="lgdog1_velocity_blind_him_standard_v1", run="2026-10-17", name="model_100.pt"):
    path = self.root / "logs/rsl_rl" / experiment / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test fixture", encoding="utf-8")
    return path

  def test_standard_training_preview_uses_new_task_without_changing_arguments(self):
    command = build_command(self.root, sys.executable, {"mode":"train", "envs":64, "iterations":2})
    self.assertIn("LgDog1-Rough-Blind-HIM-Standard", command)
    self.assertIn("--env.scene.num-envs=64", command)
    self.assertIn("--agent.max-iterations=2", command)
    self.assertEqual(command[0], sys.executable)
    with self.assertRaisesRegex(ValueError, "实验名"):
      build_command(self.root, sys.executable, {"mode":"train", "run_name":"bad; command"})

  def test_training_preview_converts_positive_speed_bounds_to_cli_flags(self):
    command = build_command(self.root, sys.executable, {
      "mode": "train", "velocity_min_abs": "0.4", "velocity_max_abs": "0.8",
    })
    self.assertIn("--velocity-min-abs=0.4", command)
    self.assertIn("--velocity-max-abs=0.8", command)
    with self.assertRaisesRegex(ValueError, "最低速度绝对值"):
      build_command(self.root, sys.executable, {"mode": "train", "velocity_min_abs": "-0.1"})
    with self.assertRaisesRegex(ValueError, "不能大于最高"):
      build_command(self.root, sys.executable, {
        "mode": "train", "velocity_min_abs": "0.9", "velocity_max_abs": "0.8",
      })

  def test_automatic_terrain_flags_are_validated_and_flat_is_rejected(self):
    for mode in ('uniform','lp_acrl'):
      argv=build_command(self.root,sys.executable,{'mode':'train','terrain_curriculum':mode,
                                                'lp_temperature':'2','lp_uniform_mix':'0'})
      self.assertIn('--terrain-curriculum='+mode,argv)
      self.assertIn('--lp-temperature=2.0',argv)
      self.assertIn('--lp-uniform-mix=0.0',argv)
      self.assertIn('--lp-progress-alpha=0.25',argv)
      self.assertIn('--lp-probability-alpha=0.2',argv)
      self.assertIn('--lp-max-task-probability=0.25',argv)
    for options in ({'terrain_curriculum':'bad'},
                    {'terrain_curriculum':'lp_acrl','task':'him_flat'},
                    {'terrain_curriculum':'lp_acrl','lp_temperature':'nan'},
                    {'terrain_curriculum':'lp_acrl','lp_min_episodes':'0'},
                    {'terrain_curriculum':'lp_acrl','lp_progress_alpha':'0'},
                    {'terrain_curriculum':'lp_acrl','lp_probability_alpha':'nan'},
                    {'terrain_curriculum':'lp_acrl','lp_max_task_probability':'2'}):
      with self.subTest(options=options),self.assertRaises(ValueError):
        build_command(self.root,sys.executable,{'mode':'train',**options})

  def test_status_parser_accepts_own_schema_and_ignores_partial_or_bad_data(self):
    p=self.root/'curriculum/terrain_sampling.json';p.parent.mkdir()
    valid={'version':1,'mode':'lp_acrl','tasks':[{'name':'flat','terrain':'flat',
                                               'probability':1.,'learning_progress':0.}]}
    p.write_text(json.dumps(valid),encoding='utf-8')
    self.assertEqual(read_terrain_curriculum_status(self.root),valid)
    for text in ['{',json.dumps({**valid,'mode':'bad'}),
                 json.dumps({**valid,'tasks':[{'name':'flat','terrain':'flat','probability':float('nan')}]})]:
      p.write_text(text,encoding='utf-8')
      self.assertIsNone(read_terrain_curriculum_status(self.root))

  def test_latest_numeric_checkpoint_skips_new_empty_run(self):
    self.checkpoint(name="model_99.pt")
    chosen = self.checkpoint(name="model_100.pt")
    (chosen.parent.parent / "2026-10-18_empty").mkdir()
    command = build_command(self.root, sys.executable, {"mode":"play"})
    self.assertIn(f"--checkpoint_file={chosen}", command)
    self.assertEqual(len(discover_models(self.root)), 2)

  def test_shared_experiment_requires_explicit_checkpoint(self):
    path = self.checkpoint("lgdog1_velocity")
    with self.assertRaisesRegex(ValueError, "共用"):
      build_command(self.root, sys.executable, {"mode":"play", "task":"flat"})
    command = build_command(self.root, sys.executable, {"mode":"play", "task":"flat", "checkpoint":str(path)})
    self.assertIn(f"--checkpoint_file={path}", command)

  def test_resume_rejects_other_task_and_uses_exact_run_pattern(self):
    old = self.checkpoint("lgdog1_velocity_blind_him_v1")
    with self.assertRaisesRegex(ValueError, "只能选择当前算法"):
      build_command(self.root, sys.executable, {"mode":"train", "resume":True, "checkpoint":str(old)})
    new = self.checkpoint(run="2026-10-17_V8.a")
    command = build_command(self.root, sys.executable, {"mode":"train", "resume":True, "checkpoint":str(new)})
    self.assertIn(r"--agent.load-run=2026\-10\-17_V8\.a$", command)
    self.assertIn(r"--agent.load-checkpoint=model_100\.pt$", command)

  def test_sim_validates_values_and_preserves_paths_with_spaces(self):
    path = self.checkpoint(run="run with spaces", name="policy.onnx")
    command = build_command(self.root, sys.executable, {"mode":"sim", "onnx":str(path)})
    self.assertIn(f"--onnx={path}", command)
    self.assertIn("--no-keyboard", command)
    with self.assertRaisesRegex(ValueError, "vx"):
      build_command(self.root, sys.executable, {"mode":"sim", "onnx":str(path), "vx":"nan"})


class ProcessTests(unittest.TestCase):
  def test_owned_process_output_completion_history_and_stop(self):
    with tempfile.TemporaryDirectory() as directory:
      manager = JobManager(directory)
      done = manager.launch([sys.executable, "-c", "print('dashboard-test')"], "rewards", "standard")
      deadline = time.monotonic()+10
      while manager.list()[0]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
      self.assertEqual(manager.list()[0]["status"], "completed")
      self.assertIn("dashboard-test", manager.tail(done["id"]))
      sleeper = manager.launch([sys.executable, "-c", "import time; time.sleep(60)"], "rewards", "standard")
      try:
        with self.assertRaisesRegex(ValueError, "同类操作"):
          manager.launch([sys.executable, "-c", "pass"], "rewards", "standard")
        self.assertEqual(manager.stop(sleeper["id"])["status"], "stopped")
      finally:
        manager.stop(sleeper["id"])
      restored = JobManager(directory)
      self.assertTrue(all(not item["can_stop"] for item in restored.list()))
      self.assertEqual(restored.stop(done["id"])["status"], "completed")


class HTTPTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.server = create_server(self.temp.name, sys.executable, port=0)
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()
    self.addCleanup(self.cleanup_server)

  def cleanup_server(self):
    self.server.shutdown()
    self.server.server_close()
    self.thread.join(timeout=3)
    self.temp.cleanup()

  def request(self, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
    connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data

  def test_page_and_read_only_state_work_without_robotics_dependencies(self):
    status, body = self.request("GET", "/")
    self.assertEqual(status, 200)
    self.assertIn("训练工作台", body.decode())
    self.assertIn("最低速度绝对值", body.decode())
    self.assertIn("速度追踪 / 地形课程", body.decode())
    self.assertIn("策略裁剪比例", body.decode())
    self.assertIn("打开 3D 查看器", body.decode())
    self.assertIn("当前地形 / 编辑", body.decode())
    self.assertIn("LP 地形采样（稳定版 v2）", body.decode())
    status, body = self.request("GET", "/api/state")
    self.assertEqual(status, 200)
    state = json.loads(body)
    self.assertEqual(len(state["tasks"]), 8)
    self.assertEqual({item["alias"] for item in state["algorithms"]}, {"him", "ppo"})
    self.assertIn("lgdog1", {item["alias"] for item in state["sim2sim_maps"]})
    self.assertEqual(self.request("GET", "/assets/../../setup.py")[0], 404)

  def test_lp_metrics_fallback_and_status_file_refresh_over_http(self):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    run=Path(self.temp.name)/'logs/rsl_rl/lgdog1_velocity_blind_him_standard_v1/run'
    run.mkdir(parents=True)
    writer=EventFileWriter(str(run))
    writer.add_event(Event(step=10,summary=Summary(value=[
      Summary.Value(tag='Curriculum/terrain_sampling/level_mean',simple_value=2.),
      Summary.Value(tag='Metrics/swing_clearance_height_numerator',simple_value=.015),
      Summary.Value(tag='Metrics/swing_clearance_valid_fraction',simple_value=.25)])))
    writer.flush();writer.close()
    p=run/'curriculum/terrain_sampling.json';p.parent.mkdir()
    data={'version':1,'mode':'lp_acrl','updates':1,
          'tasks':[{'name':'flat/row_all','terrain':'flat','probability':1.,'learning_progress':0.}]}
    p.write_text(json.dumps(data),encoding='utf-8')
    result={};deadline=time.monotonic()+5
    while time.monotonic()<deadline:
      status,body=self.request('GET','/api/metrics?task=standard')
      self.assertEqual(status,200)
      result=json.loads(body)
      if result.get('available'):break
      time.sleep(.05)
    self.assertEqual(result['metrics']['terrain_level']['value'],2.)
    self.assertAlmostEqual(result['metrics']['foot_clearance_height']['value']/result['metrics']['foot_clearance_valid']['value'],.06,places=5)
    self.assertEqual(result['terrain_curriculum']['mode'],'lp_acrl')
    data['updates']=2;p.write_text(json.dumps(data),encoding='utf-8')
    # No new event required: the cached scalar response must read fresh LP status.
    result=json.loads(self.request('GET','/api/metrics?task=standard')[1])
    self.assertEqual(result['terrain_curriculum']['updates'],2)

  def test_launch_requires_token_and_rejects_cross_origin(self):
    self.assertEqual(self.request("POST", "/api/run", {"mode":"train"})[0], 403)
    token = json.loads(self.request("GET", "/api/state")[1])["token"]
    headers = {"X-LgDog-Token":token, "Content-Type":"application/json"}
    status, body = self.request("POST", "/api/preview", {"mode":"train", "envs":64}, headers)
    self.assertEqual(status, 200)
    self.assertIn("--env.scene.num-envs=64", json.loads(body)["argv"])
    headers["Origin"] = "https://unrelated.example"
    self.assertEqual(self.request("POST", "/api/run", {"mode":"train"}, headers)[0], 403)
    self.assertEqual(self.request("GET", "/api/state", headers={"Host":"unrelated.example"})[0], 403)

  def test_validated_run_calls_only_the_command_builder_output(self):
    token = json.loads(self.request("GET", "/api/state")[1])["token"]
    with patch.object(self.server.jobs, "launch", return_value={"id":"test"}) as launch:
      status, _ = self.request("POST", "/api/run", {"mode":"rewards", "term":"foot_clearance"}, {"X-LgDog-Token":token})
      self.assertEqual(status, 200)
      self.assertIn("list_rewards.py", launch.call_args.args[0][1])
      self.assertIn("foot_clearance", launch.call_args.args[0])

  def test_delete_endpoints_require_token_and_refuse_running_job(self):
    self.assertEqual(self.request("POST", "/api/jobs/delete", {"id": "a" * 12})[0], 403)
    self.assertEqual(self.request("POST", "/api/jobs/clear-finished", {})[0], 403)
    token = json.loads(self.request("GET", "/api/state")[1])["token"]
    headers = {"X-LgDog-Token": token}
    job = self.server.jobs.launch([sys.executable, "-c", "import time; time.sleep(60)"], "train", "standard")
    try:
      self.assertEqual(self.request("POST", "/api/jobs/delete", {"id": job["id"]}, headers)[0], 400)
      status, body = self.request("POST", "/api/jobs/clear-finished", {}, headers)
      self.assertEqual((status, json.loads(body)["deleted"]), (200, []))
      status, body = self.request("POST", "/api/stop", {"id": job["id"], "force": True}, headers)
      self.assertEqual(status, 200, body)
      self.assertEqual(json.loads(body)["job"]["status"], "stopped")
      runs = json.loads(self.request("GET", "/api/history")[1])["runs"]
      self.assertTrue(runs[0]["can_delete"])
      status, body = self.request("POST", "/api/jobs/delete", {"id": job["id"]}, headers)
      self.assertEqual((status, json.loads(body)["deleted"]), (200, [job["id"]]))
      self.assertEqual(json.loads(self.request("GET", "/api/state")[1])["jobs"], [])
      self.assertEqual(json.loads(self.request("GET", "/api/history")[1])["runs"], [])
    finally:
      if job["id"] in self.server.jobs.jobs:
        self.server.jobs.stop(job["id"], force=True)

  def test_stop_reports_failure_instead_of_false_success(self):
    token = json.loads(self.request("GET", "/api/state")[1])["token"]
    with patch.object(self.server.jobs, "stop", side_effect=ValueError("停止失败：driver busy")):
      status, body = self.request("POST", "/api/stop", {"id": "a" * 12}, {"X-LgDog-Token": token})
    self.assertEqual(status, 400)
    self.assertIn("driver busy", json.loads(body)["error"])


if __name__ == "__main__":
  unittest.main()
