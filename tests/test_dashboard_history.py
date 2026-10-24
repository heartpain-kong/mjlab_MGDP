"""Audit trail for panel reward edits and launched training runs."""

import http.client
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest

from src.dashboard.jobs import JobManager
from src.dashboard.server import create_server
from src.dashboard import history


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_SOURCE = REPO_ROOT / "src/tasks/velocity/config/lgdog1/reward_cfg.py"


def read_jsonl(root, name):
  path = Path(root) / history.DIRECTORY / name
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RecordTests(unittest.TestCase):
  def test_only_changed_rewards_are_recorded(self):
    with tempfile.TemporaryDirectory() as directory:
      stored = history.record_reward_edit(
        directory, task="standard", profile="him_standard",
        before={"foot_clearance": {"weight": 0.0, "params": {"target_height": 0.1}},
                "pose": {"weight": 1.5, "params": {}}},
        after={"foot_clearance": {"weight": -0.8, "params": {"target_height": 0.1}},
               "pose": {"weight": 1.5, "params": {}}},
        source="/tmp/reward_cfg.py", backup="/tmp/reward_cfg.py.bak")
      self.assertIsNotNone(stored)
      records = read_jsonl(directory, history.REWARD_LOG)
      self.assertEqual(len(records), 1)
      record = records[0]
      self.assertEqual(record["kind"], "reward_edit")
      self.assertEqual(record["task"], "standard")
      self.assertEqual(record["profile"], "him_standard")
      # pose 没变，不应出现
      self.assertEqual([item["name"] for item in record["changes"]], ["foot_clearance"])
      self.assertEqual(record["changes"][0]["weight"], [0.0, -0.8])
      self.assertTrue(record["time"])

  def test_no_record_when_nothing_changed(self):
    with tempfile.TemporaryDirectory() as directory:
      same = {"pose": {"weight": 1.5, "params": {}}}
      self.assertIsNone(history.record_reward_edit(
        directory, task="standard", profile="him", before=same, after=dict(same), source="x"))
      self.assertEqual(read_jsonl(directory, history.REWARD_LOG), [])

  def test_history_survives_corrupt_lines(self):
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / history.DIRECTORY
      target.mkdir(parents=True)
      (target / history.REWARD_LOG).write_text(
        '{"kind": "reward_edit", "task": "standard", "profile": "him", "changes": [], '
        '"time": "2026-10-24T10:00:00"}\nnot json\n\n{"kind": "reward_edit", "task": "flat", '
        '"profile": "flat", "changes": [], "time": "2026-10-24T11:00:00"}\n', encoding="utf-8")
      edits = history.read_history(directory)["reward_edits"]
      self.assertEqual([item["task"] for item in edits], ["flat", "standard"])

  def test_argv_summary_extracts_training_flags(self):
    summary = history.summarize_argv([
      "python", "scripts/train.py", "LgDog1-Rough-Blind-HIM-Standard",
      "--env.scene.num-envs=1024", "--agent.max-iterations=16000",
      "--agent.seed=42", "--agent.run-name=V12", "--agent.resume=True"])
    self.assertEqual(summary, {"run_name": "V12", "envs": 1024,
                               "iterations": 16000, "seed": 42})

  def test_argv_summary_handles_sim_flags(self):
    summary = history.summarize_argv(["python", "scripts/sim2sim.py", "LgDog1-Flat",
                                      "--onnx=logs/a.onnx", "--map", "stairs"])
    self.assertEqual(summary.get("onnx"), "logs/a.onnx")


class RunRecordTests(unittest.TestCase):
  def test_job_events_are_recorded_with_status_and_duration(self):
    with tempfile.TemporaryDirectory() as directory:
      manager = JobManager(directory, on_event=lambda event, record:
                           history.record_run_event(directory, event, record))
      job = manager.launch([sys.executable, "-c", "print('history-test')"],
                           "train", "standard")
      deadline = time.monotonic() + 10
      while manager.list()[0]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
      self.assertEqual(manager.list()[0]["status"], "completed")
      runs = history.read_history(directory)["runs"]
      self.assertEqual(len(runs), 1)
      self.assertEqual(runs[0]["id"], job["id"])
      self.assertEqual(runs[0]["mode"], "train")
      self.assertEqual(runs[0]["status"], "completed")
      self.assertEqual(runs[0]["returncode"], 0)
      self.assertIsNotNone(runs[0]["duration_s"])
      self.assertTrue(runs[0]["started"])
      self.assertTrue(runs[0]["ended"])

  def test_recorder_failure_does_not_break_launch(self):
    def explode(event, record):
      raise RuntimeError("recorder down")
    with tempfile.TemporaryDirectory() as directory:
      manager = JobManager(directory, on_event=explode)
      job = manager.launch([sys.executable, "-c", "pass"], "rewards", "standard")
      self.assertTrue(job["id"])
      self.assertEqual(manager.stop(job["id"])["status"] in {"stopped", "completed"}, True)


class HistoryEndpointTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    root = Path(self.temp.name)
    target = root / "src/tasks/velocity/config/lgdog1"
    target.mkdir(parents=True)
    shutil.copy2(REWARD_SOURCE, target / "reward_cfg.py")
    self.root = root
    self.server = create_server(str(root), sys.executable, port=0)
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()
    self.addCleanup(self.cleanup)

  def cleanup(self):
    self.server.shutdown()
    self.server.server_close()
    self.thread.join(timeout=3)
    self.temp.cleanup()

  def request(self, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
    connection.request(method, path,
                       body=json.dumps(body) if body is not None else None,
                       headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data

  def token(self):
    return json.loads(self.request("GET", "/api/state")[1])["token"]

  def test_history_endpoint_returns_empty_shape(self):
    status, body = self.request("GET", "/api/history")
    self.assertEqual(status, 200)
    payload = json.loads(body)
    self.assertEqual(payload, {"reward_edits": [], "runs": []})

  def test_saving_rewards_writes_an_audit_record(self):
    headers = {"X-LgDog-Token": self.token(), "Content-Type": "application/json"}
    status, body = self.request("POST", "/api/rewards/save",
                                {"task": "standard",
                                 "changes": [{"name": "foot_clearance", "weight": -0.8,
                                              "params": {"target_height": 0.1}}]}, headers)
    self.assertEqual(status, 200)
    saved = json.loads(body)
    self.assertEqual(saved["profile"], "him_standard")
    status, body = self.request("GET", "/api/history")
    edits = json.loads(body)["reward_edits"]
    self.assertEqual(len(edits), 1)
    self.assertEqual(edits[0]["task"], "standard")
    self.assertEqual(edits[0]["profile"], "him_standard")
    names = [item["name"] for item in edits[0]["changes"]]
    self.assertEqual(names, ["foot_clearance"])
    self.assertEqual(edits[0]["changes"][0]["weight"], [0.0, -0.8])
    # 写回的面板覆盖块应包含新值，证明保存仍然生效
    source = (self.root / "src/tasks/velocity/config/lgdog1/reward_cfg.py").read_text(encoding="utf-8")
    self.assertIn("'foot_clearance': -0.8", source)


if __name__ == "__main__":
  unittest.main()
