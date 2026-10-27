"""Real CPU process-tree/restart tests; no training is started or stopped."""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.dashboard import history
from src.dashboard.jobs import JobManager
from src.dashboard.process_control import process_info, same_process


class JobControlTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.root = Path(self.temp.name).resolve()
    self.manager = JobManager(self.root, on_event=lambda event, record:
                              history.record_run_event(self.root, event, record))
    self.addCleanup(self.cleanup)

  def cleanup(self):
    for record, process in self.manager.jobs.values():
      if process is not None and process.poll() is None:
        self.manager.stop(record["id"], force=True)
    self.temp.cleanup()

  def sleeper(self, mode="train"):
    return self.manager.launch([sys.executable, "-c", "import time; time.sleep(60)"], mode, "standard")

  def wait_done(self, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
      record = next(item for item in self.manager.list() if item["id"] == job["id"])
      if record["can_delete"]:
        return record
      time.sleep(0.03)
    self.fail("test subprocess did not exit")

  def test_live_task_is_controllable_after_dashboard_restart(self):
    job = self.sleeper()
    restored = JobManager(self.root)
    record = restored.list()[0]
    self.assertEqual(record["status"], "running")
    self.assertTrue(record["recovered"])
    self.assertTrue(record["can_stop"])
    self.assertFalse(record["can_delete"])
    self.assertEqual(restored.stop(job["id"])["status"], "stopped")
    self.manager.jobs[job["id"]][1].wait(timeout=3)

  def test_pid_reuse_never_stops_an_unrelated_process(self):
    job = self.sleeper()
    path = self.manager.directory / f"{job['id']}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["process_identity"]["token"] = "unrelated-creation-time"
    path.write_text(json.dumps(record), encoding="utf-8")
    restored = JobManager(self.root)
    self.assertFalse(restored.list()[0]["can_stop"])
    with patch("src.dashboard.jobs.terminate_tree") as kill:
      self.assertEqual(restored.stop(job["id"])["status"], "finished")
      kill.assert_not_called()
    self.assertIsNone(self.manager.jobs[job["id"]][1].poll())

  def test_recovered_live_task_blocks_duplicate_launch(self):
    self.sleeper()
    restored = JobManager(self.root)
    with self.assertRaisesRegex(ValueError, "同类操作"):
      restored.launch([sys.executable, "-c", "pass"], "train", "standard")

  @unittest.skipIf(os.name == "nt", "legacy /proc adoption is Linux-only")
  def test_legacy_linux_record_requires_exact_argv_and_cwd(self):
    job = self.sleeper()
    path = self.manager.directory / f"{job['id']}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("process_identity")
    record.pop("argv")  # oldest records only had the displayed command
    path.write_text(json.dumps(record), encoding="utf-8")
    restored = JobManager(self.root)
    self.assertTrue(restored.list()[0]["can_stop"])
    restored.stop(job["id"])
    self.manager.jobs[job["id"]][1].wait(timeout=3)

  def test_stop_cleans_child_even_when_parent_exits_first(self):
    child_file = self.root / "child.pid"
    child_code = (
      "import os,signal,time; from pathlib import Path; "
      "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
      f"Path({str(child_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
      "import subprocess,sys,time; "
      f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(60)"
    )
    job = self.manager.launch([sys.executable, "-c", parent_code], "train", "standard")
    deadline = time.monotonic() + 5
    while not child_file.exists() and time.monotonic() < deadline:
      time.sleep(0.03)
    self.assertTrue(child_file.exists())
    child = process_info(int(child_file.read_text()))
    self.assertIsNotNone(child)
    self.assertEqual(self.manager.stop(job["id"])["status"], "stopped")
    self.assertFalse(same_process(child))

  def test_state_remains_responsive_during_stop(self):
    job = self.sleeper()
    entered, release = threading.Event(), threading.Event()
    from src.dashboard.process_control import terminate_tree
    errors = []

    def slow_stop(*args, **kwargs):
      entered.set()
      release.wait(timeout=3)
      return terminate_tree(*args, **kwargs)

    def stop():
      try:
        self.manager.stop(job["id"])
      except Exception as error:
        errors.append(error)

    with patch("src.dashboard.jobs.terminate_tree", side_effect=slow_stop):
      thread = threading.Thread(target=stop)
      thread.start()
      try:
        self.assertTrue(entered.wait(timeout=2))
        started = time.monotonic()
        record = self.manager.list()[0]
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(record["status"], "stopping")
        self.assertFalse(record["can_stop"])
        with self.assertRaisesRegex(ValueError, "请先停止"):
          self.manager.delete(job["id"])
      finally:
        release.set()
        thread.join(timeout=8)
    self.assertFalse(thread.is_alive())
    self.assertEqual(errors, [])

  def test_failed_stop_is_visible_and_retryable(self):
    job = self.sleeper()
    with patch("src.dashboard.jobs.terminate_tree", side_effect=ValueError("driver busy")):
      with self.assertRaisesRegex(ValueError, "停止失败"):
        self.manager.stop(job["id"])
    record = self.manager.list()[0]
    self.assertTrue(record["can_stop"])
    self.assertFalse(record["can_delete"])
    self.assertIn("driver busy", record["stop_error"])
    self.assertEqual(self.manager.stop(job["id"], force=True)["status"], "stopped")

  def test_delete_rejects_live_job_and_invalid_ids(self):
    job = self.sleeper()
    with self.assertRaisesRegex(ValueError, "请先停止"):
      self.manager.delete(job["id"])
    for job_id in ("../outside", "", "a" * 13, None):
      with self.assertRaisesRegex(ValueError, "ID 无效"):
        self.manager.delete(job_id)
    self.assertIsNone(self.manager.jobs[job["id"]][1].poll())

  def test_delete_removes_both_lists_but_preserves_models_and_rewards(self):
    job = self.manager.launch([sys.executable, "-c", "print('done')"], "train", "standard")
    self.wait_done(job)
    model = self.root / "logs/rsl_rl/experiment/run/model_100.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"keep-model")
    reward_log = self.root / history.DIRECTORY / history.REWARD_LOG
    reward_log.write_text('{"kind":"reward_edit","changes":[]}\n', encoding="utf-8")
    self.assertEqual(self.manager.delete(job["id"]), {"deleted": [job["id"]]})
    self.assertEqual(self.manager.list(), [])
    self.assertFalse((self.manager.directory / f"{job['id']}.json").exists())
    self.assertFalse((self.manager.directory / f"{job['id']}.log").exists())
    self.assertEqual(history.read_history(self.root)["runs"], [])
    self.assertEqual(model.read_bytes(), b"keep-model")
    self.assertEqual(len(history.read_history(self.root)["reward_edits"]), 1)
    self.assertEqual(JobManager(self.root).list(), [])

  def test_clear_finished_preserves_running_and_unverified_records(self):
    done = self.manager.launch([sys.executable, "-c", "pass"], "rewards", "standard")
    self.wait_done(done)
    running = self.sleeper()
    old = {"id": "a" * 12, "mode": "play", "task": "standard", "pid": 99999999,
           "created": "2026-10-17T00:00:00+00:00", "status": "running"}
    self.manager._save(old)
    restored = JobManager(self.root)
    self.assertEqual(restored.clear_finished(), {"deleted": [done["id"]]})
    remaining = {item["id"]: item for item in restored.list()}
    self.assertTrue(remaining[running["id"]]["can_stop"])
    self.assertEqual(remaining[old["id"]]["status"], "unknown")
    self.assertTrue(remaining[old["id"]]["can_delete"])

  def test_delete_history_only_record(self):
    job = {"id": "b" * 12, "mode": "play", "task": "standard", "status": "completed"}
    history.record_run_event(self.root, "start", job)
    history.record_run_event(self.root, "end", job)
    self.assertEqual(self.manager.delete(job["id"])["deleted"], [job["id"]])
    self.assertEqual(history.read_history(self.root)["runs"], [])

  def test_cleanup_includes_records_older_than_visible_thirty(self):
    for index in range(35):
      self.manager._save({
        "id": f"{index:012x}", "mode": "play", "task": "standard", "pid": 99999999,
        "created": f"2026-10-17T00:00:{index:02d}+00:00", "status": "completed",
      })
    restored = JobManager(self.root)
    self.assertEqual(len(restored.list()), 30)
    self.assertEqual(restored.counts(), {"total": 35, "active": 0, "clearable": 35})
    self.assertEqual(len(restored.clear_finished()["deleted"]), 35)
    self.assertEqual(JobManager(self.root).list(), [])


if __name__ == "__main__":
  unittest.main()
