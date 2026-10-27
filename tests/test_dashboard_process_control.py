"""Exercise POSIX escalation on Windows too, without signalling real PIDs."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.dashboard import process_control

SIGTERM, SIGKILL = 15, 9


class PosixTerminationTests(unittest.TestCase):
  def setUp(self):
    # Windows' signal module has no SIGKILL; supply POSIX constants only in
    # this mocked branch, never patch the host OS or its real kill functions.
    self.enterContext(patch.object(process_control, "signal", SimpleNamespace(
      SIGTERM=SIGTERM, SIGKILL=SIGKILL)))

  def identities(self):
    return [{"pid": 123450}, {"pid": 123451}]

  def test_sigterm_escalates_and_verifies_surviving_child(self):
    parent, child = self.identities()
    alive = {parent["pid"], child["pid"]}

    def kill_group(pid, sig):
      if sig == SIGTERM:
        alive.discard(parent["pid"])
      else:
        alive.clear()

    fake_os = SimpleNamespace(name="posix", killpg=Mock(side_effect=kill_group), kill=Mock())
    with patch.object(process_control, "os", fake_os), \
         patch.object(process_control, "same_process", side_effect=lambda item: item["pid"] in alive), \
         patch.object(process_control, "process_info", side_effect=lambda pid: {
           "pid": pid, "pgid": parent["pid"], "session": parent["pid"],
         }):
      process_control.terminate_tree(parent, [parent, child], timeout=0.001)
    self.assertEqual([call.args[1] for call in fake_os.killpg.call_args_list],
                     [SIGTERM, SIGKILL])
    self.assertEqual(alive, set())

  def test_only_verified_individuals_are_signalled_outside_owned_group(self):
    parent, child = self.identities()
    alive = {child["pid"]}
    fake_os = SimpleNamespace(name="posix", killpg=Mock(), kill=Mock(
      side_effect=lambda pid, sig: alive.discard(pid)))
    with patch.object(process_control, "os", fake_os), \
         patch.object(process_control, "same_process", side_effect=lambda item: item["pid"] in alive), \
         patch.object(process_control, "process_info", return_value={"pgid": 999, "session": 999}):
      process_control.terminate_tree(parent, [parent, child], force=True)
    fake_os.killpg.assert_not_called()
    fake_os.kill.assert_called_once_with(child["pid"], SIGKILL)

  def test_reused_or_exited_pids_are_never_signalled(self):
    parent, child = self.identities()
    fake_os = SimpleNamespace(name="posix", killpg=Mock(), kill=Mock())
    with patch.object(process_control, "os", fake_os), \
         patch.object(process_control, "same_process", return_value=False):
      process_control.terminate_tree(parent, [parent, child], force=True)
    fake_os.killpg.assert_not_called()
    fake_os.kill.assert_not_called()


if __name__ == "__main__":
  unittest.main()
