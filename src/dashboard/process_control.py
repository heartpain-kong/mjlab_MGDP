"""OS process identities and bounded tree termination, without GPU/psutil imports.

A PID alone is never ownership proof: it can be reused after exit or reboot.
Linux identities include the boot UUID and kernel start tick; Windows uses the
kernel creation FILETIME.  Only a matching identity can be stopped after restart.
"""

from functools import lru_cache
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time


@lru_cache(maxsize=1)
def _boot_id():
  return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_info(pid):
  """Return None for a dead process; permission failures remain explicit."""
  if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
    return None
  if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle:
      error = ctypes.get_last_error()
      if error in (87, 1168):  # PID does not exist
        return None
      raise ctypes.WinError(error)
    try:
      if kernel.WaitForSingleObject(handle, 0) == 0:
        return None
      created, exited, kerneltime, usertime = (wintypes.FILETIME() for _ in range(4))
      if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in
                                            (created, exited, kerneltime, usertime))):
        raise ctypes.WinError(ctypes.get_last_error())
      token = str((created.dwHighDateTime << 32) | created.dwLowDateTime)
      return {"pid": pid, "token": token, "platform": "nt", "host": socket.gethostname()}
    finally:
      kernel.CloseHandle(handle)
  try:
    stat = (Path("/proc") / str(pid) / "stat").read_text()
  except (FileNotFoundError, ProcessLookupError):
    return None
  fields = stat.rsplit(")", 1)[1].split()
  if fields[0] in {"Z", "X"}:
    return None
  return {"pid": pid, "token": f"{_boot_id()}:{fields[19]}",
          "platform": "posix", "host": socket.gethostname(),
          "ppid": int(fields[1]), "pgid": int(fields[2]), "session": int(fields[3])}


def same_process(identity):
  if (not isinstance(identity, dict) or identity.get("host") != socket.gethostname()
      or identity.get("platform") != os.name):
    return False
  current = process_info(identity.get("pid"))
  return bool(current and all(current.get(key) == identity.get(key)
                              for key in ("pid", "token", "platform", "host")))


def legacy_identity(record, root):
  """Older Linux records can be adopted only after argv AND cwd match."""
  if os.name == "nt":
    return None
  pid = record.get("pid")
  current = process_info(pid)
  if current is None:
    return None
  try:
    directory = Path("/proc") / str(pid)
    args = directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    expected = record.get("argv") or shlex.split(record.get("command", ""))
    if [os.fsdecode(arg) for arg in args] != expected:
      return None
    if directory.joinpath("cwd").resolve(strict=True) != Path(root).resolve():
      return None
    return current
  except (OSError, ValueError):
    return None


def _windows_parents():
  import ctypes
  from ctypes import wintypes

  class ProcessEntry(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                ("ppid", wintypes.DWORD), ("priority", wintypes.LONG),
                ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]

  kernel = ctypes.WinDLL("kernel32", use_last_error=True)
  kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
  kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
  kernel.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
  kernel.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
  kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
  snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
  if snapshot == ctypes.c_void_p(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
  result = {}
  try:
    entry = ProcessEntry()
    entry.size = ctypes.sizeof(entry)
    found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
    while found:
      result[int(entry.pid)] = int(entry.ppid)
      found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    return result
  finally:
    kernel.CloseHandle(snapshot)


def process_tree(identity):
  """Snapshot descendants before their launcher can exit during termination."""
  if not same_process(identity):
    return []
  info = {}
  if os.name == "nt":
    parents = _windows_parents()
  else:
    parents = {}
    for directory in Path("/proc").iterdir():
      if not directory.name.isdigit():
        continue
      try:
        item = process_info(int(directory.name))
        if item:
          info[item["pid"]] = item
          parents[item["pid"]] = item["ppid"]
      except (OSError, ValueError):
        continue
  selected = {identity["pid"]}
  while True:
    extra = {pid for pid, parent in parents.items() if parent in selected} - selected
    if not extra:
      break
    selected.update(extra)
  result = []
  for pid in selected:
    try:
      item = info.get(pid) or process_info(pid)
      if item:
        result.append(item)
    except OSError:
      continue
  if not same_process(identity):
    return []
  return result


def terminate_tree(identity, tracked, force=False, timeout=3.0):
  """TERM -> bounded wait -> KILL, verifying every tracked process afterwards.

  Linux also signals the dedicated process group, so a worker forked during
  shutdown cannot escape just because its parent exits first.
  """
  if os.name == "nt":
    # Do the root first (/T includes descendants), then any captured survivors
    # if a venv wrapper exited before taskkill could inspect its children.
    for item in sorted(tracked, key=lambda p: p["pid"] != identity["pid"]):
      if not same_process(item):
        continue
      try:
        subprocess.run(["taskkill", "/PID", str(item["pid"]), "/T", "/F"],
                       capture_output=True, check=False, timeout=5,
                       creationflags=subprocess.CREATE_NO_WINDOW)
      except subprocess.TimeoutExpired:
        pass  # Verify below; a failure is surfaced, never reported as success.
  else:
    def send(sig):
      survivors = [item for item in tracked if same_process(item)]
      # Each newly launched job is its own session/group. Never signal a group
      # unless one of the verified owned processes still belongs to it.
      group_safe = any(
        (current := process_info(item["pid"]))
        and current["pgid"] == identity["pid"]
        and current["session"] == identity["pid"] for item in survivors
      )
      if group_safe:
        try:
          os.killpg(identity["pid"], sig)
        except ProcessLookupError:
          pass
      for item in survivors:
        try:
          if same_process(item):
            os.kill(item["pid"], sig)
        except ProcessLookupError:
          pass

    send(signal.SIGKILL if force else signal.SIGTERM)
    deadline = time.monotonic() + (0.0 if force else timeout)
    while time.monotonic() < deadline and any(same_process(p) for p in tracked):
      time.sleep(0.05)
    if any(same_process(p) for p in tracked):
      send(signal.SIGKILL)
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    remaining = [item for item in tracked if same_process(item)]
    if not remaining:
      return
    time.sleep(0.05)
  pids = ", ".join(str(item["pid"]) for item in remaining)
  raise ValueError(f"进程尚未退出（PID {pids}），请检查权限或点击强制结束")
