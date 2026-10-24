"""Small helpers for automatic, collision-free experiment labels."""

from __future__ import annotations

import os
import re
from pathlib import Path


_VERSION_RE = re.compile(r"(?:^|_)V(\d+)(?=_|$)", re.IGNORECASE)


def next_run_name(log_root: Path, seed: int) -> str:
  """Return the next ``V<n>_seed<seed>`` label in an experiment directory.

  The version is taken only from existing run directory names.  Timestamps,
  checkpoint numbers, and unrelated folders do not affect the counter.
  """
  log_root = Path(log_root).resolve()
  highest = 0
  roots = [log_root]
  # Completed runs in this workspace are also archived in E:\mjlab\数据.
  # Include that sibling archive so a fresh checkout continues at the next V.
  archive = os.environ.get("LGDOG_RUN_ARCHIVE", "").strip()
  if archive:
    roots.append(Path(archive).expanduser().resolve())
  else:
    for parent in log_root.parents:
      sibling_archive = parent / "数据"
      if sibling_archive.is_dir():
        roots.append(sibling_archive)
        break
  seen = set()
  for root in roots:
    if root in seen or not root.is_dir():
      continue
    seen.add(root)
    for path in root.iterdir():
      if not path.is_dir():
        continue
      match = _VERSION_RE.search(path.name)
      if match:
        highest = max(highest, int(match.group(1)))
  return f"V{highest + 1}_seed{int(seed)}"


def resolve_run_name(log_root: Path, requested: str, seed: int) -> str:
  """Keep an explicit label; otherwise allocate the next V-number."""
  requested = str(requested or "").strip()
  return requested or next_run_name(log_root, seed)
