"""Reclaim MJWarp's unused full geometry-pair table before data/graph allocation.

MJWarp 3.5.0 builds both full and filtered geom-pair arrays. Runtime collision
kernels use nxn_geom_pair_filtered; nxn_pairid is still required and is kept.
Check installed kernel sources before enabling this private-backend optimization.
"""

import ast
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import threading
from importlib.metadata import version, PackageNotFoundError

_BUILD_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _unused_pair_table_safe():
  import mujoco_warp
  directory = Path(mujoco_warp.__file__).parent / "_src"
  try:
    if version("mujoco-warp") != "3.5.0":
      return False
    io_source = (directory / "io.py").read_text(encoding="utf-8")
    if "np.triu_indices" not in io_source or "nxn_geom_pair_filtered" not in io_source:
      return False
    for path in directory.rglob("*.py"):
      if path.name.endswith("_test.py") or path.name.startswith("test_"):
        continue
      tree = ast.parse(path.read_text(encoding="utf-8"))
      if path.name == "io.py":
        tree.body = [node for node in tree.body if not (
          isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "put_model")]
      for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "nxn_geom_pair" and isinstance(node.ctx, ast.Load):
          return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
          if any(isinstance(arg, ast.Constant) and arg.value == "nxn_geom_pair" for arg in node.args):
            return False
    return True
  except (OSError, SyntaxError, PackageNotFoundError):
    return False


@contextmanager
def compact_model_allocations(enabled=True):
  """Scope the model factory adaptation to environment construction only."""
  stats = {"unused_pair_bytes_released": 0}
  if not enabled:
    yield stats
    return
  import mujoco_warp as mjwarp
  import warp as wp
  with _BUILD_LOCK:
    if not _unused_pair_table_safe():
      try:
        backend = version("mujoco-warp")
      except PackageNotFoundError:
        backend = "unknown"
      print(f"[memory] Backend mujoco-warp {backend} pair-table use could not be verified; "
            "keeping native allocations", flush=True)
      yield stats
      return
    original = mjwarp.put_model

    def build(*args, **kwargs):
      model = original(*args, **kwargs)
      array = model.nxn_geom_pair
      released = int(array.capacity)
      model.nxn_geom_pair = wp.empty((0,), dtype=array.dtype, device=array.device)
      del array
      stats["unused_pair_bytes_released"] += released
      print(f"[memory] Reclaimed unused full geom-pair table: {released/1024**2:.1f} MiB; "
            "filtered pairs, pair-id lookup and broadphase unchanged", flush=True)
      return model

    mjwarp.put_model = build
    try:
      yield stats
    finally:
      mjwarp.put_model = original
