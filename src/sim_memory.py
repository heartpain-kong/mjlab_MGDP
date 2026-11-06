"""Bound MJWarp's host construction peak before allocating physics data.

MJWarp 3.5.0 builds full int64 geom-pair indices before filtering. Replace only
that verified source block with chunked construction, retaining its int32 dense
lookup and all runtime collision fields. Check installed source before enabling.
"""

import ast
import hashlib
import inspect
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import threading
from importlib.metadata import version, PackageNotFoundError

_BUILD_LOCK = threading.RLock()
_PAIR_BLOCK_SHA256 = "69d9f35f299d211624eb14231cf591e580a5b992e34380e9bfe42f69bdc66eb8"


def _replace_pair_block(source):
  """Only replace the exact upstream block we have compared with native output."""
  tree = ast.parse(source)
  function = tree.body[0]
  starts = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "filterparent" for t in node.targets)]
  ends = [i+1 for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
          and any(isinstance(t, ast.Attribute) and t.attr == "geom_pair_type_count" for t in node.targets)]
  if len(starts) != 1 or len(ends) != 1:
    raise ValueError("unrecognized MJWarp pair construction")
  start, end = starts[0], ends[0]
  block = ast.Module(body=function.body[start:end], type_ignores=[])
  # Python 3.12 adds empty type_params to nested FunctionDefs; Linux uses 3.11.
  canonical = ast.dump(block, include_attributes=False).replace(", type_params=[]", "")
  digest = hashlib.sha256(canonical.encode()).hexdigest()
  if digest != _PAIR_BLOCK_SHA256:
    raise ValueError("MJWarp pair construction source changed")
  call = ast.parse("sensor_collision_adr = _lgdog_build_pairs(mjm, m, opt, types)").body[0]
  ast.copy_location(call, function.body[start])
  function.body[start:end] = [call]
  return ast.fix_missing_locations(tree)


@lru_cache(maxsize=1)
def _streaming_model_factory():
  from mujoco_warp._src import io
  from src._mjwarp_pairs import build_collision_pairs
  try:
    tree = _replace_pair_block(inspect.getsource(io.put_model))
    namespace = dict(io.put_model.__globals__, _lgdog_build_pairs=build_collision_pairs)
    exec(compile(tree, "<lgdog-streaming-mjwarp-put-model>", "exec"), namespace)
    return namespace["put_model"]
  except (OSError, TypeError, SyntaxError, ValueError):
    return None


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
    if _streaming_model_factory() is None:
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
  stats = {"unused_pair_bytes_avoided": 0, "streamed_pair_construction": False}
  if not enabled:
    yield stats
    return
  import mujoco_warp as mjwarp
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
    print("[memory] bounded host pair construction enabled (verified mujoco-warp 3.5.0)", flush=True)

    factory = _streaming_model_factory()

    def build(*args, **kwargs):
      print("[init] MuJoCo model compiled; building MJWarp model with bounded pair construction...", flush=True)
      model = factory(*args, **kwargs)
      stats["streamed_pair_construction"] = True
      stats["unused_pair_bytes_avoided"] += model.ngeom * (model.ngeom-1) * 4
      return model

    mjwarp.put_model = build
    try:
      yield stats
    finally:
      mjwarp.put_model = original
