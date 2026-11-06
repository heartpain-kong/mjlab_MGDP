"""Bound MJWarp model and sensor initialization memory peaks.

MJWarp 3.5.0 builds full int64 geom-pair indices before filtering. Replace only
that verified source block with chunked construction, retaining its int32 dense
lookup and all runtime collision fields. Check installed source before enabling.
Share initial BVH bounds staging and retain contexts for nonvisual DR expansion.
"""

import ast
import hashlib
import inspect
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import threading
import textwrap
from importlib.metadata import version, PackageNotFoundError

_BUILD_LOCK = threading.RLock()
_PAIR_BLOCK_SHA256 = "69d9f35f299d211624eb14231cf591e580a5b992e34380e9bfe42f69bdc66eb8"
_BVH_BUILD_SHA256 = "29656b286be00502eeb55fd77768b4850be6b018ce2328e5df7effc470c1ef7b"
_BVH_BOUNDS_SHA256 = "6172d6eca242a4b932437e0897b09d982eb4aa683d466042897ab40eb685ad27"
_SENSOR_RECREATE_SHA256 = "2643976a39ff0846060529d5fce214d463b3527f67cde67005e488672eb4e1d8"
_NONVISUAL_FIELDS = frozenset({"body_mass", "body_inertia", "body_ipos", "geom_friction",
                               "actuator_gainprm", "actuator_biasprm", "body_subtreemass",
                               "body_invweight0", "dof_invweight0", "tendon_length0",
                               "tendon_invweight0"})


@lru_cache(maxsize=1)
def _nonvisual_recreate_factory():
  from mjlab.sensor.sensor_context import SensorContext
  original = SensorContext.recreate
  try:
    if version("mjlab") != "1.2.0" or version("mujoco-warp") != "3.5.0":
      return None
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
    if hashlib.sha256(canonical.encode()).hexdigest() != _SENSOR_RECREATE_SHA256:
      return None
  except (OSError, TypeError, SyntaxError, PackageNotFoundError):
    return None

  def recreate(self, mj_model, expanded_fields=None):
    if (expanded_fields is not None and set(expanded_fields) <= _NONVISUAL_FIELDS
        and getattr(self, "_ctx", None) is not None and not self._disable_precomputed_rays):
      print("[memory] reusing sensor BVH after nonvisual model-field expansion: "
            + ", ".join(sorted(expanded_fields)), flush=True)
      return
    return original(self, mj_model, expanded_fields)

  return recreate


@lru_cache(maxsize=1)
def _shared_bvh_factory():
  from mujoco_warp._src import bvh
  try:
    if version("mujoco-warp") != "3.5.0":
      return None
    tree = ast.parse(inspect.getsource(bvh.build_scene_bvh))
    canonical = ast.dump(tree, include_attributes=False).replace(", type_params=[]", "")
    if hashlib.sha256(canonical.encode()).hexdigest() != _BVH_BUILD_SHA256:
      return None
    bounds = ast.parse(inspect.getsource(bvh._compute_bvh_bounds.func))
    canonical = ast.dump(bounds, include_attributes=False).replace(", type_params=[]", "")
    if hashlib.sha256(canonical.encode()).hexdigest() != _BVH_BOUNDS_SHA256:
      return None
    from src._mjwarp_bvh import build_scene_bvh
    return build_scene_bvh
  except (OSError, TypeError, SyntaxError, AttributeError, PackageNotFoundError):
    return None


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
  """Scope verified backend adaptations to environment construction only."""
  stats = {"unused_pair_bytes_avoided": 0, "streamed_pair_construction": False,
           "bvh_staging_bytes_avoided": 0}
  if not enabled:
    yield stats
    return
  import mujoco_warp as mjwarp
  from mujoco_warp._src import bvh
  from mjlab.sensor.sensor_context import SensorContext
  with _BUILD_LOCK:
    pair_safe = _unused_pair_table_safe()
    bvh_factory = _shared_bvh_factory()
    recreate_factory = _nonvisual_recreate_factory()
    if not pair_safe:
      try:
        backend = version("mujoco-warp")
      except PackageNotFoundError:
        backend = "unknown"
      print(f"[memory] Backend mujoco-warp {backend} pair-table use could not be verified; "
            "keeping native allocations", flush=True)
    original = mjwarp.put_model
    original_bvh = bvh.build_scene_bvh
    original_recreate = SensorContext.recreate

    def build(*args, **kwargs):
      print("[init] MuJoCo model compiled; building MJWarp model with bounded pair construction...", flush=True)
      model = _streaming_model_factory()(*args, **kwargs)
      stats["streamed_pair_construction"] = True
      stats["unused_pair_bytes_avoided"] += model.ngeom * (model.ngeom-1) * 4
      return model

    def build_bvh(mjm, mjd, rc, nworld):
      result = bvh_factory(mjm, mjd, rc, nworld)
      stats["bvh_staging_bytes_avoided"] += max(nworld-1,0) * mjm.ngeom * 60
      return result

    if pair_safe:
      print("[memory] bounded host pair construction enabled (verified mujoco-warp 3.5.0)", flush=True)
      mjwarp.put_model = build
    if bvh_factory is not None:
      print("[memory] shared BVH staging enabled (verified mujoco-warp 3.5.0)", flush=True)
      bvh.build_scene_bvh = build_bvh
    else:
      print("[memory] BVH construction source could not be verified; keeping native BVH staging", flush=True)
    if recreate_factory is not None:
      SensorContext.recreate = recreate_factory
    else:
      print("[memory] Sensor context reuse source could not be verified; "
            "keeping native context recreation", flush=True)
    try:
      yield stats
    finally:
      mjwarp.put_model = original
      bvh.build_scene_bvh = original_bvh
      SensorContext.recreate = original_recreate
