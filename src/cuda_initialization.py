"""Localize asynchronous initialization failures without changing CUDA kernels."""

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import sys
import threading
import traceback

from src.scene_diagnostics import report_device_memory

_TRACE_LOCK = threading.RLock()


class InitializationTrace:
  def __init__(self, device, directory):
    self.device = str(device)
    self.directory = Path(directory)
    self.stack = []
    self.last_completed = None
    self.failed_phase = None

  def _write_progress(self):
    try:
      self.directory.mkdir(parents=True, exist_ok=True)
      target = self.directory / "init_stage.json"
      temporary = target.with_suffix(".tmp")
      temporary.write_text(json.dumps({"device": self.device,
        "active_phase": self.stack[-1] if self.stack else None,
        "last_completed": self.last_completed, "failed_phase": self.failed_phase},
        ensure_ascii=False, indent=2), encoding="utf-8")
      temporary.replace(target)
    except OSError as error:
      print(f"[init] could not write stage record: {error}", file=sys.stderr, flush=True)

  def _write_error(self, phase, error):
    # Record before higher frames/partial environments are destroyed. Keep the
    # most specific failure when enclosing phases subsequently see the same error.
    print(f"[init:error] {phase}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    if self.failed_phase is None:
      self.failed_phase = phase
    try:
      self.directory.mkdir(parents=True, exist_ok=True)
      with (self.directory / "init_error.json").open("x", encoding="utf-8") as stream:
        json.dump({"phase": phase, "device": self.device,
          "last_completed": self.last_completed, "error_type": type(error).__name__,
          "message": str(error), "traceback": traceback.format_exc()}, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
      pass
    except OSError as write_error:
      print(f"[init] could not write error record: {write_error}", file=sys.stderr, flush=True)

  @contextmanager
  def phase(self, name):
    self.stack.append(name)
    self._write_progress()
    print(f"[init:phase] {name} begin", flush=True)
    try:
      # This sync is outside graph capture. Do not attribute an already-pending
      # CUDA error to the next allocation or leave it until destructor cleanup.
      report_device_memory(self.device, name + "/before", strict=True)
      yield
      report_device_memory(self.device, name + "/after", strict=True)
    except Exception as error:
      self._write_error(name, error)
      raise
    else:
      self.last_completed = name
      print(f"[init:phase] {name} complete", flush=True)
    finally:
      self.stack.pop()
      self._write_progress()

  def wrap(self, name, function):
    def checked(*args, **kwargs):
      with self.phase(name):
        return function(*args, **kwargs)
    return checked


@contextmanager
def trace_cuda_initialization(device, directory):
  if not str(device).startswith("cuda"):
    yield
    return
  import mujoco_warp as mjwarp
  from mjlab.sim import Simulation
  # Enter after compact_model_allocations so these wrappers surround its model
  # factory too. Always restore the factories, including on failed construction.
  with _TRACE_LOCK:
    trace = InitializationTrace(device, directory)
    with ExitStack() as stack:
      def replace(target, attribute, phase):
        original = getattr(target, attribute)
        stack.callback(setattr, target, attribute, original)
        setattr(target, attribute, trace.wrap(phase, original))
      for attribute, phase in (("put_model", "model-upload"),
                               ("put_data", "physics-data"),
                               ("create_render_context", "sensor-context")):
        replace(mjwarp, attribute, phase)
      replace(Simulation, "create_graph", "simulation-graphs")
      with trace.phase("environment-construction"):
        yield


@contextmanager
def trace_cuda_execution(device, directory):
  """Optional sync boundaries around execution, outside graph capture.

  In particular reset launches forward before sensing; report a failing forward
  here instead of attributing its pending error to the next Torch ray operation.
  """
  if not str(device).startswith("cuda"):
    yield
    return
  from mjlab.sim import Simulation
  with _TRACE_LOCK:
    trace = InitializationTrace(device, directory)
    with ExitStack() as stack:
      for attribute in ("forward", "step", "sense"):
        original = getattr(Simulation, attribute)
        stack.callback(setattr, Simulation, attribute, original)
        setattr(Simulation, attribute, trace.wrap("runtime-" + attribute, original))
      yield
