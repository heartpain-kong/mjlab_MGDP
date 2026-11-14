"""Optional stage timings without a CUDA synchronization on every call."""
from contextlib import ExitStack
from time import perf_counter
from unittest.mock import patch

import torch
import warp as wp


class StageProfiler:
  def __init__(self, device):
    self.device = torch.device(device)
    self.records = {}
    self.stack = ExitStack()
    self.stream = (torch.cuda.ExternalStream(wp.get_stream(str(self.device)).cuda_stream,
                                            device=self.device)
                   if self.device.type == "cuda" else None)

  def watch(self, obj, method, label, stream="warp"):
    original = getattr(obj, method)
    records = self.records.setdefault(label, [])
    def measured(*args, **kwargs):
      if self.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timing_stream = self.stream if stream == "warp" else torch.cuda.current_stream(self.device)
        start.record(timing_stream)
        result = original(*args, **kwargs)
        end.record(timing_stream)
        records.append((start, end))
      else:
        start = perf_counter()
        result = original(*args, **kwargs)
        records.append(perf_counter() - start)
      return result
    self.stack.enter_context(patch.object(obj, method, measured))

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.stack.__exit__(*args)

  def report(self):
    # The caller synchronizes the device once after the entire rollout.
    return {label: {"calls": len(values), "seconds": sum(
      start.elapsed_time(end) / 1000 for start, end in values)
      if self.device.type == "cuda" else sum(values)}
      for label, values in self.records.items()}
