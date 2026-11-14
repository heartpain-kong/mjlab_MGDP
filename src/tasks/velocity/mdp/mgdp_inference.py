"""Frozen CNN inference: reusable graph per chunk shape, one full-batch GRU.

Graph inputs/outputs have stable storage; callers receive independent tokens.
Only the CNN may use BF16. Recurrent state and exported weights stay FP32.
"""
from contextlib import nullcontext

import torch


class FrozenDepthRuntime:
  def __init__(self, module, execution="eager", precision="fp32"):
    if execution not in ("eager", "cuda_graph") or precision not in ("fp32", "bf16"):
      raise ValueError("invalid MGDP inference execution/precision")
    self.module = module
    self.execution = execution
    self.precision = precision
    self.graphs = {}
    self.graph_replays = 0

  def _encode(self, depth):
    context = (torch.autocast(depth.device.type, dtype=torch.bfloat16, cache_enabled=False)
               if self.precision == "bf16" else nullcontext())
    with context:
      token = self.module.depth_encoder(depth)
    self.module.depth_encoder.skip_connections.clear()
    return token.float()

  def _capture(self, depth):
    # Never advance recurrent state or use live observations during warmup.
    static = torch.zeros_like(depth)
    stream = torch.cuda.Stream(device=depth.device)
    current = torch.cuda.current_stream(depth.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
      for _ in range(3):
        self._encode(static)
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
      output = self._encode(static)
    current.wait_stream(stream)
    print(f"[speed] frozen CNN CUDA graph: batch={len(depth)}, precision={self.precision}", flush=True)
    return graph, static, output

  @torch.no_grad()
  def encode(self, depth):
    if self.module.training:
      raise RuntimeError("frozen MGDP inference requires eval()")
    if self.precision == "bf16":
      if depth.is_cuda:
        with torch.cuda.device(depth.device):
          if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support BF16")
    if self.execution != "cuda_graph" or not depth.is_cuda:
      return self._encode(depth)
    key = (tuple(depth.shape), depth.device, depth.dtype)
    if key not in self.graphs:
      if len(self.graphs) >= 2:
        raise RuntimeError("MGDP graph runtime expects fixed full/tail chunk shapes")
      self.graphs[key] = self._capture(depth)
    graph, static, output = self.graphs[key]
    static.copy_(depth)
    graph.replay()
    self.graph_replays += 1
    # A later chunk replay overwrites graph output; do not expose that buffer.
    return output.clone()


def configure_perception_runtime(cfg, execution=None, precision=None, batch_size=None):
  """Override only MGDP perception terms; None preserves checkpoint settings."""
  group = cfg.observations.get("perception")
  if group is None:
    if any(value is not None for value in (execution, precision, batch_size)):
      raise ValueError("当前任务不使用 MGDP 感知模型")
    return
  if batch_size is not None and batch_size < 0:
    raise ValueError("perception batch size must be nonnegative")
  if execution is not None and execution not in ("eager", "cuda_graph"):
    raise ValueError("invalid perception execution")
  if precision is not None and precision not in ("fp32", "bf16"):
    raise ValueError("invalid perception precision")
  params = group.terms["depth_feature"].params
  for name, value in (("inference_execution", execution), ("inference_precision", precision),
                      ("inference_batch_size", batch_size)):
    if value is not None:
      params[name] = value
