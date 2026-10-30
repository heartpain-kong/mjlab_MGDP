"""Deployment history contract shared by sim2sim and regression tests.

Only NumPy is required. Each push is one new 50 Hz observation frame.
"""

import numpy as np
from src.him_contract import HIM_HISTORY_LENGTH, HIM_FRAME_DIM, HIM_HISTORY_ORDER


class PolicyHistory:
  def __init__(self, length=HIM_HISTORY_LENGTH, frame_dim=HIM_FRAME_DIM):
    if length < 1 or frame_dim < 1:
      raise ValueError("History dimensions must be positive")
    self.length, self.frame_dim = length, frame_dim
    self.frames = None

  def reset(self):
    self.frames = None

  def push(self, frame):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (self.frame_dim,) or not np.isfinite(frame).all():
      raise ValueError(f"Expected a finite ({self.frame_dim},) observation frame")
    if self.frames is None:
      self.frames = np.repeat(frame[None, :], self.length, axis=0)
    else:
      self.frames[:-1] = self.frames[1:].copy()
      self.frames[-1] = frame
    return self.frames.reshape(-1).copy()


def validate_policy_history(input_shape, metadata, history_length, frame_dim=HIM_FRAME_DIM, control_dt=.02):
  """Fail before control starts if task/model history conventions disagree."""
  if len(input_shape) != 2 or input_shape[1] != history_length * frame_dim:
    raise ValueError(f"Policy shape {input_shape} does not match {history_length} x {frame_dim}")
  if history_length > 1:
    expected = {"policy_architecture": "lgdog_him_v1", "history_length": str(history_length),
                "frame_dim": str(frame_dim), "history_order": HIM_HISTORY_ORDER,
                "history_reset": "repeat_first_frame"}
    for key, value in expected.items():
      if metadata.get(key) != value:
        raise ValueError(f"ONNX metadata mismatch: {key}, expected {value!r}")
    if abs(float(metadata.get("control_dt", "nan")) - control_dt) > 1e-8 or not np.isfinite(
      float(metadata.get("control_dt", "nan"))
    ):
      raise ValueError("ONNX control period does not match deployment")
