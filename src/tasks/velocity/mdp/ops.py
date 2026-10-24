"""TorchScript kernels for hot velocity-task tensor operations."""

import torch


@torch.jit.script
def command_active(command: torch.Tensor, threshold: float) -> torch.Tensor:
  speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
  return (speed > threshold).to(command.dtype)


@torch.jit.script
def linear_velocity_tracking(
  command: torch.Tensor, actual: torch.Tensor, std: float
) -> torch.Tensor:
  error_xy = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  error_z = torch.square(actual[:, 2])
  return torch.exp(-(error_xy + 2.0 * error_z) / (std * std))


@torch.jit.script
def angular_velocity_tracking(
  command: torch.Tensor, actual: torch.Tensor, std: float
) -> torch.Tensor:
  error_z = torch.square(command[:, 2] - actual[:, 2])
  error_xy = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(error_z + 0.05 * error_xy) / (std * std))


@torch.jit.script
def commanded_distance(command: torch.Tensor, step_dt: float) -> torch.Tensor:
  speed = torch.linalg.vector_norm(command[:, :2], dim=1)
  return torch.where(speed > 0.1, speed * step_dt, torch.zeros_like(speed))


@torch.jit.script
def tracking_progress(
  command: torch.Tensor, velocity: torch.Tensor, step_dt: float
) -> torch.Tensor:
  speed = torch.linalg.vector_norm(command[:, :2], dim=1)
  along = torch.sum(velocity * command[:, :2], dim=1) / torch.clamp_min(speed, 1e-6)
  along = torch.minimum(torch.clamp_min(along, 0.0), speed)
  return torch.where(speed > 0.1, along * step_dt, torch.zeros_like(speed))


@torch.jit.script
def velocity_error_xy(command: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
  return torch.linalg.vector_norm(command[:, :2] - velocity, dim=1)
