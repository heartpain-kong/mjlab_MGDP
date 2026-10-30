"""HIM-inspired explicit velocity and implicit response identification.

Method reference: Long et al., HIMLoco (arXiv:2312.11460).
See LGDOG_HIM.md for attribution and implementation differences. This module
implements the estimator for the local RSL-RL 5 API.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from src.him_contract import HIM_HISTORY_LENGTH, HIM_FRAME_DIM


@torch.no_grad()
def balanced_assignments(scores, epsilon=0.05, iterations=3):
  """Sinkhorn assignments in log space, with sample/prototype marginals."""
  if scores.ndim != 2 or scores.shape[0] == 0:
    raise ValueError("Expected a nonempty [batch, prototypes] score matrix")
  batch, prototypes = scores.shape
  log_q = scores.float().T / epsilon
  log_q -= torch.logsumexp(log_q.flatten(), dim=0)
  for _ in range(iterations):
    log_q -= torch.logsumexp(log_q, dim=1, keepdim=True) + math.log(prototypes)
    log_q -= torch.logsumexp(log_q, dim=0, keepdim=True) + math.log(batch)
  return (log_q + math.log(batch)).exp().T


class HIMEstimator(nn.Module):
  def __init__(self, frame_dim=HIM_FRAME_DIM, history_length=HIM_HISTORY_LENGTH, latent_dim=16, prototypes=32):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(frame_dim * history_length, 128), nn.ELU(),
      nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 3 + latent_dim),
    )
    self.target = nn.Sequential(
      nn.Linear(frame_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(),
      nn.Linear(64, latent_dim),
    )
    self.prototypes = nn.Parameter(torch.randn(prototypes, latent_dim))

  def encode(self, normalized_history):
    prediction = self.encoder(normalized_history.flatten(start_dim=1))
    return prediction[:, :3], F.normalize(prediction[:, 3:], dim=-1)

  def losses(self, normalized_history, next_frame, next_velocity):
    velocity, source = self.encode(normalized_history.detach())
    target = F.normalize(self.target(next_frame.detach()), dim=-1)
    with torch.no_grad():
      self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))
    scores_source = source @ self.prototypes.T
    scores_target = target @ self.prototypes.T
    assignments_source = balanced_assignments(scores_source)
    assignments_target = balanced_assignments(scores_target)
    # Match upstream's mean over both batch and prototype axes (factor 1/K).
    swap = -0.5 * (
      assignments_source * F.log_softmax(scores_target / 3.0, dim=-1)
      + assignments_target * F.log_softmax(scores_source / 3.0, dim=-1)
    ).mean()
    velocity_loss = F.mse_loss(velocity, next_velocity.detach())
    return velocity_loss, swap
