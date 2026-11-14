"""Shared CLI/workbench resource defaults for perception pretraining."""

from dataclasses import dataclass
from src.config.mgdp_camera import D435I20_TASK,D435I20_ALIAS


@dataclass(frozen=True)
class PretrainResources:
  num_envs: int = 256
  terrain_rows: int = 12
  terrain_cols: int | None = None
  batch_size: int = 256


RACE_PRETRAIN = PretrainResources(num_envs=64, terrain_rows=4, terrain_cols=10, batch_size=64)
WALL_PRETRAIN = PretrainResources(num_envs=64, terrain_rows=8, terrain_cols=10, batch_size=64)
WALL_PRETRAIN_CHECKPOINT = 'logs/mgdp/perception/mgdp_wall_v5.pt'


def pretrain_resources(task: str) -> PretrainResources:
  if task in ('race_mgdp_wall', 'LgDog1-Race-MGDP-WallTraining',D435I20_TASK,D435I20_ALIAS):
    return WALL_PRETRAIN
  return RACE_PRETRAIN if task in ("race_mgdp", "LgDog1-Race-MGDP") else PretrainResources()
