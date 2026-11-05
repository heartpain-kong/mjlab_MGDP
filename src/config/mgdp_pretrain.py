"""Shared CLI/workbench resource defaults for perception pretraining."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PretrainResources:
  num_envs: int = 256
  terrain_rows: int = 12
  terrain_cols: int | None = None
  batch_size: int = 256


RACE_PRETRAIN = PretrainResources(num_envs=64, terrain_rows=4, terrain_cols=10, batch_size=64)


def pretrain_resources(task: str) -> PretrainResources:
  return RACE_PRETRAIN if task in ("race_mgdp", "LgDog1-Race-MGDP") else PretrainResources()
