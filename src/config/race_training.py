"""Compact race PPO presets; all ten obstacle types remain present."""

RACE_TRAIN_PRESETS = {
  "smoke": {"envs": 64, "iterations": 2, "terrain_num_rows": 2, "terrain_num_cols": 10},
  "full": {"envs": 256, "iterations": 16000, "terrain_num_rows": 4, "terrain_num_cols": 10},
}
