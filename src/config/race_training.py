"""Race PPO presets retaining the requested twenty terrain columns."""

MIN_RACE_TERRAIN_COLS = 10

RACE_TRAIN_PRESETS = {
  "smoke": {"envs": 64, "iterations": 2, "terrain_num_rows": 2, "terrain_num_cols": 20, "ppo_mini_batches": 4},
  "full": {"envs": 2048, "iterations": 16000, "terrain_num_rows": 8, "terrain_num_cols": 20, "ppo_mini_batches": 4},
  "probe3072": {"envs": 3072, "iterations": 2, "terrain_num_rows": 8, "terrain_num_cols": 20, "ppo_mini_batches": 8},
}
