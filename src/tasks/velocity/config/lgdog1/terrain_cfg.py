"""Project-local copy of the Unitree/mjlab rough terrain configuration.

The returned object starts as a deep copy of mjlab's native
``ROUGH_TERRAINS_CFG``.  Edit the returned copy here when a LgDog experiment
needs a different terrain distribution; the installed mjlab package remains
untouched and other robots keep their original terrain settings.
"""

from copy import deepcopy

from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def lgdog1_rough_terrain_cfg():
  """Return the Unitree rough terrain config with the LgDog edit profile."""
  cfg = deepcopy(ROUGH_TERRAINS_CFG)
  # LgDog blind terrain profile: retain native proportions and generators,
  # while limiting the first training campaign to reachable stairs/slopes.
  cfg.num_rows = 10
  cfg.num_cols = 20
  cfg.sub_terrains["flat"].proportion = 0.20
  cfg.sub_terrains["pyramid_stairs"].proportion = 0.20
  cfg.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.15)
  cfg.sub_terrains["pyramid_stairs_inv"].proportion = 0.20
  cfg.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.15)
  cfg.sub_terrains["hf_pyramid_slope"].proportion = 0.10
  cfg.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.4)
  cfg.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.10
  cfg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)
  cfg.sub_terrains["random_rough"].proportion = 0.10
  cfg.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
  cfg.sub_terrains["wave_terrain"].proportion = 0.10
  cfg.sub_terrains["wave_terrain"].amplitude_range = (0.0, 0.20)
  cfg.sub_terrains["wave_terrain"].num_waves = 4
  return cfg


TERRAIN_TYPES = {
  "flat": "平地；无高度变化，用于基础行走和恢复策略。",
  "pyramid_stairs": "正向台阶；difficulty 增大时台阶高度增加。",
  "pyramid_stairs_inv": "反向台阶/下台阶；测试落差和下台阶冲击。",
  "hf_pyramid_slope": "正向坡面；difficulty 增大时坡度增加。",
  "hf_pyramid_slope_inv": "反向坡面/下坡。",
  "random_rough": "随机高度起伏；由 noise_range 和 noise_step 控制。",
  "wave_terrain": "波浪地形；由 amplitude_range 和 num_waves 控制。",
}
