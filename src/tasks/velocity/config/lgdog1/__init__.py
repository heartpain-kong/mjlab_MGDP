from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  lgdog1_flat_env_cfg,
  lgdog1_rough_env_cfg,
)
from .rl_cfg import lgdog1_ppo_runner_cfg
from .blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg
from .him_env_cfg import lgdog1_him_env_cfg, lgdog1_him_standard_env_cfg, lgdog1_him_runner_cfg
from src.tasks.velocity.rl.him import HIMRunner

register_mjlab_task(
  task_id="LgDog1-Rough-Blind-HIM-Standard",
  env_cfg=lgdog1_him_standard_env_cfg(),
  play_env_cfg=lgdog1_him_standard_env_cfg(play=True),
  rl_cfg=lgdog1_him_runner_cfg("lgdog1_velocity_blind_him_standard_v1"),
  runner_cls=HIMRunner,
)

register_mjlab_task(
  task_id="LgDog1-Rough-Blind-HIM",
  env_cfg=lgdog1_him_env_cfg(),
  play_env_cfg=lgdog1_him_env_cfg(play=True),
  rl_cfg=lgdog1_him_runner_cfg(),
  runner_cls=HIMRunner,
)

register_mjlab_task(
  task_id="LgDog1-Rough-Blind-TerrainV4",
  env_cfg=lgdog1_blind_terrain_env_cfg(),
  play_env_cfg=lgdog1_blind_terrain_env_cfg(play=True),
  rl_cfg=lgdog1_ppo_runner_cfg(experiment_name="lgdog1_velocity_blind_terrain_v4"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="LgDog1-Rough",
  env_cfg=lgdog1_rough_env_cfg(),
  play_env_cfg=lgdog1_rough_env_cfg(play=True),
  rl_cfg=lgdog1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="LgDog1-Rough-Blind",
  env_cfg=lgdog1_rough_env_cfg(use_height_scan=False),
  play_env_cfg=lgdog1_rough_env_cfg(play=True, use_height_scan=False),
  rl_cfg=lgdog1_ppo_runner_cfg(experiment_name="lgdog1_velocity_blind"),
  runner_cls=VelocityOnPolicyRunner,
)

# ★ 盲跑 + 温和地形：纯本体感觉 + 够得着的地形难度
#   （平地 40%、台阶 ≤5cm、坡度 0~0.5、起伏/波浪减半）。
#   标准难度对盲跑太难（实测 62% 回合被降级、地形等级常年趴 level≈0.1），先用这套起步。
register_mjlab_task(
  task_id="LgDog1-Rough-Blind-Gentle",
  env_cfg=lgdog1_rough_env_cfg(use_height_scan=False, gentle_terrain=True),
  play_env_cfg=lgdog1_rough_env_cfg(
    play=True, use_height_scan=False, gentle_terrain=True
  ),
  rl_cfg=lgdog1_ppo_runner_cfg(experiment_name="lgdog1_velocity_blind_gentle"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="LgDog1-Flat",
  env_cfg=lgdog1_flat_env_cfg(),
  play_env_cfg=lgdog1_flat_env_cfg(play=True),
  rl_cfg=lgdog1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
