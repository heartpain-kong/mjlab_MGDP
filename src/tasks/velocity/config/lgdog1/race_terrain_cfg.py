"""2027 比赛场地的训练地形。

场地本身（`src/assets/terrains/race_obstacle_2027.xml`）是**任意 MJCF 刚体**，
而 mjlab 的训练地形只能是程序化生成的 box / heightfield 拼块，所以这里按
《2027 年足式机器人挑战赛》规则里的尺寸，用 mjlab 自带的拼块类型复刻比赛里
**能落到地面高度场**的那几类障碍：

| 比赛障碍 | 规则尺寸 | 这里的拼块 |
|---|---|---|
| 梅花桩 `stepping_stones` | 0.2×0.2×0.1，桩间净距 0.2 / 0.4，两端 0.8×0.8 平台 | `BoxSteppingStonesTerrainCfg`（完全对应） |
| 单木桥 `balance_beam` | 3×0.4×0.1 长条 | `BoxNarrowBeamsTerrainCfg`（宽 0.4） |
| 沟壑组合 `gap_assembly` | 0.8×0.8 平台、外露台阶高 0.12 / 深 0.32、20° 坡 | `BoxOpenStairsTerrainCfg`（0.10~0.14 台阶） |
| 砖木坑 `brick_bark_pit` | L 形 2×2，内部碎砖 10~30 mm 起伏 | `BoxRandomSpreadTerrainCfg`（0.1~0.2 m 碎块，0~6 cm 高） |
| 绕桩 `slalom` | 桩径 0.1、高 0.1、中心距 0.9 | `BoxRandomSpreadTerrainCfg`（细高柱近似） |
| 高墙 `high_wall` | 厚 0.08、高 0.3 | `BoxRandomSpreadTerrainCfg`（薄板） |
| 轮胎阵 `tire_array` / 限高框架 `clearance_frame` | 轮胎、头顶横梁 | **不做**：不是地面高度场，只在 sim2sim 真场地上练 |
| 启动区 | 0.8×0.8 平地 | `BoxFlatTerrainCfg` |

用法：`LgDog1-Race-Blind-HIM` 任务（`python lgdog.py train race --envs 4096`）。
仿真里走真场地：`python scripts/sim2sim.py LgDog1-Race-Blind-HIM --map race`。
"""

from copy import deepcopy

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxNarrowBeamsTerrainCfg,
  BoxOpenStairsTerrainCfg,
  BoxRandomSpreadTerrainCfg,
  BoxSteppingStonesTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGeneratorCfg,
)

# 每个拼块占的比例（curriculum 模式按列分配，比例决定该障碍占几列）
RACE_PROPORTIONS = {
  "start_flat": 0.10,
  "stepping_stones": 0.18,
  "balance_beam": 0.15,
  "gap_steps": 0.15,
  "brick_pit": 0.15,
  "slalom_posts": 0.12,
  "high_wall": 0.10,
  "rough": 0.05,
}


def lgdog1_race_terrain_cfg() -> TerrainGeneratorCfg:
  """按比赛场地尺寸拼出来的训练地形（8×8 m 一块，10 行 × 12 列）。"""
  cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=12,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      # 启动区：0.8×0.8 平地（规则没给尺寸，这里给足落脚面积）
      "start_flat": BoxFlatTerrainCfg(proportion=RACE_PROPORTIONS["start_flat"]),
      # 梅花桩：方块 0.2×0.2×0.1，桩间 0.2~0.4（difficulty 越大越疏/越高）
      "stepping_stones": BoxSteppingStonesTerrainCfg(
        proportion=RACE_PROPORTIONS["stepping_stones"],
        stone_size_range=(0.2, 0.2),
        stone_distance_range=(0.2, 0.4),
        stone_height=0.1,
        stone_height_variation=0.0,
        stone_size_variation=0.0,
        displacement_range=0.0,
        floor_depth=1.5,
        platform_width=0.8,
        border_width=0.25,
      ),
      # 单木桥：宽 0.4、高 0.1，三条并排（沿 x 方向下坡/上坡都能走）
      "balance_beam": BoxNarrowBeamsTerrainCfg(
        proportion=RACE_PROPORTIONS["balance_beam"],
        num_beams=3,
        beam_width_range=(0.4, 0.4),
        beam_height=0.1,
        spacing=0.9,
        platform_width=0.8,
        border_width=0.25,
        floor_depth=1.5,
      ),
      # 沟壑组合：台阶 0.12 m 高、0.32 m 深（规则值），两端平台 0.8
      "gap_steps": BoxOpenStairsTerrainCfg(
        proportion=RACE_PROPORTIONS["gap_steps"],
        step_height_range=(0.10, 0.14),
        step_width_range=(0.30, 0.36),
        platform_width=0.8,
        border_width=0.25,
        step_thickness=0.05,
        inverted=False,
      ),
      # 砖木坑：0.1~0.2 m 碎块、0~6 cm 高，脚下永远踩不平
      "brick_pit": BoxRandomSpreadTerrainCfg(
        proportion=RACE_PROPORTIONS["brick_pit"],
        num_boxes=36,
        box_width_range=(0.10, 0.20),
        box_length_range=(0.10, 0.20),
        box_height_range=(0.0, 0.06),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=0.8,
        border_width=0.25,
      ),
      # 绕桩：细高柱（近似直径 0.1 m 的桩），要侧着绕过去
      "slalom_posts": BoxRandomSpreadTerrainCfg(
        proportion=RACE_PROPORTIONS["slalom_posts"],
        num_boxes=7,
        box_width_range=(0.05, 0.06),
        box_length_range=(0.05, 0.06),
        box_height_range=(0.45, 0.55),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=0.8,
        border_width=0.25,
      ),
      # 高墙：0.3 m 高、0.08 m 厚的薄板，需要跨过去
      "high_wall": BoxRandomSpreadTerrainCfg(
        proportion=RACE_PROPORTIONS["high_wall"],
        num_boxes=3,
        box_width_range=(0.5, 0.8),
        box_length_range=(0.04, 0.04),
        box_height_range=(0.28, 0.30),
        box_yaw_range=(0.0, 0.0),
        add_floor=True,
        platform_width=0.8,
        border_width=0.25,
      ),
      # 场地之间的过渡：小起伏，别让策略只认"规整障碍"
      "rough": HfRandomUniformTerrainCfg(
        proportion=RACE_PROPORTIONS["rough"],
        noise_range=(0.02, 0.06),
        noise_step=0.02,
        border_width=0.25,
      ),
    },
  )
  return cfg


def race_terrain_summary() -> dict:
  """给测试和文档用：返回 {拼块: 比例}。"""
  return dict(RACE_PROPORTIONS)


def copy_race_terrain_cfg(cfg: TerrainGeneratorCfg) -> TerrainGeneratorCfg:
  """深拷贝一份，避免改到别的任务共享的子地形对象。"""
  return deepcopy(cfg)
