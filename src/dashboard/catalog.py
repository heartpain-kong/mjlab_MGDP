"""Lightweight launcher labels. Registry consistency is checked in tests."""

TASKS = {
  "standard": {"id": "LgDog1-Rough-Blind-HIM-Standard", "label": "HIM · 标准崎岖地形", "experiment": "lgdog1_velocity_blind_him_standard_v1", "obs": 282, "critic": 74, "description": "标准 Rough + 六帧历史；actor/critic 无机身地形扫描。", "profile": "him_standard"},
  "him": {"id": "LgDog1-Rough-Blind-HIM", "label": "HIM · 温和地形（V7）", "experiment": "lgdog1_velocity_blind_him_v1", "obs": 282, "critic": 261, "description": "原 V7/V7b 配置，保留 critic 地形扫描。", "profile": "him"},
  "terrain": {"id": "LgDog1-Rough-Blind-TerrainV4", "label": "TerrainV4 · 盲走", "experiment": "lgdog1_velocity_blind_terrain_v4", "obs": 47, "critic": 261, "description": "改进地形课程、相对抬脚奖励与接触终止。", "profile": "terrain_v4"},
  "gentle": {"id": "LgDog1-Rough-Blind-Gentle", "label": "Gentle · 温和地形盲走", "experiment": "lgdog1_velocity_blind_gentle", "obs": 47, "critic": 74, "description": "普通 PPO，降低台阶、坡面与起伏难度。", "profile": "rough"},
  "blind": {"id": "LgDog1-Rough-Blind", "label": "Blind · 标准地形盲走", "experiment": "lgdog1_velocity_blind", "obs": 47, "critic": 74, "description": "普通 PPO，标准 Rough，无机身地形扫描。", "profile": "rough"},
  "flat": {"id": "LgDog1-Flat", "label": "Flat · 平地", "experiment": "lgdog1_velocity", "obs": 47, "critic": 74, "description": "平地基线。与 Rough 共用日志目录，请手动匹配模型。", "profile": "flat", "shared": True},
  "rough": {"id": "LgDog1-Rough", "label": "Rough · 地形感知", "experiment": "lgdog1_velocity", "obs": 234, "critic": 261, "description": "actor 含地形扫描。与 Flat 共用日志目录，请手动匹配模型。", "profile": "rough", "shared": True},
}


def resolve_task(value="standard"):
  for alias, task in TASKS.items():
    if value in (alias, task["id"]):
      return alias, task
  raise ValueError(f"未知任务 {value!r}，可选：{', '.join(TASKS)}")
