"""Lightweight launcher labels. Registry consistency is checked in tests."""

TASKS = {
  "standard": {"id": "LgDog1-Rough-Blind-HIM-Standard", "label": "HIM · 标准崎岖地形", "algorithm": "him", "scene": "rough", "experiment": "lgdog1_velocity_blind_him_standard_v1", "obs": 282, "critic": 74, "description": "标准 Rough + 六帧历史；actor/critic 无机身地形扫描。", "profile": "him_standard", "resume_sources": ("him_flat",)},
  "him_flat": {"id": "LgDog1-Flat-Blind-HIM-Standard", "label": "HIM · 平地预训练", "algorithm": "him", "scene": "flat", "experiment": "lgdog1_velocity_flat_him_standard_v1", "obs": 282, "critic": 74, "description": "与标准 HIM 相同的六帧历史接口，使用平地奖励预训练，支持无缝续训到 Rough。", "profile": "flat"},
  "him": {"id": "LgDog1-Rough-Blind-HIM", "label": "HIM · 温和地形（V7）", "algorithm": "him", "scene": "rough_gentle", "experiment": "lgdog1_velocity_blind_him_v1", "obs": 282, "critic": 261, "description": "原 V7/V7b 配置，保留 critic 地形扫描。", "profile": "him"},
  "terrain": {"id": "LgDog1-Rough-Blind-TerrainV4", "label": "TerrainV4 · 盲走", "algorithm": "ppo", "scene": "rough", "experiment": "lgdog1_velocity_blind_terrain_v4", "obs": 47, "critic": 261, "description": "改进地形课程、相对抬脚奖励与接触终止。", "profile": "terrain_v4"},
  "gentle": {"id": "LgDog1-Rough-Blind-Gentle", "label": "PPO · 温和地形盲走", "algorithm": "ppo", "scene": "rough_gentle", "experiment": "lgdog1_velocity_blind_gentle", "obs": 47, "critic": 74, "description": "普通 PPO，降低台阶、坡面与起伏难度。", "profile": "rough"},
  "blind": {"id": "LgDog1-Rough-Blind", "label": "PPO · 标准地形盲走", "algorithm": "ppo", "scene": "rough", "experiment": "lgdog1_velocity_blind", "obs": 47, "critic": 74, "description": "普通 PPO，标准 Rough，无机身地形扫描。", "profile": "rough"},
  "flat": {"id": "LgDog1-Flat", "label": "PPO · 平地", "algorithm": "ppo", "scene": "flat", "experiment": "lgdog1_velocity", "obs": 47, "critic": 74, "description": "平地基线。与 Rough 共用日志目录，请手动匹配模型。", "profile": "flat", "shared": True},
  "rough": {"id": "LgDog1-Rough", "label": "PPO · 地形感知", "algorithm": "ppo", "scene": "rough_scan", "experiment": "lgdog1_velocity", "obs": 234, "critic": 261, "description": "actor 含地形扫描。与 Flat 共用日志目录，请手动匹配模型。", "profile": "rough", "shared": True},
}

ALGORITHMS = {
  "him": {"label": "HIM", "description": "六帧历史本体感觉，适合盲地形续训。"},
  "ppo": {"label": "PPO", "description": "普通 PPO，可选择平地、温和地形或标准地形。"},
}

SCENES = {
  "flat": {"label": "平地", "description": "无限平面，适合动作和速度跟踪预训练。"},
  "rough": {"label": "标准崎岖地形", "description": "课程地形，包含台阶、坡面、随机起伏和波浪。"},
  "rough_gentle": {"label": "温和崎岖地形", "description": "降低台阶和坡面难度的过渡场景。"},
  "rough_scan": {"label": "带扫描崎岖地形", "description": "actor 使用地形扫描，不能与盲走模型直接互换。"},
}

SIM2SIM_MAPS = {
  "flat": {"label": "平地", "description": "无限平面；适合先验证站立和速度跟踪。", "geoms": 0, "spawn_y": 0.0},
  "stairs": {"label": "四组楼梯", "description": "5 / 10 / 15 / 20 cm 四条楼梯车道。", "geoms": 44, "spawn_y": 4.0},
  "slope": {"label": "10°坡面", "description": "单条 10° 坡面和顶部平台。", "geoms": 2, "spawn_y": 0.0},
  "obstacles": {"label": "高台障碍", "description": "10～60 cm 六组高台。", "geoms": 6, "spawn_y": 0.0},
  "mixed": {"label": "混合地形", "description": "楼梯、坡面、高台和散落障碍组合。", "geoms": 55, "spawn_y": 4.0},
  "lgdog1": {"label": "LgDog 测试图", "description": "部署测试地图，包含 191 个地形 geom。", "geoms": 191, "spawn_y": 4.0},
}


def resolve_task(value="standard"):
  for alias, task in TASKS.items():
    if value in (alias, task["id"]):
      return alias, task
  raise ValueError(f"未知任务 {value!r}，可选：{', '.join(TASKS)}")
