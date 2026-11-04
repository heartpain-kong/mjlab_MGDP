"""Task catalog: the single source of truth for launcher metadata."""

from src.him_contract import HIM_INPUT_DIM

from .tasks import TaskSpec, capabilities

# 默认任务保持 HIM 标准地形；MGDP 深度感知通过显式别名 `mgdp` 使用
# （首次训练前先跑 `python scripts/pretrain_mgdp.py` 生成感知检查点）。
DEFAULT_TASK = "standard"

_TASK_LIST = (
  TaskSpec(
    alias="standard", id="LgDog1-Rough-Blind-HIM-Standard",
    label="HIM-PPO · 标准崎岖地形", algorithm="him", scene="rough",
    experiment="lgdog1_velocity_blind_him_standard_v1", obs=HIM_INPUT_DIM,
    critic=74, profile="him_standard", docs="doc/LGDOG_HIM_STANDARD.md",
    description="PPO + 三帧历史 + HIM 辅助估计；actor/critic 无机身地形扫描。续训要求任务和历史长度一致。",
  ),
  TaskSpec(
    alias="mgdp", id="LgDog1-Rough-MGDP",
    label="MGDP · 深度感知", algorithm="ppo", scene="rough_depth",
    experiment="lgdog1_velocity_mgdp_v2", obs=252, critic=261, profile="mgdp",
    docs="doc/MGDP_MJLAB.md", requires_pretrain=True,
    pretrain_command="python scripts/pretrain_mgdp.py --num-envs 256 --steps 4000 --iterations 3000",
    pretrain_checkpoint="logs/mgdp/perception/mgdp_perception.pt",
    description="MGDP 对比感知 + PPO：actor = 4×47 本体历史 + 64 维深度特征；训练前先跑 scripts/pretrain_mgdp.py。",
  ),
  TaskSpec(
    alias="race_mgdp", id="LgDog1-Race-MGDP",
    label="MGDP · 定向越障训练", algorithm="ppo", scene="race_depth",
    experiment="lgdog1_velocity_race_mgdp_v2", obs=252, critic=261, profile="mgdp",
    docs="doc/MGDP_PASSAGE.md", requires_pretrain=True, directed_passage=True,
    pretrain_command="python scripts/pretrain_mgdp.py --num-envs 256 --steps 4000 --iterations 3000",
    pretrain_checkpoint="logs/mgdp/perception/mgdp_perception.pt",
    description="低速从入口穿过障碍到出口；越界失败，各类通过率与跟踪达标后扩速。奖励关闭抬腿高度，启用 feet_air_time。",
  ),
  TaskSpec(
    alias="him_flat", id="LgDog1-Flat-Blind-HIM-Standard",
    label="HIM-PPO · 平地", algorithm="him", scene="flat",
    experiment="lgdog1_velocity_flat_him_standard_v1", obs=HIM_INPUT_DIM,
    critic=74, profile="flat", supports_curriculum=False,
    description="PPO + 三帧历史的平地训练。平地 checkpoint 不能直接续训崎岖或障碍场景。",
  ),
  TaskSpec(
    alias="race", id="LgDog1-Race-Blind-HIM",
    label="HIM-PPO · 2027 比赛场地", algorithm="him", scene="race",
    experiment="lgdog1_velocity_race_him_v1", obs=HIM_INPUT_DIM, critic=74,
    profile="him_standard", docs="doc/RACE_COURSE.md",
    description="按 2027 比赛规则尺寸拼的地形：梅花桩、单木桥、沟壑台阶、砖木坑、绕桩、0.3 m 高墙。观测/奖励同标准 HIM；真场地用 sim2sim --map race。",
  ),
  TaskSpec(
    alias="him", id="LgDog1-Rough-Blind-HIM",
    label="HIM-PPO · 温和地形（V7）", algorithm="him", scene="rough_gentle",
    experiment="lgdog1_velocity_blind_him_v1", obs=HIM_INPUT_DIM, critic=261,
    profile="him",
    description="PPO + 三帧HIM历史估计器，保留 V7/V7b 的 critic 地形扫描。",
  ),
  TaskSpec(
    alias="terrain", id="LgDog1-Rough-Blind-TerrainV4",
    label="TerrainV4 · 盲走", algorithm="ppo", scene="rough_terrain_v4",
    experiment="lgdog1_velocity_blind_terrain_v4", obs=47, critic=261,
    profile="terrain_v4",
    description="改进地形课程、相对抬脚奖励与接触终止。",
  ),
  TaskSpec(
    alias="gentle", id="LgDog1-Rough-Blind-Gentle",
    label="PPO · 温和地形盲走", algorithm="ppo", scene="rough_gentle",
    experiment="lgdog1_velocity_blind_gentle", obs=47, critic=74, profile="rough",
    description="普通 PPO，降低台阶、坡面与起伏难度。",
  ),
  TaskSpec(
    alias="blind", id="LgDog1-Rough-Blind",
    label="PPO · 标准地形盲走", algorithm="ppo", scene="rough",
    experiment="lgdog1_velocity_blind", obs=47, critic=74, profile="rough",
    description="普通 PPO，标准 Rough，无机身地形扫描。",
  ),
  TaskSpec(
    alias="flat", id="LgDog1-Flat",
    label="PPO · 平地", algorithm="ppo", scene="flat",
    experiment="lgdog1_velocity", obs=47, critic=74, profile="flat",
    shared=True, supports_curriculum=False,
    description="平地基线。与 Rough 共用日志目录，请手动匹配模型。",
  ),
  TaskSpec(
    alias="rough", id="LgDog1-Rough",
    label="PPO · 地形感知", algorithm="ppo", scene="rough_scan",
    experiment="lgdog1_velocity", obs=234, critic=261, profile="rough",
    shared=True,
    description="actor 含地形扫描。与 Flat 共用日志目录，请手动匹配模型。",
  ),
)

TASKS: dict[str, TaskSpec] = {spec.alias: spec for spec in _TASK_LIST}

ALGORITHMS = {
  "him": {"label": "HIM-PPO", "description": "PPO 主优化 + 三帧历史编码器和 HIM 辅助估计。"},
  "ppo": {"label": "PPO", "description": "同一 PPO 主优化流程，使用单帧观测并关闭 HIM 辅助分支。"},
}

SCENES = {
  "flat": {"label": "平地", "description": "无限平面，适合动作和速度跟踪预训练。"},
  "rough": {"label": "标准崎岖地形", "description": "课程地形，包含台阶、坡面、随机起伏和波浪。"},
  "race": {"label": "2027 比赛场地", "description": "按比赛规则尺寸拼的障碍地形：梅花桩、单木桥、沟壑台阶、砖木坑、绕桩、高墙。"},
  "rough_depth": {"label": "深度感知崎岖地形", "description": "标准崎岖地形 + 16×16 深度射线；actor 使用感知特征而不是高度扫描。"},
  "race_depth": {"label": "2027 比赛地形（深度）", "description": "比赛地形 + 限高横梁 + 轮胎环；深度相机可感知悬空障碍。"},
  "rough_gentle": {"label": "温和崎岖地形", "description": "降低台阶和坡面难度的过渡场景。"},
  "rough_scan": {"label": "带扫描崎岖地形", "description": "actor 使用地形扫描，不能与盲走模型直接互换。"},
  "rough_terrain_v4": {"label": "TerrainV4 崎岖地形", "description": "PPO TerrainV4 课程，独立于标准盲走任务。"},
}

SIM2SIM_MAPS = {
  "flat": {"label": "平地", "description": "无限平面；适合先验证站立和速度跟踪。", "geoms": 0, "spawn_y": 0.0},
  "stairs": {"label": "四组楼梯", "description": "5 / 10 / 15 / 20 cm 四条楼梯车道。", "geoms": 44, "spawn_y": 4.0},
  "slope": {"label": "10°坡面", "description": "单条 10° 坡面和顶部平台。", "geoms": 2, "spawn_y": 0.0},
  "obstacles": {"label": "高台障碍", "description": "10～60 cm 六组高台。", "geoms": 6, "spawn_y": 0.0},
  "mixed": {"label": "混合地形", "description": "楼梯、坡面、高台和散落障碍组合。", "geoms": 55, "spawn_y": 4.0},
  "lgdog1": {"label": "LgDog 测试图", "description": "部署测试地图，包含 191 个地形 geom。", "geoms": 191, "spawn_y": 4.0},
  "race": {"label": "2027 障碍赛", "description": "比赛场地：绕桩、限高框架、高墙、沟壑、砖木坑、轮胎阵、梅花桩、单木桥（14×6 m）。",
           "geoms": 984, "spawn_y": 0.0},
  "race_task": {"label": "2027 任务赛", "description": "任务赛内场：6×4 m、8 个物资箱、8 个存放区、4 个归位区、2 道挡车杆。",
                "geoms": 105, "spawn_y": 0.0},
}


def resolve_task(value=DEFAULT_TASK):
  for alias, task in TASKS.items():
    if value in (alias, task.id):
      return alias, task
  raise ValueError(f"未知任务 {value!r}，可选：{', '.join(TASKS)}")


def task_payload(alias: str) -> dict:
  """Serialised task entry for /api/state (keeps the flat shape + capabilities)."""
  spec = TASKS[alias]
  return {"alias": alias, **spec.as_dict(), "capabilities": capabilities(spec)}

