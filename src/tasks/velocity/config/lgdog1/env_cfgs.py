"""LgDog1 机器人、观测、地形、指令与终止配置。

奖励权重、参数和函数绑定统一在同目录 reward_cfg.py。
机器人增益、力矩限幅、站姿和动作缩放在 src/config/lgdog1.py。
"""

from typing import Literal

from src.assets.robots import LGDOG1_ACTION_SCALE, get_lgdog1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from .reward_cfg import make_lgdog_rewards
from .race_terrain_cfg import lgdog1_race_terrain_cfg
from .terrain_cfg import lgdog1_rough_terrain_cfg

TerrainType = Literal["rough", "obstacles"]

# ---- LgDog1 命名约定（必须与 lgdog1.xml 一致）----
BASE_BODY = "trunk"
FOOT_SITES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
# 足端接触体是小腿末端的球，名字挂在 calf 上（见 lgdog1.xml 头注释）。
# 序号必须在 `_collision` 之前，否则匹配不上 `.*_collision\d*$`。
FOOT_GEOMS = (
  "FR_calf1_collision",
  "FL_calf1_collision",
  "RR_calf1_collision",
  "RL_calf1_collision",
)
# 4 条腿的自碰撞体（用于非法接触终止）
_TRUNK_GEOM = "trunk_collision"


def lgdog1_rough_env_cfg(
  play: bool = False, use_height_scan: bool = True, gentle_terrain: bool = False,
  terrain: str = "rough",
) -> ManagerBasedRlEnvCfg:
  """LgDog1 崎岖地形配置。

  ``use_height_scan=False`` → 关掉 187 维高度扫描，只靠本体感觉。
    · 优点：实机【不需要】高程图/地形传感器（本机实机就没雷达）；训练也快。
    · 代价：看不见前方地形，只能靠脚底反馈（踩到才知道）。
    · 观测维度 234 → 47；⚠️ 与带扫描的 checkpoint 互不兼容。

  ``gentle_terrain=True`` → 更温和的地形（给【盲跑】用）。
    ★ 为什么需要：实测（2026-10-16）纯本体感觉策略在标准崎岖地形上每回合只走
    2.43 m（平地策略 5.93 m），够不到课程 4.0 m 的升级门槛 ⇒ 62% 回合被降级 ⇒
    地形等级常年趴在 level≈0.1；`illegal_contact=1.0` 说明回合几乎都是"腿磕到地形"
    结束的 —— 它是一路磕绊着挪。盲跑看不见台阶，标准难度对它太难，于是"保命挪动"
    成了局部最优。先给一套够得着的地形，让它有机会成功。
    改动：平地 0.20→0.40；台阶 0~10cm→0~5cm；坡度 0~1.0→0~0.5；
          随机起伏 0.02~0.10→0.02~0.06；波浪 0~0.2→0~0.1（占比之和仍为 1.0）。
  """
  cfg = make_velocity_env_cfg()

  # Keep LgDog on the native Unitree/mjlab rough terrain distribution, while
  # giving this project an isolated editable copy.  A change in terrain_cfg.py
  # cannot mutate ROUGH_TERRAINS_CFG or another robot's task config.
  # ``terrain="race"`` 换成按 2027 比赛场地尺寸拼出来的地形（见 race_terrain_cfg.py）。
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    if terrain == "race":
      cfg.scene.terrain.terrain_generator = lgdog1_race_terrain_cfg()
    else:
      cfg.scene.terrain.terrain_generator = lgdog1_rough_terrain_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_lgdog1_robot_cfg()}

  # 地形扫描挂在机身 trunk 上（默认是空的占位 name=""）。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = BASE_BODY

  # ★ use_height_scan=False → 拆掉地形扫描传感器和 187 维 height_scan 观测
  if not use_height_scan:
    cfg.scene.sensors = tuple(
      s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=FOOT_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # ★ A：盲跑版用更温和的地形（子地形配置是可变的 @dataclass，直接改字段）
  if gentle_terrain and terrain != "race":
    tg = cfg.scene.terrain.terrain_generator
    assert tg is not None
    # ⚠️ 必须先【拷贝】再改！`make_velocity_env_cfg()` 里是
    #    `replace(ROUGH_TERRAINS_CFG)` —— 浅拷贝，sub_terrains 字典和里面的
    #    cfg 对象是【所有任务共享】的。直接改字段会污染 LgDog1-Rough / -Blind
    #    甚至其他机器人（实测就是这么发现所有任务都变成温和地形的）。
    import copy

    st = {k: copy.copy(v) for k, v in tg.sub_terrains.items()}
    tg.sub_terrains = st
    st["flat"].proportion = 0.40  # 0.20 → 0.40
    st["pyramid_stairs"].proportion = 0.15  # 0.20 → 0.15
    st["pyramid_stairs"].step_height_range = (0.0, 0.05)  # 0~10cm → 0~5cm
    st["pyramid_stairs_inv"].proportion = 0.15
    st["pyramid_stairs_inv"].step_height_range = (0.0, 0.05)
    st["hf_pyramid_slope"].slope_range = (0.0, 0.5)  # 0~1.0 → 0~0.5
    st["hf_pyramid_slope_inv"].slope_range = (0.0, 0.5)
    st["random_rough"].proportion = 0.05  # 0.10 → 0.05
    st["random_rough"].noise_range = (0.02, 0.06)  # 0.02~0.10 → 0.02~0.06
    st["wave_terrain"].proportion = 0.05  # 0.10 → 0.05
    st["wave_terrain"].amplitude_range = (0.0, 0.1)  # 0~0.2 → 0~0.1

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LGDOG1_ACTION_SCALE  # 0.25，= legged_gym 的 action_scale

  cfg.viewer.body_name = BASE_BODY
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  ##
  # 指令范围：★ 按 Go2 对齐（原为 legged_gym 的 x±1.0 / y±1.0 / yaw±3.5）
  #   Go2: lin_vel_x (-1.0, 2.0) / lin_vel_y ±1.0 / ang_vel_z ±1.0 / heading ±π
  #   注意：yaw 从 ±3.5 收到 ±1.0，偏航任务难度降了不少；想找回原来的
  #   大偏航能力，把 ang_vel_z 改回 (-3.5, 3.5) 即可。
  ##
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  # ★ ① 治"原地转圈/站不住"：加大【零指令·站立】环境的比例
  #   mjlab 默认 0.05 —— 训练里只有 5% 的环境被要求"指令为 0 → 站住别动"，
  #   这是策略最稀缺的技能。play/sim2sim 一开就是零指令，等于考它最不熟的科目，
  #   于是表现成原地小碎步 + 转圈。提到 0.25 给它足量样本。
  #   （只改 LgDog1，不动 Go2/G1 共用的 velocity_env_cfg.py）
  twist_cmd.rel_standing_envs = 0.25
  # 指令重采样不再固定为 3 s。固定切换会在每次换指令时制造动作尖峰，
  # 3~5 s 仍能覆盖足够多的指令，同时给策略更长的稳定响应窗口。
  twist_cmd.resampling_time_range = (3.0, 5.0)

  ##
  # 观测噪声：取自用户 LgDog1_config.noise.noise_scales（noise_level = 1.0）
  #   dof_pos 0.01 / dof_vel 1.5 / ang_vel 0.2 / gravity 0.05
  # mjlab 默认正好是 dof_pos 0.01、dof_vel 1.5、ang_vel 0.2、gravity 0.05 → 无需改。
  ##

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
    FOOT_SITES
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  cfg.rewards = make_lgdog_rewards("rough", base_body=BASE_BODY, foot_sites=FOOT_SITES)

  # 用户配置 terminate_after_contacts_on = ['base'] → 机身触地即终止。
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    # ★ 基类 `velocity_env_cfg.py` 把 nconmax 限死在 35/世界（为训练省显存）。
    #   play 模式地形是 5×5 + border_width=10 的大块地形，初始接触数会远超 35
    #   （实测见过需要 159），于是 `mjwarp.put_data` 抛：
    #     ValueError: nconmax overflow (nconmax must be >= 159)
    #   地形生成器的 seed 默认 None（每次随机），所以这个是【间歇性】的 ——
    #   同一个 checkpoint 有时能起、有时报错，很难排查。
    #   play 只有几个环境，放开让 mjlab 用启发式值，显存无压力。
    #   注意：训练路径保持 35 不动，1024 envs 下 35×1024 已经吃掉不少显存。
    cfg.sim.nconmax = None

  return cfg


def lgdog1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create LgDog1 flat terrain velocity configuration."""
  cfg = lgdog1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  cfg.rewards = make_lgdog_rewards("flat", base_body=BASE_BODY, foot_sites=FOOT_SITES)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
