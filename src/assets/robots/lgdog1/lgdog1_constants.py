"""LgDog1 constants.

LgDog1 是用户自研四足（trunk 0.404 m / thigh 0.19 m + calf 0.245 m / 整机 ≈10.06 kg）。
名义控制和初始状态参数集中在 src.config.lgdog1；模型和碰撞定义保留在资产层。
"""

from pathlib import Path
from dataclasses import asdict

import mujoco

from src import SRC_PATH
from src.config import lgdog1 as params
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

LGDOG1_XML: Path = SRC_PATH / "assets" / "robots" / "lgdog1" / "xmls" / "lgdog1.xml"
assert LGDOG1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, LGDOG1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LGDOG1_XML))
  spec.assets = get_assets(spec.meshdir)
  body = spec.body("trunk")
  if not any(camera.name == params.FRONT_CAMERA_NAME for camera in body.cameras):
    body.add_camera(name=params.FRONT_CAMERA_NAME, pos=params.FRONT_CAMERA_POS,
                    quat=params.FRONT_CAMERA_QUAT, fovy=params.FRONT_CAMERA_FOVY)
  return spec


##
# Actuator config.
##

# ★ 2026-10-14：PD 增益与 armature 已按用户要求改成 Go2 的值（分关节风格：
#   hip/thigh 软、calf 硬）。**力矩上限没抄 Go2** —— Go2 是 23.5/45 Nm，
#   而 LgDog1 真机是 hip,thigh 10 Nm / calf 20 Nm（1:2 减速比），
#   抄过来策略会学到真机给不出的力矩，sim2real 直接废掉。
#   frictionloss 也保留 LgDog1 原值（Go2 是 0），它是这套模型的关节摩擦。
LGDOG1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  **asdict(params.HIP),
)
LGDOG1_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  **asdict(params.THIGH),
)
LGDOG1_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  **asdict(params.CALF),
)

##
# Keyframes.
##

# 出生高度 0.31 m ≈ 默认姿态（thigh 0 / calf 0.2）的被动站高 0.303 m + 7 mm 余量。
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=params.INITIAL_POSITION,
  joint_pos=dict(params.INITIAL_JOINT_POSITIONS),
  joint_vel={".*": params.INITIAL_JOINT_VELOCITY},
)

##
# Collision config.
##

# LgDog1 的足端接触体是小腿末端的球（r=0.012），名字带在 calf 上 —— 见 lgdog1.xml 头注释。
# ★ 序号在 `_collision` 之前（`FR_calf1_collision`，1 = 足端球、0 = 小腿胶囊）。
#   这个位置很关键：mjlab 用 `.*_collision\d*$` 匹配碰撞体，写反成 `FR_calf_collision_1`
#   就会匹配不上 → 足端 contype/conaffinity 被清 0 → 机器人穿地自由落体。
_foot_regex = r"^[FR][LR]_calf1_collision$"

# 只留足端碰撞，且足端之间不自碰。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# 全部碰撞体参与（counaffinity=0 → 机器人自身不互撞，与 sim2sim XML 的口径一致）。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision\d*$",),
  condim={_foot_regex: 3, r".*_collision\d*$": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

LGDOG1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LGDOG1_ACTUATOR_HIP,
    LGDOG1_ACTUATOR_THIGH,
    LGDOG1_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=params.SOFT_JOINT_POS_LIMIT_FACTOR,
)

# 动作缩放（= legged_gym control.action_scale，也是部署侧 config.yaml 的值）。
LGDOG1_ACTION_SCALE = params.ACTION_SCALE


def get_lgdog1_robot_cfg() -> EntityCfg:
  """Get a fresh LgDog1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=LGDOG1_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_lgdog1_robot_cfg())

  viewer.launch(robot.spec.compile())
