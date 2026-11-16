#!/usr/bin/env python3
"""sim2sim —— 把导出策略放进【普通 MuJoCo（CPU）】里跑，验证部署链路。

为什么要它
──────────
mjlab 训练本身就跑在 MuJoCo 上（mujoco_warp / GPU）。所以这里的 "sim2sim" 不是
"换个物理引擎"，而是 **换掉整条训练栈**：不经过 mjlab 的 manager 体系、不经过 rsl_rl、
不用 mujoco_warp，只用

    普通 mujoco.MjModel + onnxruntime + 一个 50 Hz 的 PD 控制环

这跟真机部署走的是**同一条路**：原始观测 → ONNX → 关节目标 → PD。
所以它能验证的东西正是实机会踩的坑：

  · ONNX 文件本身能不能被独立加载、接口对不对（PPO [1,47] / HIM默认 [1,141]（旧模型 [1,282]） → [1,12]）
  · 观测拼装是否与训练一致（这是 sim2real 最常见的翻车点）
  · 动作缩放 / 默认站姿 / 关节顺序 / 控制频率 / PD 增益是否配套
  · 策略在"没有域随机化、没有观测噪声"的干净环境里能不能稳稳站住走起来

它**不能**验证的：真机的电机延迟、通信抖动、传感器噪声、模型误差 —— 那些要靠实机。

用法
────
  # 站立（零指令）自检
  python scripts/sim2sim.py LgDog1-Flat --onnx=export/lgdog1_flat_4900/policy.onnx --duration 10

  # 给前进指令
  python scripts/sim2sim.py LgDog1-Flat --onnx=... --vx 0.8 --duration 20

  # 边走边看 + 键盘遥控
  python scripts/sim2sim.py LgDog1-Flat --onnx=... --viewer
      W/S 前后   A/D 转向   Q/E 横移（数字小键盘同效，需 NumLock）
      空格/R 复位（边沿触发）   Esc 退出
      手感：按住即走、松手即停；转向用 --speed-yaw（默认 1.0 = 训练范围上限）；
            有键盘时【朝向保持】默认开 —— 松手后朝向会被保持住，直行不画圈。

  # 不指定 --onnx 时，自动取该任务最新 run 里的 policy.onnx
  python scripts/sim2sim.py LgDog1-Flat

  # 和 mjlab 训练环境对拍观测（验证拼装一致性，最严格的检查）
  python scripts/sim2sim.py LgDog1-Flat --check-obs --onnx=...

  # 无窗口也能跑（键盘是全局钩子，不挑窗口焦点；只是看不见画面）
  python scripts/sim2sim.py LgDog1-Flat --vx 1.0 --duration 20

依赖：键鼠控制要 `pip install pynput`（已装在 unitree_rl_mjlab 环境里）。
      没有 pynput 也能跑，只是键盘不可用（会打印提示）。

键盘为什么不用 glfw（与 LgRobotDog 那边不同）
────────────────────────────────────────────
LgRobotDog 用的是主线程渲染的 mujoco-python-viewer，主线程能 glfw.get_current_context()。
mjlab 用的是 `mujoco.viewer.launch_passive`，**渲染跑在独立线程**，主线程拿不到 glfw 上下文
（实测直接 `glfwGetKey: Assertion 'window != NULL' failed` 崩掉）；而它的 `key_callback`
只传 keycode、没有松开事件，判断不了"按住"。所以这里改用 pynput 全局钩子。
"""

from __future__ import annotations

import argparse
from collections import deque
from itertools import count
import sys
from threading import Event
import time
from pathlib import Path

import mujoco
import numpy as np

# ── 工程内的地形资源 ───────────────────────────────────────────────────────
TERRAIN_DIR = Path(__file__).resolve().parent.parent / "src" / "assets" / "terrains"
# 2027 ROBOCON 比赛场地：由 scripts/import_race_course.py 从 27rc-dog-race 转成
# 「弧度 + quat + 无 default」的地理 XML，可以安全 attach 到机器人场景里。
COURSE_MAPS = {
  "race": "race_obstacle_2027.xml",
  "race_task": "race_task_2027.xml",
}
COURSE_STARTS = {
  # (x, y, yaw_deg)：场地里的两个启动区（建模说明 docs/modeling.md）
  "race": (0.0, 0.0, 0.0),
  "race_start2": (-5.95, -1.9, 0.0),
}

# ── 与训练强绑定的常量（改训练配置必须同步改这里）───────────────────────────
# 来源：src/tasks/velocity/config/lgdog1/env_cfgs.py + lgdog1_constants.py
CTRL_DT = 0.02  # 控制周期 = sim dt 0.005 × decimation 4  → 50 Hz
PHASE_PERIOD = 0.6  # 步态相位周期 [s]
CMD_THRESHOLD = 0.1  # 指令范数小于它 → phase 清零、stand_still 生效
ACTION_SCALE_OVERRIDE = None  # None = 从配置里读（LgDog1 = 0.25）
SPAWN_HEIGHT = 0.31
GOAL_JOINT = {  # 默认站姿（hip, thigh, calf）
  "hip": 0.0,
  "thigh": 0.0,
  "calf": 0.2,
}

# ── 朝向保持层（与训练的 heading 控制器同一律）─────────────────────────────
# mjlab 的速度指令是 heading_command=True：训练时 wz 不是直接采样的，而是
#     wz = clip(heading_control_stiffness × wrap(目标朝向 − 当前朝向), ang_vel_z 范围)
# 取值范围见 velocity_env_cfg.py（stiffness 0.5）与 lgdog1/env_cfgs.py（ang_vel_z ±1.0）。
# 策略观测里【没有绝对朝向】，所以开环下发 wz 会漂；这里复刻训练的律来保持朝向。
# （思路与 LgRobotDog/legged_gym/scripts/sim2sim.py 的朝向保持层一致）
HEADING_KP = 0.5  # = cfg.heading_control_stiffness
WZ_MAX = 1.0  # = cfg.commands["twist"].ranges.ang_vel_z 的绝对值上限


# ══════════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════════
_INTEGRATOR = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "rk4": mujoco.mjtIntegrator.mjINT_RK4,
  "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_CONE = {
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}
_JACOBIAN = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_SOLVER = {
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
}


def _box(spec, name, pos, size, rgba=(0.75, 0.72, 0.55, 1.0)):
  g = spec.worldbody.add_geom()
  g.name = name
  g.type = mujoco.mjtGeom.mjGEOM_BOX
  g.pos = list(pos)
  g.size = list(size)
  g.rgba = list(rgba)
  g.contype = 1
  g.conaffinity = 1
  g.friction = [1.0, 0.005, 0.0001]
  return g


def add_terrain(spec, kind: str) -> int:
  """往场景里加测试地形。返回加了多少个 geom。

  ── 为什么需要它 ──
  ★ 宇树自带的地图【只有一块平地】：`src/assets/robots/*/xmls/scene_*.xml` 的
    worldbody 里就一个 `<geom name="floor" type="plane">`，没有任何楼梯/斜坡/障碍
    （那些 scene 里的 box 全是机器人自己的碰撞体）。simulate/mujoco/model/ 里则是
    MuJoCo 官方示例模型，也没有地形。想测崎岖地形只能自己加。

  ── 关于 lgdog1 这张图 ──
    `lgdog1` 用的是从 LgRobotDog 的 LgDog1/xml/scene.xml 里抽出来的纯地形
    （191 个 geom → src/assets/terrains/lgdog1_testmap.xml），布局：
        楼梯车道 y=+4.0 / +1.0 / -2.0 / -5.0（每级 5 / 10 / 15 / 20 cm，各 10 级、
                   踏面 30 cm、宽 1.6 m，从 x≈1.0 起，顶部带平台）
        高台     x=-1.5、y=-1.4~+5.6 一列（10~60 cm）
        另有 5cm 台阶组等
    出生点在原点时面朝 +x，想上某条楼梯车道就配合 --spawn-y。
  """
  if kind in ("flat", "", None):
    return 0

  # ── 2027 ROBOCON 比赛场地（27rc-dog-race 转换而来，见 scripts/import_race_course.py）──
  #    race      障碍赛：14×6 m，绕桩 / 限高框架 / 高墙 / 沟壑 / 砖木坑 / 轮胎阵 / 梅花桩 / 单木桥
  #    race_task 任务赛：6×4 m 内场 + 8 个动态物资箱（贴图在 race_task_labels/）
  #    两张图都自带地板（arena_floor），下面 build_model 不会再补无限平面。
  if kind in COURSE_MAPS:
    path = TERRAIN_DIR / COURSE_MAPS[kind]
    if not path.exists():
      raise FileNotFoundError(
        f"找不到 {path}\n  重新转换： python scripts/import_race_course.py /path/to/27rc-dog-race-main"
      )
    return _attach(spec, path)

  if kind == "lgdog1":
    path = TERRAIN_DIR / "lgdog1_testmap.xml"
    if not path.exists():
      raise FileNotFoundError(
        f"找不到 {path}\n  重新生成： python scripts/sim2sim.py --help 见说明，或用 --map-file 指定别的"
      )
    return _attach(spec, path)

  n = 0
  # ── 四组楼梯（照 LgDog1 scene.xml 的尺寸）──
  if kind in ("stairs", "mixed"):
    for step_h, lane_y in ((0.05, 4.0), (0.10, 1.0), (0.15, -2.0), (0.20, -5.0)):
      for i in range(10):
        h = step_h * (i + 1)
        _box(
          spec,
          f"stair{int(step_h * 100):02d}_{i:02d}",
          (1.0 + 0.30 * i + 0.15, lane_y, h / 2),
          (0.15, 0.80, h / 2),
          rgba=(0.75, 0.72, 0.55, 1.0),
        )
        n += 1
      _box(
        spec,
        f"stair{int(step_h * 100):02d}_top",
        (1.0 + 3.0 + 0.30, lane_y, step_h * 10 / 2),
        (0.30, 0.80, step_h * 10 / 2),
        rgba=(1.0, 0.9, 0.4, 1.0),
      )
      n += 1

  # ── 10° 斜坡（y=+2.5 车道，从 x=1.0 起，长 2.5 m）──
  if kind in ("slope", "mixed"):
    ang = np.deg2rad(10.0)
    L, W = 2.5, 0.8
    lane_y = 2.5 if kind == "mixed" else 0.0
    g = _box(
      spec, "ramp", (1.0 + L / 2 * np.cos(ang), lane_y, L / 2 * np.sin(ang)), (L / 2, W, 0.02),
      rgba=(0.55, 0.70, 0.85, 1.0),
    )
    g.quat = [np.cos(-ang / 2), 0.0, np.sin(-ang / 2), 0.0]
    n += 1
    g2 = _box(spec, "ramp_top", (1.0 + L * np.cos(ang) + 0.4, lane_y, L * np.sin(ang) - 0.02 + 0.0),
              (0.4, W, 0.02), rgba=(0.55, 0.70, 0.85, 1.0))
    g2.pos = [1.0 + L * np.cos(ang) + 0.4, lane_y, L * np.sin(ang) - 0.02]
    g2.size = [0.4, W, 0.02]
    n += 1

  # ── 高台一列（10~60 cm，x=-1.5 侧）──
  if kind in ("obstacles", "mixed"):
    for i, h in enumerate((0.10, 0.20, 0.30, 0.40, 0.50, 0.60)):
      y = -1.4 + 1.4 * i
      _box(spec, f"plat{int(h * 100):02d}", (-1.5, y, h / 2), (0.8, 0.5, h / 2),
           rgba=(0.95, 0.55, 0.30, 1.0))
      n += 1

  # ── 散落小方块（练避障/跨步）──
  if kind == "mixed":
    for i, (x, y, h) in enumerate(((2.0, 6.5, 0.06), (3.0, 6.8, 0.10), (2.5, -6.5, 0.08))):
      _box(spec, f"blk{i}", (x, y, h / 2), (0.15, 0.15, h / 2), rgba=(0.7, 0.5, 0.5, 1.0))
      n += 1

  return n


def _attach(spec, path: Path) -> int:
  """把外部地形 XML（只含 worldbody geom）挂到场景里。

  MjSpec.attach 必须给挂载点（frame 或 site）——mjlab 的 Scene 里也是这么做的：
  `frame = self._spec.worldbody.add_frame(); self._spec.attach(terrain.spec, frame=frame)`
  """
  other = mujoco.MjSpec.from_file(str(path))
  n = len(list(other.worldbody.geoms))
  frame = spec.worldbody.add_frame()
  spec.attach(other, frame=frame)
  return n


def apply_mujoco_cfg(model: mujoco.MjModel, mc) -> None:
  """把任务的 `SimulationCfg.mujoco` 原样写到 MjModel.opt 上。

  ★★ 这一步不做的话 sim2sim 会假失败：
  训练跑在 mujoco_warp 上，用的是 `integrator=implicitfast / iterations=10 /
  ls_iterations=20`；而裸 MjSpec.compile() 出来的是 MuJoCo 默认的
  `euler / 100 / 50`。积分器都不一样，同一个策略的表现会天差地别
  （实测：不改的话机器人站着一路乱晃、倾角冲到 58°）。
  """
  opt = model.opt
  opt.timestep = mc.timestep
  opt.integrator = _INTEGRATOR[mc.integrator]
  opt.impratio = mc.impratio
  opt.cone = _CONE[mc.cone]
  opt.jacobian = _JACOBIAN[mc.jacobian]
  opt.solver = _SOLVER[mc.solver]
  opt.iterations = mc.iterations
  opt.tolerance = mc.tolerance
  opt.ls_iterations = mc.ls_iterations
  opt.ls_tolerance = mc.ls_tolerance
  opt.ccd_iterations = mc.ccd_iterations
  opt.gravity = mc.gravity
  opt.enableflags = 0
  if getattr(mc, "multiccd", False):
    opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD


def build_model(
  task_id: str, map_kind: str = "flat", map_file: str | None = None
) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
  """用任务自己的机器人配置建模型（保证与训练**同一套几何/执行器/碰撞**），再补地面与地形。"""
  import src.tasks  # noqa: F401  触发任务注册
  from mjlab.entity import Entity
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(task_id)
  robot_cfg = env_cfg.scene.entities["robot"]

  robot = Entity(robot_cfg)
  spec = robot.spec

  # 补一块无限地面（机器人的 xml 是"纯机器人"，没有场景）
  # ⚠ 比赛场地自带 arena_floor；再叠一块 z=0 的无限平面会出现共面接触，所以跳过。
  if map_kind not in COURSE_MAPS:
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.contype = 1
    floor.conaffinity = 1
    floor.friction = [1.0, 0.005, 0.0001]
    floor.rgba = [0.3, 0.35, 0.4, 1.0]

  # 地形（默认平地；宇树自带场景也只有平地，见 add_terrain 的说明）
  n_terrain = _attach(spec, Path(map_file)) if map_file else add_terrain(spec, map_kind)

  model = spec.compile()
  apply_mujoco_cfg(model, env_cfg.sim.mujoco)  # ★ 必须！否则动力学与训练不一致

  data = mujoco.MjData(model)

  # 收集关节/执行器信息
  jnames = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
  ]
  act_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  ]
  sensors = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): model.sensor_adr[i]
    for i in range(model.nsensor)
  }

  meta = {
    "env_cfg": env_cfg,
    "joint_names": jnames,
    "actuator_names": act_names,
    "sensors": sensors,
    "action_scale": (
      ACTION_SCALE_OVERRIDE
      if ACTION_SCALE_OVERRIDE is not None
      else float(env_cfg.actions["joint_pos"].scale)
    ),
    "default_qpos": np.array(
      [GOAL_JOINT["calf"] if "calf" in n else 0.0 for n in jnames], dtype=np.float64
    ),
    "nq_joint": len(jnames),
  }

  # ★★ 执行器顺序 ≠ 关节顺序，必须建映射！
  #   mjlab 的三个 BuiltinPositionActuatorCfg（hip / thigh / calf）会把执行器按
  #   「所有 hip → 所有 thigh → 所有 calf」排列，而关节顺序是「每条腿 hip,thigh,calf」：
  #       关节  : FR_hip FR_thigh FR_calf FL_hip FL_thigh FL_calf …
  #       执行器: FR_hip FL_hip   RR_hip   RL_hip FR_thigh FL_thigh …
  #   所以 data.ctrl（执行器序）不能直接写关节序的动作 —— 写错的话 12 个关节全乱，
  #   机器人立刻抽搐（实测就是这么发现这个坑的）。部署到真机同样是致命错误。
  perm = []
  for a in range(model.nu):
    jid = int(model.actuator_trnid[a, 0])  # 该执行器驱动的 joint id
    perm.append(jid - 1)  # joint 0 是 floating_base_joint，减 1 得 12 维关节序号
  meta["ctrl_joint_perm"] = np.array(perm, dtype=int)
  meta["joint_of_actuator"] = [jnames[i] for i in perm]
  meta["n_terrain_geom"] = n_terrain

  return model, data, meta


def reset(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  meta: dict,
  x: float = 0.0,
  y: float = 0.0,
  yaw_deg: float = 0.0,
):
  mujoco.mj_resetData(model, data)
  yaw = np.deg2rad(yaw_deg)
  data.qpos[0:3] = [x, y, SPAWN_HEIGHT]
  data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
  data.qpos[7 : 7 + meta["nq_joint"]] = meta["default_qpos"]
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════════════════════════════════
# 观测拼装 —— 必须与 mjlab 训练时逐位一致
# ══════════════════════════════════════════════════════════════════════════
def build_obs(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  meta: dict,
  cmd: np.ndarray,
  t_since_reset: float,
  last_action: np.ndarray,
) -> np.ndarray:
  """拼出 47 维观测。索引与 mjlab 的 actor 观测项顺序一一对应。"""
  n = meta["nq_joint"]

  # [0:3] base_ang_vel —— 陀螺仪（机体系 rad/s）
  ang_vel = data.sensordata[
    meta["sensors"]["imu_ang_vel"] : meta["sensors"]["imu_ang_vel"] + 3
  ].copy()

  # [3:6] projected_gravity —— gravity_vec_w = [0,0,-1]，转到机体系
  w, x, y, z = data.qpos[3:7]
  R = np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )
  proj_grav = R.T @ np.array([0.0, 0.0, -1.0])

  # [6:9] command
  command = cmd.astype(np.float64)

  # [9:11] phase —— 指令范数 < 0.1 时清零（与 mjlab 的 phase() 一致）
  ph = (t_since_reset % PHASE_PERIOD) / PHASE_PERIOD
  if np.linalg.norm(command) < CMD_THRESHOLD:
    phase = np.zeros(2)
  else:
    phase = np.array([np.sin(ph * 2 * np.pi), np.cos(ph * 2 * np.pi)])

  # [11:23] joint_pos - default
  q = data.qpos[7 : 7 + n]
  joint_pos_rel = q - meta["default_qpos"]

  # [23:35] joint_vel - 0
  dq = data.qvel[6 : 6 + n]

  return np.concatenate(
    [ang_vel, proj_grav, command, phase, joint_pos_rel, dq, last_action]
  ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# 策略
# ══════════════════════════════════════════════════════════════════════════
def load_onnx(path: Path):
  import onnxruntime as ort

  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  inp = sess.get_inputs()[0]
  out = sess.get_outputs()[0]
  return sess, inp.name, [d if isinstance(d, int) else None for d in inp.shape], out.shape


def find_latest_onnx(task_id: str, repo: Path) -> Path | None:
  """自动找该任务最新 run 里的 policy.onnx。"""
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_rl_cfg

  exp_name = load_rl_cfg(task_id).experiment_name
  root = repo / "logs" / "rsl_rl" / exp_name
  if not root.exists():
    return None
  cands = sorted(root.glob("*/policy.onnx"), key=lambda p: p.stat().st_mtime)
  return cands[-1] if cands else None


# ══════════════════════════════════════════════════════════════════════════
# 键盘（照 LgRobotDog/legged_gym/scripts/sim2sim.py 的手感）
# ══════════════════════════════════════════════════════════════════════════
def wrap_pi(a: float) -> float:
  return (a + np.pi) % (2 * np.pi) - np.pi


def heading_of_quat_wxyz(q: np.ndarray) -> float:
  """机身 x 轴在世界系的朝向角（mjlab 的 forward_vec_b = [1,0,0]）。"""
  w, x, y, z = q
  fx = 1 - 2 * (y * y + z * z)
  fy = 2 * (x * y + w * z)
  return float(np.arctan2(fy, fx))


class Keyboard:
  """全局键盘（pynput）——不挑窗口焦点，有【按下 + 松开】事件，所以能判断"按住"。

  键位（WASD 与数字小键盘等价，小键盘需 NumLock 开）：
      W / 小键盘8  前进        S / 小键盘2  后退
      A / 小键盘4  左转        D / 小键盘6  右转
      Q / 小键盘7  左移        E / 小键盘9  右移
      空格 / R     复位（边沿触发，按一下一次）      Esc  退出

  ★ 为什么不用 glfw：mjlab 用的是 `mujoco.viewer.launch_passive`，它的渲染跑在
  **独立线程**里，主线程 `glfw.get_current_context()` 拿到的是空
  （实测直接 `glfwGetKey: Assertion 'window != NULL' failed` 崩掉）；
  而 launch_passive 的 `key_callback` 只传 keycode、拿不到松开事件，也就没法判断"按住"。
  LgRobotDog 那边能用 glfw 是因为它用的是主线程渲染的 mujoco-python-viewer。
  """

  AXES = {
    "x": (("w", "8"), ("s", "2")),
    "y": (("q", "7"), ("e", "9")),
    "yaw": (("a", "4"), ("d", "6")),
  }

  def __init__(self):
    from pynput import keyboard  # 全局钩子，X11 下即插即用

    self._kb = keyboard
    self._down: set[str] = set()
    self._reset_pending = False
    self._exit_pending = False
    self._listener = keyboard.Listener(
      on_press=self._on_press, on_release=self._on_release
    )

  # ── 按键规范化：字母取 char，特殊键取 name ──
  def _name(self, key) -> str:
    if isinstance(key, self._kb.Key):
      return key.name
    if isinstance(key, self._kb.KeyCode):
      if key.char:
        return key.char.lower()
      return f"vk{key.vk}"
    return str(key)

  def _on_press(self, key):
    n = self._name(key)
    self._down.add(n)
    if n in ("space", "r"):
      self._reset_pending = True
    elif n == "esc":
      self._exit_pending = True

  def _on_release(self, key):
    self._down.discard(self._name(key))

  def start(self) -> None:
    self._listener.start()

  def stop(self) -> None:
    try:
      self._listener.stop()
    except Exception:
      pass

  def poll(self) -> tuple[float, float, float, bool, bool]:
    """返回 (x, y, yaw, reset, exit)；x/y/yaw ∈ {-1,0,+1}，reset/exit 是边沿。"""
    d = self._down

    def axis(name: str) -> float:
      pos, neg = self.AXES[name]
      return float(any(k in d for k in pos)) - float(any(k in d for k in neg))

    reset, self._reset_pending = self._reset_pending, False
    exit_, self._exit_pending = self._exit_pending, False
    return axis("x"), axis("y"), axis("yaw"), reset, exit_


# ══════════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════════
def run(args) -> int:
  repo = Path(__file__).resolve().parent.parent
  sys.path.insert(0, str(repo))

  onnx_path = Path(args.onnx) if args.onnx else find_latest_onnx(args.task, repo)
  if onnx_path is None or not onnx_path.exists():
    print(f"[x] 找不到 policy.onnx（--onnx 没给，自动查找也失败）")
    return 2

  model, data, meta = build_model(args.task, args.map, args.map_file)
  sess, iname, in_shape, out_shape = load_onnx(onnx_path)
  from src.policy_history import PolicyHistory, validate_policy_history

  # ★ 历史长度必须取自 **ONNX 自带元数据**，不能用 --task 反推。
  #   踩过的坑：拿 HIM 的 policy.onnx 配 --task LgDog1-Flat（其 history_length=1）
  #   → 报 "Policy shape [1, 282] does not match 1 x 47"。ONNX 里写了
  #   history_length / frame_dim / history_order / history_reset，就是为了让部署侧
  #   不需要训练配置；env_cfg 只作为元数据缺失时的兜底。
  onnx_meta = sess.get_modelmeta().custom_metadata_map
  mgdp = "perception" in meta["env_cfg"].observations
  if mgdp:
    import json
    # Match the actor's exported mapping, including legacy 0.25 policies.
    if ACTION_SCALE_OVERRIDE is None:
      saved_scale=onnx_meta.get('action_scale')
      if saved_scale is None:
        saved_scale=.25
      else:
        values=[float(x) for x in saved_scale.split(',')]
        if not values or any(not np.isfinite(x) or x<=0 or abs(x-values[0])>1e-8 for x in values):
          raise ValueError('MGDP ONNX action_scale must be a finite uniform positive scale')
        saved_scale=values[0]
      meta['action_scale']=float(saved_scale)
    from src.mgdp_checkpoint import restore_sensing_contract
    sensing = onnx_meta.get("mgdp_sensing")
    camera=onnx_meta.get('mgdp_camera')
    restore_sensing_contract(meta["env_cfg"],
      {'sensing':json.loads(sensing) if sensing else None,
       'camera':json.loads(camera) if camera else None} if sensing or camera else None)
    from src.mgdp_deployment import MGDPPolicyInput, file_digest
    architecture = onnx_meta.get("policy_architecture")
    if architecture == "lgdog_mgdp_v1":
      history_length = int(onnx_meta["history_length"])
      if (onnx_meta.get("history_order") != "term_major_oldest_to_newest"
          or onnx_meta.get("frame_dim") != "47"
          or not np.isclose(float(onnx_meta.get("control_dt", "nan")), CTRL_DT)):
        raise ValueError("MGDP policy history/control contract mismatch")
      checkpoint = Path(getattr(args, "perception", None) or
                        onnx_path.parent / onnx_meta["perception_checkpoint"])
      if file_digest(checkpoint) != onnx_meta["perception_sha256"]:
        raise ValueError("MGDP perception checkpoint does not match the exported policy")
    else:
      # Older exports mislabeled MGDP as one-frame PPO. Their encoder cannot
      # be recovered from the actor; require the user to name its original file.
      if not getattr(args, "perception", None):
        raise ValueError("Legacy MGDP policy requires --perception <original perception.pt>")
      checkpoint = Path(args.perception)
      history_length = meta["env_cfg"].observations["actor"].history_length
    sensor_cfg = next(s for s in meta["env_cfg"].scene.sensors if s.name == "depth_rays")
    history = MGDPPolicyInput(model, sensor_cfg, checkpoint, history_length,
                             onnx_meta.get("mgdp_inference_precision", "fp32"))
    expected = history_length * 47 + history.module.cfg.gru_hidden
    if len(in_shape) != 2 or in_shape[1] != expected:
      raise ValueError(f"MGDP policy shape {in_shape} does not match {expected}")
    if architecture == "lgdog_mgdp_v1" and int(onnx_meta["perception_dim"]) != history.module.cfg.gru_hidden:
      raise ValueError("MGDP policy perception width mismatch")
  else:
    if "history_length" in onnx_meta:
      history_length = int(onnx_meta["history_length"])
      frame_dim = int(onnx_meta.get("frame_dim", 47))
    else:
      history_length = meta["env_cfg"].observations["actor"].history_length or 1
      frame_dim = 47
    validate_policy_history(in_shape, onnx_meta, history_length, frame_dim=frame_dim,
                            control_dt=CTRL_DT)
    history = PolicyHistory(history_length, frame_dim)

  print("=" * 74)
  print(f"  sim2sim —— 普通 MuJoCo(CPU) + onnxruntime，完全绕过训练栈")
  print("=" * 74)
  print(f"  任务        {args.task}")
  print(f"  ONNX        {onnx_path}")
  print(f"  接口        in {in_shape} → out {out_shape}")
  print(f"  关节({meta['nq_joint']})   {meta['joint_names']}")
  print(f"  动作缩放    {meta['action_scale']}")
  print(f"  动作裁剪    ±{args.action_clip:g}（同时统计裁剪前原始动作）")
  print(f"  控制频率    {1.0 / CTRL_DT:.0f} Hz（物理 {model.opt.timestep * 1000:.0f} ms × 4）")
  print(f"  默认站姿    {np.round(meta['default_qpos'], 3).tolist()}")
  print(f"  执行器顺序  {meta['joint_of_actuator']}")
  print(f"  ctrl↔关节映射 {meta['ctrl_joint_perm'].tolist()}  ← 执行器序 → 关节序")
  print(f"  地形        {args.map if not args.map_file else args.map_file}"
        f"（{meta['n_terrain_geom']} 个地形 geom）")
  print(f"  出生点      x={args.spawn_x}  y={args.spawn_y}  yaw={args.spawn_yaw}°")
  print(f"  指令        vx={args.vx}  vy={args.vy}  wz={args.wz}")
  print(f"  查看器      {'开（会弹 MuJoCo 窗口）' if args.viewer else '关 —— 想看画面要加 --viewer'}")
  print("=" * 74)

  # ★ 测地形却没给指令 = 什么都没测到（机器人原地站着，永远碰不到地形），
  #   但最后还是会打印"✅ 通过"，非常误导。这里提前说清楚。
  if meta["n_terrain_geom"] > 0 and args.vx == 0.0 and args.vy == 0.0 and args.wz == 0.0:
    print()
    print("  ⚠️  选/ 了地形但指令是 0 —— 机器人会原地站着，碰不到任何台阶。")
    print("      想测地形请给指令，例如：  --vx 0.6        （再用 --spawn-y 对准某条车道）")
    print("      下面的『通过』只说明它能站住，不代表它能过地形。")
    print()

  if args.check_obs:
    return check_obs(model, data, meta, args)

  cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float64)
  last_action = np.zeros(meta["nq_joint"], dtype=np.float32)

  reset(model, data, meta, args.spawn_x, args.spawn_y, args.spawn_yaw)
  episode_step = 0

  # Interactive viewing has no timer. Keep a bounded default for batch runs.
  duration = args.duration if args.duration is not None else (None if args.viewer else 20.)
  n_ctrl = int(duration / CTRL_DT) if duration is not None else None
  decim = int(round(CTRL_DT / model.opt.timestep))
  exit_requested = Event()

  viewer = None
  inset = None
  if args.viewer:
    from mujoco import viewer as mj_viewer  # 别写 import mujoco.viewer，会把模块级 mujoco 遮蔽掉

    from src.camera_inset import CameraInset, KEY_F8
    from glfw import KEY_ESCAPE
    if getattr(args, "camera_inset", True):
      inset = CameraInset(model, args.camera_inset_size, args.camera_inset_fps, args.camera_inset_corner)
    def viewer_key(key):
      if key == KEY_ESCAPE:
        exit_requested.set()
      elif key == KEY_F8 and inset is not None:
        inset.toggle()
    viewer = mj_viewer.launch_passive(model, data, key_callback=viewer_key)
    if inset is not None:
      print("  相机小窗    前头顶水平朝前，F8 开关（RGB，非策略深度图）")
  print("  运行方式    持续回放，关闭窗口 / Esc / Ctrl+C 结束" if n_ctrl is None
        else f"  运行方式    定时验证 {duration:g} 秒")

  # 键盘（pynput 全局钩子，不依赖查看器窗口；没有查看器也能遥控，只是看不见）
  kb = None
  if not args.no_keyboard:
    try:
      kb = Keyboard()
      kb.start()
      print(
        "  键盘        W/S 前后   A/D 转向   Q/E 横移（数字小键盘同效，需 NumLock）\n"
        "              空格/R 复位   Esc 退出\n"
        "              默认速度    --speed-x %.1f  --speed-y %.1f  --speed-yaw %.1f"
        % (args.speed_x, args.speed_y, args.speed_yaw)
      )
    except Exception as e:  # pynput 没装 / 没有 X11 权限
      print(f"  键盘        ⚠️ 不可用（{type(e).__name__}: {e}）")
      print("              装了就能用： pip install pynput")
      kb = None

  hdg_target: float | None = None  # 朝向保持层的目标（None = 用当前朝向初始化）
  hdg_hold = args.heading_hold
  if args.heading_hold is None:
    hdg_hold = kb is not None  # 有键盘时默认开（否则直行会画圈）

  # Only retain the most recent 100 simulation seconds during an open-ended
  # session. Whole-session aggregates stay accurate without growing memory.
  log = deque(maxlen=5000) if n_ctrl is None else []
  n = 0
  totals = np.zeros(8)
  first_sample = None
  min_z, max_tilt, velocity_error_sum = float("inf"), 0., 0.
  fell, interrupted = False, False
  stop_reason = "达到指定时长"
  raw_action_abs_max = 0.0
  raw_action_abs_sum = 0.0
  raw_action_saturated = 0
  raw_action_samples = 0
  t0 = time.perf_counter()
  n_reset = 0
  try:
    for step in count():
      if n_ctrl is not None and n >= n_ctrl:
        break
      if exit_requested.is_set() or (viewer is not None and not viewer.is_running()):
        interrupted, stop_reason = True, "关闭窗口或按 Esc"
        break
      step_started = time.perf_counter()
      cur_hdg = heading_of_quat_wxyz(data.qpos[3:7])
      turning = False

      # ── 输入 ──────────────────────────────────────────────────────────
      if kb is not None:
        kx, ky, kyaw, do_reset, do_exit = kb.poll()
        if do_exit:
          print("\n  [Esc] 退出")
          interrupted, stop_reason = True, "按 Esc"
          break
        if do_reset:
          reset(model, data, meta, args.spawn_x, args.spawn_y, args.spawn_yaw)
          last_action[:] = 0.0
          hdg_target = None
          n_reset += 1
          history.reset()
          episode_step = 0
          if viewer is not None:
            viewer.sync()
          continue

        # x/y：按键按下就覆盖命令行给的固定速度
        cmd[0] = kx * args.speed_x if kx != 0 else args.vx
        cmd[1] = ky * args.speed_y if ky != 0 else args.vy
        # yaw：有转向按键 → 直接透传该速率，并把目标同步到当前朝向（松手后从这里保持）
        if kyaw != 0:
          cmd[2] = kyaw * args.speed_yaw
          hdg_target = cur_hdg
          turning = True

      # yaw 的兜底（键盘没在转向，或干脆没键盘）：
      #   · 命令行给了固定 wz → 直接下发（开环）
      #   · 否则若开了朝向保持 → 复刻训练的 heading 控制律
      if not turning:
        if args.wz != 0:
          cmd[2] = args.wz
        elif hdg_hold:
          if hdg_target is None:
            hdg_target = cur_hdg
          err = wrap_pi(hdg_target - cur_hdg)
          cmd[2] = float(np.clip(HEADING_KP * err, -WZ_MAX, WZ_MAX))
        else:
          cmd[2] = 0.0

      # ── 策略 ──────────────────────────────────────────────────────────
      # Append exactly once, after updating commands. Reset also restarts phase.
      # Like mjlab, refresh derived/sensor quantities after the final mj_step.
      mujoco.mj_forward(model, data)
      frame = build_obs(model, data, meta, cmd, episode_step * CTRL_DT, last_action)
      obs = history.push(frame, model, data) if mgdp else history.push(frame)
      raw_action = sess.run(None, {iname: obs[None, :]})[0][0].astype(np.float64)
      raw_abs = np.abs(raw_action)
      raw_action_abs_max = max(raw_action_abs_max, float(raw_abs.max()))
      raw_action_abs_sum += float(raw_abs.mean())
      raw_action_saturated += int(np.count_nonzero(raw_abs >= args.action_clip))
      raw_action_samples += raw_action.size
      action = np.clip(raw_action, -args.action_clip, args.action_clip)

      # 关节序目标 → 执行器序 ctrl（见 build_model 里的 perm 说明）
      target_joint = meta["default_qpos"] + meta["action_scale"] * action
      data.ctrl[:] = target_joint[meta["ctrl_joint_perm"]]

      for _ in range(decim):
        mujoco.mj_step(model, data)

      last_action = action.astype(np.float32)
      episode_step += 1

      # 量测
      pos = data.qpos[0:3].copy()
      w, x, y, z = data.qpos[3:7]
      R = np.array(
        [
          [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
      )
      z_axis = R[:, 2]
      tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1, 1)))
      vel_b = R.T @ data.qvel[0:3]
      sample = np.array([pos[2], tilt, *vel_b, float(np.abs(data.qvel[0:3]).max()), cur_hdg, pos[0]])
      log.append(sample)
      if first_sample is None:
        first_sample = sample.copy()
      n += 1
      totals += sample
      min_z, max_tilt = min(min_z, pos[2]), max(max_tilt, tilt)
      velocity_error_sum += abs(vel_b[0] - cmd[0])

      if viewer is not None:
        mujoco.mj_forward(model, data)
        viewer.sync()
        if inset is not None:
          inset.update(viewer, data)
        # Account for inference/rendering time instead of adding a fixed delay.
        time.sleep(max(0.0, CTRL_DT - (time.perf_counter() - step_started)))
      elif step % max(1, n_ctrl // 10) == 0:
        print(
          f"  t={step * CTRL_DT:6.2f}s  z={pos[2]:.3f}  tilt={tilt:5.1f}°  "
          f"vx={vel_b[0]:+.3f} (指令 {cmd[0]:+.2f})  |v|={np.linalg.norm(data.qvel[0:3]):.3f}"
        )

      # 摔了就停
      if tilt > 70.0 or pos[2] < 0.08:
        print(f"\n  ✗ 第 {step} 步摔倒：t={step * CTRL_DT:.2f}s  z={pos[2]:.3f}  tilt={tilt:.1f}°")
        fell, stop_reason = True, "摔倒"
        break

  except KeyboardInterrupt:
    interrupted, stop_reason = True, "按 Ctrl+C"
  finally:
    if inset is not None:
      inset.close()
    if kb is not None:
      kb.stop()
    if viewer is not None:
      viewer.close()

  # ── 汇总 ──
  log = np.asarray(log)
  survived = n * CTRL_DT
  wall = max(time.perf_counter() - t0, 1e-9)

  print("\n" + "=" * 74)
  print("  结果")
  print("=" * 74)
  print(f"  结束原因    {stop_reason}")
  if not n:
    print("  运行步数    0（尚未推进仿真）")
    return 0 if interrupted else 1
  mean = totals / n
  if n_ctrl is None:
    print(f"  仿真累计    {survived:.2f}s  ({n} 个控制步，无时长限制)")
  else:
    print(f"  存活        {survived:.2f}s / {duration:.2f}s  ({n}/{n_ctrl} 个控制步)")
  print(f"  机身高度    z 均值 {mean[0]:.4f} m  (min {min_z:.4f})")
  print(
    f"  最终位置    x={log[-1, 7]:+.2f} m  (直线距离 {abs(log[-1, 7] - first_sample[7]):.2f} m)"
    f"  ← 测地形时看这个：被挡住就不会涨"
  )
  print(f"  倾角        tilt 均值 {mean[1]:.2f}°  max {max_tilt:.2f}°")
  print(f"  实测速度    vx 均值 {mean[2]:+.3f} m/s   指令 {cmd[0]:+.3f}")
  print(f"              vy 均值 {mean[3]:+.3f} m/s   指令 {cmd[1]:+.3f}")
  print(f"              vz 均值 {mean[4]:+.3f} m/s")
  print(f"  速度跟踪误差 |vx-指令| 均值 {velocity_error_sum / n:.3f} m/s")
  hdg0, hdgN = float(first_sample[6]), float(log[-1, 6])
  net = np.degrees(wrap_pi(hdgN - hdg0))
  # 末段（后 25% 时间）的朝向漂移 —— 转向结束后的"保持质量"看这个。
  # （用全程首尾差会在"转了 218° 又保持住"时误报成漂移，角度回绕会骗人）
  tail = log[int(len(log) * 0.75) :, 6]
  tail_label = f"最近 {len(tail)*CTRL_DT:.2f} 仿真秒" if n_ctrl is None else "后 25% 时间"
  tail_drift = np.degrees(wrap_pi(tail[-1] - tail[0])) if len(tail) > 1 else 0.0
  print(f"  净转动      {net:+.1f}°（起点 {np.degrees(hdg0):+.1f}° → 终点 {np.degrees(hdgN):+.1f}°）")
  print(
    f"  末段朝向漂移 {tail_drift:+.2f}°（{tail_label}，3σ={np.degrees(tail.std()):.2f}°）"
    f"{'  [朝向保持 开]' if hdg_hold else '  [朝向保持 关=开环]'}"
  )
  if n_reset:
    print(f"  复位次数    {n_reset}")
  print(f"  实时率      {survived / wall:.2f}× 实时（CPU 单线程）")
  saturation_rate = raw_action_saturated / raw_action_samples if raw_action_samples else 0.0
  raw_action_abs_mean = raw_action_abs_sum / n if n else 0.0
  print(f"  原始动作    abs max {raw_action_abs_max:.3f}  abs mean {raw_action_abs_mean:.3f}  "
        f"饱和率 {saturation_rate * 100:.2f}%（阈值 ±{args.action_clip:g}）")
  if raw_action_abs_max > 3.0 or saturation_rate > 0.05:
    print("  ⚠️ 动作健康告警：策略输出在裁剪前偏大或饱和率偏高，可能出现腿部抽动。")

  if not fell and (n_ctrl is None or interrupted):
    print("\n  交互结束    正常退出；未进行定时姿态达标判定。")
    return 0
  ok = duration is not None and not fell and survived >= duration * 0.95 and mean[1] < 30
  print(f"\n  姿态检查    {'✅ 通过：存活与倾角达标' if ok else '⚠️ 未通过：见上面数据'}")
  if meta["n_terrain_geom"] > 0:
    print("  地形通过    未判定：需核对车道、接触、通过距离和终点；未摔倒不代表越过障碍。")
  if meta["n_terrain_geom"] > 0 and args.vx == 0.0 and args.vy == 0.0 and args.wz == 0.0:
    print("  ⚠️ 注意：本次指令为 0，机器人没动过 —— 这个『通过』与地形无关。")
  return 0 if ok else 1


def check_obs(model, data, meta, args) -> int:
  """与 mjlab 训练环境逐位对拍观测 —— 验证我们的拼装与训练一致。"""
  import torch

  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  print("\n[check-obs] 与 mjlab 训练环境对拍观测拼装…")
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = 1
  cfg.observations["actor"].enable_corruption = False  # 关掉观测噪声才可比
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()

  robot = env.scene["robot"]
  # 把 mjlab 的状态搬到普通 mujoco
  pose = robot.data.root_link_pose_w[0].cpu().numpy()  # [x y z qw qx qy qz]
  vel = robot.data.root_link_vel_w[0].cpu().numpy()  # [vx vy vz wx wy wz]
  reset(model, data, meta, args.spawn_x, args.spawn_y, args.spawn_yaw)
  data.qpos[0:7] = pose
  data.qpos[7 : 7 + meta["nq_joint"]] = robot.data.joint_pos[0].cpu().numpy()
  data.qvel[0:6] = vel
  data.qvel[6 : 6 + meta["nq_joint"]] = robot.data.joint_vel[0].cpu().numpy()
  mujoco.mj_forward(model, data)

  cmd = np.array([args.vx, args.vy, args.wz])
  dummy_action = np.zeros(meta["nq_joint"], dtype=np.float32)
  mine = build_obs(model, data, meta, cmd, 0.0, dummy_action)

  with torch.no_grad():
    ref = env.observation_manager.compute_group("actor")[0].cpu().numpy()

  # ★ 指令要拿 mjlab 自己采样出来的那条来比 —— 否则比的是两条不同的指令（第一次跑就踩到了）
  real_cmd = env.command_manager.get_command("twist")[0].cpu().numpy()
  mine = build_obs(model, data, meta, real_cmd, 0.0, dummy_action)
  if ref.ndim == 2:
    # At reset mjlab backfills every historical slot with the first frame.
    mine = np.repeat(mine[None, :], ref.shape[0], axis=0)
  mine, ref = mine.reshape(-1), ref.reshape(-1)
  print(f"  mjlab 采样到的指令 = {np.round(real_cmd, 4)}（用它来比）")

  print(f"  我的 obs  shape {mine.shape}    mjlab obs shape {ref.shape}")
  if mine.shape != ref.shape:
    print("  ✗ 维度不一致")
    return 2
  diff = np.abs(mine - ref)
  print(f"  最大绝对差 {diff.max():.3e}   最大差的下标 {int(diff.argmax())}")
  print(f"  我的前 8 维 {np.round(mine[:8], 5)}")
  print(f"  参考前 8 维 {np.round(ref[:8], 5)}")
  ok = diff.max() < 1e-4
  print(f"\n  {'✅ 观测拼装与训练完全一致' if ok else '⚠️ 有差异，见上面（注意 mjlab 侧含观测噪声，本检查已关噪声才可比）'}")
  return 0 if ok else 1


def main():
  ap = argparse.ArgumentParser(
    description="sim2sim：用普通 MuJoCo(CPU) + onnxruntime 跑导出的策略",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  ap.add_argument("task", help="任务 ID，如 LgDog1-Flat")
  ap.add_argument("--onnx", default=None, help="policy.onnx 路径（默认自动找最新 run）")
  ap.add_argument("--perception", default=None, help="MGDP 感知检查点；新导出默认读取 ONNX 旁的配套文件，旧模型需显式指定")
  ap.add_argument("--vx", type=float, default=0.0, help="前向速度指令 [m/s]")
  ap.add_argument("--vy", type=float, default=0.0, help="侧向速度指令 [m/s]")
  ap.add_argument("--wz", type=float, default=0.0, help="偏航角速度指令 [rad/s]")
  ap.add_argument("--duration", type=float, default=None,
                  help="可选定时验证秒数；有窗口默认持续回放，无窗口默认20秒")
  ap.add_argument("--camera-inset", action=argparse.BooleanOptionalAction, default=True, help="显示机器人第一视角角落小窗，F8 开关")
  ap.add_argument("--camera-inset-size", type=int, default=256)
  ap.add_argument("--camera-inset-fps", type=float, default=10.)
  ap.add_argument("--camera-inset-corner", choices=("bottom-left","bottom-right","top-left","top-right"), default="bottom-left")
  ap.add_argument("--viewer", action="store_true", help="弹 MuJoCo 窗口（键盘控制需要它）")
  ap.add_argument(
    "--map",
    default="flat",
    choices=("flat", "stairs", "slope", "obstacles", "mixed", "lgdog1", "race", "race_task"),
    help="测试地形：flat(默认平地) / stairs(四组楼梯) / slope(10°坡) / obstacles(高台列) / "
    "mixed(全都有) / lgdog1(LgRobotDog 那张测试图，191 个 geom) / "
    "race(2027 比赛障碍赛，14×6 m，八个障碍) / race_task(2027 任务赛内场，含 8 个物资箱)",
  )
  ap.add_argument("--map-file", default=None, help="外部地形 XML（只含 worldbody geom），优先级高于 --map")
  ap.add_argument("--spawn-x", type=float, default=0.0, help="出生点 x [m]")
  ap.add_argument("--spawn-y", type=float, default=0.0, help="出生点 y [m]（上某条楼梯车道就填车道 y）")
  ap.add_argument("--spawn-yaw", type=float, default=0.0, help="出生朝向 [度]，0 = 面朝 +x")
  ap.add_argument("--no-keyboard", action="store_true", help="关掉键盘控制（只看不控）")
  ap.add_argument(
    "--heading-hold",
    action="store_true",
    default=None,
    help="朝向保持（默认：有键盘时开）——不转向时复刻训练的 heading 控制律保持朝向，直行不画圈",
  )
  ap.add_argument("--no-heading-hold", dest="heading_hold", action="store_false", help="关掉朝向保持（开环下发 wz）")
  ap.add_argument("--speed-x", type=float, default=1.0, help="键盘前进/后退速度 [m/s]（默认 1.0）")
  ap.add_argument("--speed-y", type=float, default=0.5, help="键盘横移速度 [m/s]（默认 0.5）")
  ap.add_argument(
    "--speed-yaw", type=float, default=1.0, help="键盘转向速率 [rad/s]（默认 1.0，= 训练范围上限）"
  )
  ap.add_argument("--action-clip", type=float, default=1.0, help="动作裁剪上限（默认与训练 clip_actions=1.0 一致）")
  ap.add_argument("--check-obs", action="store_true", help="只对拍观测，不跑策略")
  args = ap.parse_args()
  if args.action_clip <= 0:
    ap.error("--action-clip 必须大于 0")
  if args.duration is not None and (not np.isfinite(args.duration) or args.duration < CTRL_DT):
    ap.error(f"--duration 必须是至少 {CTRL_DT:g} 秒的有限数值")
  return run(args)


if __name__ == "__main__":
  sys.exit(main())
