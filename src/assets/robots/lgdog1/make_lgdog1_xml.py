#!/usr/bin/env python3
"""把 LgDog1 的 sim2sim MJCF 转成 mjlab 的「纯机器人」MJCF。

转换要点（与 src/assets/robots/unitree_go2/xmls/go2.xml 的约定对齐）：
  · meshdir 改为 assets（meshes 已拷到同级 assets/）
  · 去掉 floor / light / camera —— mjlab 的地形由 SceneCfg 提供
  · 去掉 <actuator> —— mjlab 由 EntityArticulationInfoCfg 生成 position 执行器
  · <joint name="root" type="free"/> → <freejoint name="floating_base_joint"/>
  · <sensor> 换成 mjlab 需要的那 4 个（imu_ang_vel / imu_lin_vel / imu_lin_acc / root_angmom）
  · body / joint / geom / site / inertial 一律逐字保留，保证物理与原 sim2sim 一致
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path("/home/heartpain/rl/LgRobotDog/legged_gym/resources/robots/LgDog1/xml/LgDog1.xml")
DST = Path("/home/heartpain/mjlab/unitree_rl_mjlab/src/assets/robots/lgdog1/xmls/lgdog1.xml")

text = SRC.read_text(encoding="utf-8")

# ---- 1. 取 <worldbody> ... </worldbody> ----
m = re.search(r"<worldbody>(.*?)</worldbody>", text, re.S)
assert m, "找不到 worldbody"
body = m.group(1)

# ---- 2. 去掉场景元素（floor / light / camera），只留机器人 ----
body_lines = []
for line in body.splitlines():
    s = line.strip()
    if s.startswith("<light ") or s.startswith("<camera "):
        continue
    if s.startswith('<geom name="floor"'):
        continue
    body_lines.append(line.rstrip())
body = "\n".join(body_lines).strip("\n")

# ---- 3. 自由关节改名 ----
body, n = re.subn(
    r'<joint name="root" type="free" />',
    '<freejoint name="floating_base_joint"/>',
    body,
)
assert n == 1, f"freejoint 替换了 {n} 次（应为 1）"

# ---- 3b. 碰撞/视觉体改名：序号从 `xxx_collision_0` 挪到 `xxx0_collision` ----
# ★ 必须改！mjlab 的 CollisionCfg 用 `.*_collision\d*$` 匹配碰撞体（go2 的命名是
#   `FL_calf1_collision`，序号在 `_collision` 之前）。原命名 `FR_calf_collision_1`
#   匹配不上 → 会被当成非碰撞体把 contype/conaffinity 清成 0 → 足端穿地、机器人
#   一出生就自由落体（实测 z 从 0.31 掉到 0.07，倾斜 0°）。
body, n = re.subn(r'name="(\w+?)_(collision|visual)_(\d+)"', r'name="\1\3_\2"', body)
assert n == 24, f"碰撞/视觉体改名了 {n} 处（应为 24 = 4 腿 × (thigh 2 + calf 2) × 2 类）"
assert "_collision_" not in body and "_visual_" not in body, "还有漏改的 _collision_N / _visual_N"


HEADER = '''<mujoco model="lgdog1">
  <!--
    LgDog1 —— 用户自研四足（trunk 长 0.404 m， thigh 0.19 m + calf 0.245 m，整机 ≈10.06 kg）
    由 resources/robots/LgDog1/xml/LgDog1.xml 转换而来（脚本见本目录生成说明）。
    与 sim2sim 的同名文件相比只做了 mjlab 适配：去场景/执行器、加 mjlab 需要的传感器。
    body / joint / geom / site / inertial 逐字保留，物理与 sim2sim 一致。

    命名约定（mjlab 任务配置依赖它们，改名要同步改 lgdog1_constants.py）：
      · 机身 body      = trunk        （imu site 在 trunk 原点）
      · 腿 body        = {FR,FL,RR,RL}_hip / _thigh / _calf
      · 关节           = {FR,FL,RR,RL}_{hip,thigh,calf}_joint
      · 足端 site      = {FR,FL,RR,RL}_foot
      · 碰撞体         = trunk_collision / *_hip_collision /
                         {leg}_thigh{1,2}_collision / {leg}_calf{1,2}_collision
                         （★ 序号在 `_collision` 之前，见下面第 3b 步的说明）
      · 足端接触 geom  = {FR,FL,RR,RL}_calf2_collision（球，r=0.012）
      · 关节顺序       = [FR,FL,RR,RL] × [hip,thigh,calf]
    关节限位 hip ±1.45 / thigh ±1.75 / calf ±0.85，力矩 hip,thigh ±10 Nm、calf ±20 Nm（1:2 减速比）。
    关节被动参数 armature=0.0042 / damping=0.01 / frictionloss=0.2 与 sim2sim 一致
    （注意：armature 与 frictionloss 由 lgdog1_constants.py 的 actuator 配置写入，两者取值必须相同）。
  -->
  <compiler angle="radian" meshdir="assets" autolimits="true" />

  <default>
    <geom contype="1" conaffinity="1" friction="0.6 0.3 0.3" rgba="0.5 0.6 0.7 1" margin="0.001" group="0" />
    <joint damping="0.01" armature="0.0042" frictionloss="0.2" />
  </default>

  <asset>
'''

MESHES = [
    ("base_link", "base_link.STL"),
    ("FL_hip", "FL_hip.STL"),
    ("FL_thigh", "FL_thigh.STL"),
    ("FL_calf", "FL_calf.STL"),
    ("FR_hip", "FR_hip.STL"),
    ("FR_thigh", "FR_thigh.STL"),
    ("FR_calf", "FR_calf.STL"),
    ("RL_hip", "RL_hip.STL"),
    ("RL_thigh", "RL_thigh.STL"),
    ("RL_calf", "RL_calf.STL"),
    ("RR_hip", "RR_hip.STL"),
    ("RR_thigh", "RR_thigh.STL"),
    ("RR_calf", "RR_calf.STL"),
]

FOOTER = '''  </worldbody>

  <sensor>
    <gyro name="imu_ang_vel" site="imu" />
    <velocimeter name="imu_lin_vel" site="imu" />
    <accelerometer name="imu_lin_acc" site="imu" />
    <subtreeangmom name="root_angmom" body="trunk" />
  </sensor>
</mujoco>
'''

parts = [HEADER]
for name, f in MESHES:
    parts.append(f'    <mesh name="{name}" file="{f}" />\n')
parts.append("  </asset>\n\n  <worldbody>\n")
parts.append(body + "\n")
parts.append(FOOTER)

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text("".join(parts), encoding="utf-8")
print(f"写入 {DST}")
print(f"  行数 {len(DST.read_text(encoding='utf-8').splitlines())}")

# ---- 4. 立刻用 mujoco 验证能编译 ----
try:
    import mujoco
except ImportError:
    print("（本解释器没有 mujoco，跳过编译校验）")
    sys.exit(0)

model = mujoco.MjModel.from_xml_path(str(DST))
print("  ✅ mujoco 编译通过")
print(f"     body={model.nbody} joint={model.njnt} sensor={model.nsensor} mass={model.body_mass.sum():.3f}")
print("     sensors:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)])
print("     joints :", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)])
assert model.njnt == 13, model.njnt
assert model.nsensor == 4, model.nsensor
assert abs(model.body_mass.sum() - 10.055) < 0.01, model.body_mass.sum()
