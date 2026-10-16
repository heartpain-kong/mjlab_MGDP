# unitree_rl_mjlab 目录与架构说明

LgDog 奖励调参的新入口是 `src/tasks/velocity/config/lgdog1/reward_cfg.py`。直接运行 `python scripts/list_rewards.py LgDog1-Rough-Blind-HIM` 查看最终权重与源码位置；完整操作见 [REWARDS.md](REWARDS.md)。

标准 Rough + HIM 融合任务见 [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md)，奖励 profile 为 `him_standard`，与 V7 的 `him` 独立。

配套：[工作台与目录说明](WORKBENCH.md)、[命令速查](commands.md)、[环境安装](setup_zh.md)。

---

## 1. 顶层目录

```
unitree_rl_mjlab/
├── scripts/          入口脚本（你敲命令的地方）
├── src/              ★ 本项目的代码：机器人资产 + RL 任务定义
├── deploy/           C++ 实机部署程序（每个机器人一个可执行文件）
├── simulate/         unitree_mujoco 仿真器（C++，用来做仿真部署）
├── doc/              官方文档 + 演示 gif
├── logs/             训练产物（模型 / onnx / tensorboard / 配置快照）
├── setup.py          依赖声明（mjlab==1.2.0 + mujoco-warp==3.5.0）
├── commands.md       命令速查（本文件旁边那份）
└── README_zh.md      官方说明
```

### `scripts/` —— 入口

| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口 |
| `play.py` | 加载 checkpoint 做仿真回放 |
| `list_envs.py` | 列出注册了哪些任务 |
| `list_rewards.py` | 查看任务最终奖励、参数及函数源码位置，不启动仿真 |
| `csv_to_npz.py` | 动作模仿：CSV 动作 → npz |
| `visualize_terrain.py` | 单独看地形生成效果 |

### `src/` —— 核心

```
src/
├── __init__.py              只定义了 SRC_PATH（别的模块靠它定位资产）
├── assets/                  机器人本体
│   ├── robots/
│   │   ├── unitree_g1/      ├─ 每个机器人一个目录，三件套：
│   │   ├── unitree_go2/     │    __init__.py
│   │   ├── unitree_h1_2/    │    <robot>_constants.py   ← 执行器/站姿/碰撞配置
│   │   ├── unitree_a2/      │    xmls/<robot>.xml       ← mjlab 版 MJCF
│   │   ├── unitree_as2/     │    xmls/assets/*.obj|STL  ← 网格
│   │   ├── unitree_h2/      │
│   │   ├── unitree_r1/      │
│   │   └── lgdog1/          └─ ★ 你们自研的狗（本次移植进来）
│   └── motions/             动作模仿的动作文件（g1 / g1_23dof）
└── tasks/                   RL 任务
    ├── __init__.py          import_packages 自动扫描 → 所有任务在这里注册
    ├── velocity/            速度跟踪（本项目主用）
    │   ├── velocity_env_cfg.py    ★ 所有机器人共用的基础配置
    │   ├── mdp/                   ★ 奖励/观测/终止/课程/指令 的【实现】
    │   │   ├── rewards.py         ★★ 奖励函数本体
    │   │   ├── observations.py    观测项
    │   │   ├── terminations.py    终止条件
    │   │   ├── curriculums.py     课程（含 reward_weight 权重调度工具）
    │   │   └── velocity_command.py 速度指令（含朝向控制器）
    │   ├── rl/                    runner（VelocityOnPolicyRunner）
    │   └── config/                ★ 每个机器人的【差异配置】
    │       ├── go2/{__init__,env_cfgs,rl_cfg}.py
    │       ├── g1/…  h1_2/…  a2/…  as2/…  h2/…  r1/…  g1_23dof/…
    │       └── lgdog1/            ★ 你们自研的狗
    └── tracking/            动作模仿（结构同上）
```

### ★ 机器人模型文件放在哪

**每个机器人一个目录：`src/assets/robots/<机器人>/`**，里面固定三样东西：

```
src/assets/robots/
├── unitree_go2/                    ← Go2
│   ├── go2_constants.py            ← 执行器(KP/KD/力矩)、出生姿态、碰撞配置
│   ├── __init__.py
│   └── xmls/
│       ├── go2.xml                 ← mjlab 版 MJCF（纯机器人，无场景）
│       ├── scene_go2.xml           ← 带场景版（unitree_mujoco 仿真器用）
│       └── assets/*.obj            ← 网格（16 个）
│
├── lgdog1/                         ← LgDog1（自研）
│   ├── lgdog1_constants.py         ← 同上
│   ├── make_lgdog1_xml.py          ← ★ MJCF 生成脚本
│   ├── LgDog1.urdf                 ← 原始 URDF（存档用，不参与训练）
│   └── xmls/
│       ├── lgdog1.xml              ← mjlab 版 MJCF
│       └── assets/*.STL            ← 网格（13 个）
│
└── unitree_g1/ unitree_h1_2/ …     其余机器人同构
```

三个必须知道的点：

1. **代码用 `SRC_PATH` 定位**（定义在 `src/__init__.py`），所以 `lgdog1_constants.py`
   里写的是 `SRC_PATH / "assets" / "robots" / "lgdog1" / "xmls" / "lgdog1.xml"`。
2. **网格目录名必须叫 `assets`**，因为 MJCF 里写的是 `meshdir="assets"`；
   加载时由 `get_assets()` 把整个目录读成 `{文件名: 字节}` 塞给 MuJoCo（支持 .obj / .STL）。
3. **LgDog1 的 `lgdog1.xml` 是脚本生成的** —— 改几何/碰撞体要改
   `make_lgdog1_xml.py` 然后重跑（`python src/assets/robots/lgdog1/make_lgdog1_xml.py`），
   别手改 xml，否则下次重跑就被覆盖了。

新增一个机器人的流程：建目录 → 放 MJCF 和网格 → 写 `<robot>_constants.py`（照抄 `go2_constants.py`）
→ 在 `src/assets/robots/__init__.py` 导出 `get_<robot>_robot_cfg`
→ 在 `src/tasks/velocity/config/<robot>/` 注册任务。

### `deploy/` —— 实机部署（C++）

```
deploy/
├── include/        公共头（FSM 状态机 / isaaclab 兼容层 / 参数解析 / 插值器）
├── thirdparty/     cnpy（读 npz）+ onnxruntime（x64 与 aarch64 两套预编译）
└── robots/<robot>/ 每个机器人一套：
    ├── main.cpp               入口
    ├── src/State_*.cpp        各状态实现
    ├── config/config.yaml     FSM 状态机与手柄按键映射
    ├── config/policy/…        策略参数 + exported/policy.onnx
    └── build/                 cmake 产物 <robot>_ctrl
```

---

## 2. 一次训练发生了什么

```
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 …
        │
        ├─ tyro 解析命令行（第一个参数是任务名，其余是覆盖项）
        │
        ├─ import src.tasks
        │     └─ import_packages 自动扫 src/tasks 下所有包
        │           └─ config/lgdog1/__init__.py 执行 register_mjlab_task(...)
        │                 → 任务 "LgDog1-Flat" 进入 mjlab 的全局注册表
        │
        ├─ load_env_cfg("LgDog1-Flat")
        │     └─ lgdog1_flat_env_cfg()
        │           └─ lgdog1_rough_env_cfg()
        │                 └─ make_velocity_env_cfg()      ← 基础配置（共用）
        │                       + 机器人差异覆盖（本文件里逐项改）
        │
        ├─ ManagerBasedRlEnv(cfg)          ← mjlab 的 manager-based 环境
        │     ├─ Scene        机器人 Entity + 地形 + 传感器
        │     └─ 各 Manager：Action / Observation / Command / Reward /
        │                    Termination / Event / Curriculum / Metrics
        │
        ├─ RslRlVecEnvWrapper → VelocityOnPolicyRunner（rsl-rl 的 PPO）
        │
        └─ mujoco_warp（GPU 物理）＋ torch（网络）
              ↓
        logs/rsl_rl/<experiment_name>/<日期时间>/model_*.pt + policy.onnx
```

**一句话**：`scripts` 是入口，`src/tasks/**/config/<robot>` 决定"用哪个机器人、长什么样"，
`src/tasks/velocity/velocity_env_cfg.py` 决定"训练任务是什么"，
`src/tasks/velocity/mdp/` 决定"每一项怎么算"，`mjlab` 包提供框架，`mujoco_warp` 提供物理。

---

## 3. ★ 在哪里改 reward

### 3.1 先理解：reward 是"字典 + 函数"两部分

每个奖励项 = **一个字典条目**（权重、参数、用哪个函数）+ **一个函数**（怎么算）：

```python
# 这条是"字典"（配置）
rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity,   # ← 指向"函数"
    weight=1.0,                       # ← 权重
    params={"command_name": "twist", "std": 0.5},   # ← 参数
)
```

实际每步的值：

```
该步奖励 = 函数返回值 × weight × dt        （dt = 0.005 × decimation = 0.02 s）
```

> ★ **`weight` 会被乘上 dt(0.02)**，所以 mjlab 里 `is_terminated` 的 `-200`
> 实际在函数返回 1 的终止步贡献 **-4.0**。其他框架的 dt 处理和奖励返回值需分别核对，
> 连续力惩罚也不能直接与二值碰撞惩罚换算。

### 3.2 四个层级，按"你要改什么"对号入座

| 想干的事 | 改哪个文件 | 频率 |
|---|---|---|
| **A. 调 LgDog 的奖励权重/参数** | `src/tasks/velocity/config/lgdog1/reward_cfg.py` | ★★★ 最常用 |
| **B. 改其他机器人通用的默认权重** | `src/tasks/velocity/velocity_env_cfg.py` 的 `rewards = {...}` | ★★ |
| **C. 改奖励的算法本身 / 加一个新奖励函数** | `src/tasks/velocity/mdp/rewards.py` | ★ |
| **D. 不改代码、只在这一次训练里试个数** | 命令行 `--env.rewards.<名字>.weight=…` | ★★ 调参时最好用 |
| **E. 让权重随训练进度变化** | `src/tasks/velocity/mdp/curriculums.py` 的 `reward_weight`（工具已备好） | ★ 进阶 |

### A. 改 LgDog 的奖励 —— 最常用

文件：**`src/tasks/velocity/config/lgdog1/reward_cfg.py`**。

- `COMMON_WEIGHTS`：公共权重，例：将 `"track_linear_velocity": 2.0` 改为 `1.6`。
- `REWARD_PROFILES`：按 `flat` / `rough` / `terrain_v4` / `him` / `him_standard` 单独覆盖权重和参数；这些配置互不继承。
- `make_lgdog_rewards()`：公共参数、函数绑定和身体/足端绑定。

只改 HIM 时，在 `REWARD_PROFILES["him"]["weights"]` 中填写要调整的项，例如 `{"foot_clearance": -0.8}`；目标高度在该项的 `params.foot_clearance.target_height` 中调整。

新增奖励先在公共权重表增加名字，再在工厂字典中增加 `term("my_term", mdp.my_reward_function, command_name="twist")`。详细步骤见 [REWARDS.md](REWARDS.md)。其他机器人仍按其原有 `config/<robot>/env_cfgs.py` 方式配置。

### B. 改所有机器人共用的默认

文件：**`src/tasks/velocity/velocity_env_cfg.py`** → `make_velocity_env_cfg()` 里的 `rewards = { ... }`

这里定义通用的奖励默认值，供 G1/Go2/H1_2 等任务继承。LgDog 会用 `reward_cfg.py` 生成完整奖励字典，因此调整这些通用奖励默认值不再改变 LgDog 的奖励配置。共享奖励函数的公式变化仍会影响所有使用该函数的任务。

### C. 改奖励函数本身 / 加新函数

文件：**`src/tasks/velocity/mdp/rewards.py`**

★ **关键：真正生效的是这个仓库自己的实现，不是 mjlab 包里的。**
证据有两条：

1. `velocity_env_cfg.py` 里 `import src.tasks.velocity.mdp as mdp`（第 33 行）
   **覆盖**了上面 `from mjlab.tasks.velocity import mdp`（第 26 行），
   所有 `func=mdp.xxx` 都指向本仓库；
2. 训练产物快照里的函数路径就是本仓库：
   `logs/rsl_rl/lgdog1_velocity/<时间>/params/env.yaml` →
   ```yaml
   track_linear_velocity:
     func: !!python/name:src.tasks.velocity.mdp.rewards.track_linear_velocity ''
     weight: 1.0
   ```

### ★★ 18 个奖励项，函数分别在哪（改之前先看这张表）

LgDog1 四个任务每轮都有 **18 个奖励项**，但函数实现分散在两个地方 —— 这决定了你能改什么：

| 模块 | 项数 | 说明 |
|---|---|---|
| **本仓库** `src/tasks/velocity/mdp/rewards.py` | **13** | 可以直接改算法 |
| **本仓库** `src/tasks/velocity/config/lgdog1/blind_mdp.py` | **1** | LgDog1 专属（地形版 `foot_clearance`） |
| **mjlab 包内**（`site-packages/mjlab/envs/mdp/rewards.py`） | 4 | 本仓库**没有**源码，只能改权重/参数，不能改算法 |

| 奖励项 | weight | 函数所在 |
|---|---|---|
| `track_linear_velocity` | 2 | ★ `mdp/rewards.py` |
| `track_angular_velocity` | 1 | ★ `mdp/rewards.py` |
| `pose` | 1.5 | ★ `mdp/rewards.py`（类名 `variable_posture`） |
| `body_orientation_l2` | −1 | ★ `mdp/rewards.py` |
| `body_ang_vel` | −0.05 | ★ `mdp/rewards.py`（`body_angular_velocity_penalty`） |
| `angular_momentum` | −0.025 | ★ `mdp/rewards.py`（`angular_momentum_penalty`） |
| `foot_clearance` | −1 | ★ `mdp/rewards.py`，**但 blind/HIM 任务覆盖成 `blind_mdp.feet_clearance_terrain`** |
| `foot_gait` | 0.5 | ★ `mdp/rewards.py` |
| `foot_slip` | −0.25 | ★ `mdp/rewards.py` |
| `feet_air_time` | 0 | ★ `mdp/rewards.py` |
| `feet_stumble` | 0 / −0.5 | ★ `mdp/rewards.py`（地形专属） |
| `leg_contact_force` | 0 / −1 | ★ `mdp/rewards.py`（地形专属） |
| `soft_landing` | −0.001 | ★ `mdp/rewards.py` |
| `stand_still` | −1 | ★ `mdp/rewards.py` |
| `action_rate_l2` | −0.05 | mjlab 包 |
| `joint_acc_l2` | −2.5e-07 | mjlab 包 |
| `joint_pos_limits` | −10 | mjlab 包 |
| `is_terminated` | −200 | mjlab 包 |

> `rewards.py` 里还定义了 `feet_swing_height`、`self_collision_cost` 两个函数，
> **当前 LgDog1 任务没有使用**（定义着备用）。

### 当前各任务的实际权重（实测导出，不是照抄文档）

| 奖励项 | Flat | Rough | Gentle | HIM |
|---|---|---|---|---|
| `track_linear_velocity` | 2 | 2 | 2 | 2 |
| `track_angular_velocity` | 1 | 1 | 1 | 1 |
| `pose` | 1.5 | 1.5 | 1.5 | 1.5 |
| `foot_gait` | 0.5 | 0.5 | 0.5 | 0.5 |
| `body_orientation_l2` | −1 | −1 | −1 | −1 |
| `foot_clearance` | −1 | −1 | −1 | −1 |
| `foot_slip` | −0.25 | −0.25 | −0.25 | −0.25 |
| `action_rate_l2` | −0.05 | −0.05 | −0.05 | −0.05 |
| `body_ang_vel` | −0.05 | −0.05 | −0.05 | −0.05 |
| `angular_momentum` | −0.025 | −0.025 | −0.025 | −0.025 |
| `joint_pos_limits` | −10 | −10 | −10 | −10 |
| `joint_acc_l2` | −2.5e-07 | −2.5e-07 | −2.5e-07 | −2.5e-07 |
| `soft_landing` | −0.001 | −0.001 | −0.001 | −0.001 |
| `stand_still` | −1 | −1 | −1 | −1 |
| `is_terminated` | −200 | −200 | −200 | −200 |
| `feet_air_time` | 0 | 0 | 0 | 0 |
| **`feet_stumble`** | **0** | **−0.5** | **−0.5** | **−0.5** |
| **`leg_contact_force`** | **0** | **−1** | **−1** | **−1** |

→ **四个任务只差最后两行**（地形专属项）+ `foot_clearance` 的 `target_height`
（Flat 0.10 / Rough 0.15 / Gentle 0.15 / HIM 0.10 —— 见下方不一致说明）。
重构保持这些任务当前的配置数值。历史训练的实际配置仍应以各 run 的 `params/env.yaml` 为准；对比还需考虑指令范围、函数定义与地形差异。

### ★ 地形奖励（`feet_stumble` / `leg_contact_force`）

LgDog 的启停由 `reward_cfg.py` 的公共权重及 profile 决定。Flat 关闭地形碰撞两项，Rough/Blind/Gentle/TerrainV4/HIM 开启；`feet_air_time` 当前在所有 LgDog 任务中关闭。其他机器人的通用配置仍保留其原有设置，不能据此推断为关闭。

**LgDog1 的当前配置**（见 `config/lgdog1/reward_cfg.py`）：

| 奖励项 | Flat | Rough | 为什么 |
|---|---|---|---|
| `feet_stumble` | 0.0 | **−0.5** | 地形里 40% 是台阶，上台阶最常见的失败就是脚磕在立面上 |
| `leg_contact_force` | 0.0 | **−1** | 罚**大腿/小腿/髋撞到地形**（上台阶时小腿容易磕在台阶棱上）。只靠 `illegal_contact` 终止是 0/1 信号给不了梯度，这里给连续力惩罚 |
| `foot_clearance` 的 `target_height` | 0.10 | **0.15** | 原目标 0.10 正好等于最高台阶高度，太勉强；脚得抬过台阶才踩得上去 |
| `feet_air_time` | 0.0 | 0.0 | 实测开启后贡献 ≈0.0001（`single_stance` 门控早期不成立），且与 `foot_gait` 重复，故不占权重 |

> `target_height` 为 Flat 0.10 / Rough、Gentle 0.15 / TerrainV4、HIM 0.10。
> Flat/Rough/Gentle 使用世界坐标 z，而 TerrainV4/HIM 使用相对足下地面的高度且仅约束摆动脚，因此两类数值不能直接当作相同口径。现在所有目标值均在 `reward_cfg.py` 中显式设置，HIM 可单独调参。

`feet_stumble` 的判据（与 legged_gym 同源，但返回连续量而非 0/1）：

```
水平接触力 > 4 × 竖直接触力   ⇒   判定为"撞立面"
cost = max(水平力 − 4×竖直力, 0)     按最严重的那只脚
```

> 该函数返回连续的力超出量，权重 −0.5 的每步贡献为 `−0.5 × 力超出量 × 0.02`，不是固定 −0.01。与其他框架的二值 stumble 奖励对照时，应同时核对返回值与时间缩放。

> 实测确认它是**地形专属**的：Rough 上 `Metrics/feet_stumble_mean` 0.02~0.19（在触发），
> Flat 上恒为 **0.0000**（平地踩地时水平力≈摩擦力 < 4×竖直力，不误触发）。

写法模板（照抄就能加一个）：

```python
def my_reward(
    env: ManagerBasedRlEnv,
    command_name: str,                       # 参数从 params 传进来
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """一句话说明这个奖励在鼓励什么。必须返回 shape=[num_envs] 的张量。"""
    asset: Entity = env.scene[asset_cfg.name]
    # …… 你的计算 ……
    return torch.sum(torch.square(x), dim=1)     # 越大越好；要罚就返回正数、把 weight 设成负数
```

改完**不用注册**：`src/tasks/velocity/mdp/__init__.py` 里是 `from .rewards import *`，
顶层函数会自动导出成 `mdp.my_reward`（文件里没有 `__all__`，不用加）。

### D. 命令行临时试权重（推荐调参用）

```bash
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 \
    --env.rewards.track_linear_velocity.weight=1.6 \
    --env.rewards.stand_still.weight=-2.5
```
✅ 实测：启动时打印的 Reward 表会直接显示改后的值：
```
|   0   | track_linear_velocity  |      1.6 |
|   14  | stand_still            |     -2.5 |
```
试出满意的数再写回 LgDog 的 `reward_cfg.py` 对应 profile。

### E. 让权重随训练进度变（进阶）

`src/tasks/velocity/mdp/curriculums.py` 里已经写好一个工具（**目前还没人用**）：

```python
def reward_weight(env, env_ids, reward_name: str, weight_stages: list[RewardWeightStage]):
    """按训练步数分段改某个奖励的权重。"""
```
用法（加到任务的 `cfg.curriculum` 里）：

```python
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from src.tasks.velocity import mdp

cfg.curriculum["track_vel_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
        "reward_name": "track_linear_velocity",
        "weight_stages": [
            {"step": 0,     "weight": 0.5},     # 前期先让它站住
            {"step": 5000,  "weight": 1.6},     # 后期加跟踪
        ],
    },
)
```

---

## 4. 改完怎么确认生效了

三个地方能看，**建议每次都看一眼**：

```bash
# ① 启动时打印的 Reward 表（最直接）
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 --agent.max-iterations=1 2>&1 | grep -A20 "RewardManager"

# ② 训练产物里的完整配置快照（事后追溯用）
grep -A6 "track_linear_velocity:" logs/rsl_rl/lgdog1_velocity/<时间>/params/env.yaml

# ③ TensorBoard 里的 Episode_Reward/<名字>（看它实际贡献多少）
tensorboard --logdir=logs/rsl_rl --port 6006
```

> ⚠️ 常见坑：**改了 `rewards.py` 里的函数但没生效** —— 检查是不是改到了 mjlab 包里
> （`…/site-packages/mjlab/tasks/velocity/mdp/rewards.py`）。要改的是
> **仓库里的 `src/tasks/velocity/mdp/rewards.py`**。

---

## 5. 其他 Manager 在哪（顺带速查）

| 你想要的 | 配置写在哪（`config/<robot>/env_cfgs.py`） | 实现写在哪 |
|---|---|---|
| 奖励 | LgDog 用 `config/lgdog1/reward_cfg.py`；其他机器人用 `cfg.rewards["…"]` | `mdp/rewards.py` 或 `list_rewards.py` 指向的函数源码 |
| 观测 | `cfg.observations["actor"｜"critic"].terms["…"]` | `src/tasks/velocity/mdp/observations.py` |
| 终止条件 | `cfg.terminations["…"]` | `src/tasks/velocity/mdp/terminations.py` |
| 指令范围 | `cfg.commands["twist"].ranges` | `src/tasks/velocity/mdp/velocity_command.py` |
| 课程 | `cfg.curriculum["…"]` | `src/tasks/velocity/mdp/curriculums.py` |
| Metrics（只记录、不进奖励） | `cfg.metrics["…"]` | `src/tasks/velocity/mdp/metrics.py` |
| 动作 | `cfg.actions["joint_pos"]`（缩放 `scale`） | mjlab 包内 |
| 随机化/事件 | `cfg.events["…"]` | mjlab 包内（`mjlab/envs/mdp/dr`） |
| 场景/地形/传感器 | `cfg.scene.*` | — |
| 物理/频率 | `cfg.sim.mujoco.*`、`cfg.decimation` | — |
| 网络/PPO | `config/<robot>/rl_cfg.py` | mjlab + rsl-rl |

**机器人本体**（不改任务，只改机器）：

| 你想要的 | 改哪 |
|---|---|
| PD 增益、力矩上限、armature | `src/assets/robots/<robot>/<robot>_constants.py` |
| 出生姿态、出生高度 | 同上（`INIT_STATE`） |
| 碰撞体怎么算 | 同上（`FULL_COLLISION` / `FEET_ONLY_COLLISION`） |
| 几何、质量、关节限位 | `src/assets/robots/<robot>/xmls/<robot>.xml` |
| LgDog1 的 MJCF | ★ 改**生成脚本** `make_lgdog1_xml.py` 后重跑，别手改 xml |

---

## 6. 举个例子：把 LgDog1 的速度跟踪从 2.0 调到 1.6

```bash
# 1) 先命令行试（不改代码）
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 --agent.max-iterations=50 \
    --agent.logger=tensorboard --env.rewards.track_linear_velocity.weight=1.6
# 2) 看 tensorboard 里 Episode_Reward/track_linear_velocity 和 Mean episode length
# 3) 满意了写回配置文件
```

编辑 `src/tasks/velocity/config/lgdog1/reward_cfg.py` 的 `COMMON_WEIGHTS`：

```python
"track_linear_velocity": 1.6,
```

保存后在新的 `train.py` 进程中生效。已经运行的训练进程不会热更新。只想改某个任务时，改其 `REWARD_PROFILES` 权重而不是公共值。

> 想确认改对了：启动时打印的 Reward 表里那一行的数字，或者事后看
> `logs/rsl_rl/lgdog1_velocity/<时间>/params/env.yaml` 的快照。
