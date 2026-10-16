# LgDog1 项目索引（从这开始看）

> 位置：`~/mjlab/unitree_rl_mjlab_lgdog/`
> 这里是 **mjlab（MuJoCo Warp）路线**。你自己的 legged_gym 项目在 `~/rl/LgRobotDog/`，两边独立。

---

## 1. 我该看哪个文档？

| 我想…… | 看这个 | 行数 |
|---|---|---|
| **跑命令**（训练/回放/看曲线/sim2sim） | `commands.md` | 567 |
| **管训练轮次**（V1–V7 成绩、开新一轮、命名规则） | `TRAINING_V.md` | 241 |
| **了解目录和架构**、**改 reward** | `ARCHITECTURE.md` | 432 |
| **调 reward 权重/参数、定位奖励函数** | [REWARDS.md](REWARDS.md)；`scripts/list_rewards.py` | — |
| LgDog 接入的整体说明 | `LGDOG_GUIDE.md` | 133 |
| HIM（历史观测 + 状态估计）原理与实现 | `LGDOG_HIM.md` | 124 |
| **mjlab + HIM，标准崎岖地形盲走新任务** | [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md) | — |
| TerrainV4 地形任务 | `LGDOG_TERRAIN_V4.md` | 118 |
| 项目原版介绍（宇树 upstream） | `README_zh.md` | 240 |
| **部署/踩坑全集**（分类整理的坑与解法） | `../DEPLOY_NOTES.md` | 335 |
| 导出的策略怎么用 | `../export/<名字>/README.md` | — |

`HIM_VALIDATION.txt`、`validation_results.txt` 是历史验证输出，一般不用看。

**推荐阅读顺序**：`commands.md`（能跑起来）→ `TRAINING_V.md`（知道在训什么）→ `ARCHITECTURE.md`（知道在哪改）。

---

## 2. 目录结构

```
~/mjlab/
├── DEPLOY_NOTES.md              ← 部署/踩坑总集
├── export/                      ← 导出的策略（每个一份 onnx + pt + README）
│   ├── lgdog1_flat_4900/        V1 平地
│   └── lgdog1_v7_him_3900/      V7 HIM（282 维历史观测）
├── unitree_rl_mjlab.zip         ← 备份（原 unitree_rl_mjlab，已删）
└── unitree_rl_mjlab_lgdog/      ★ 当前工作目录
    │
    ├── INDEX.md                 ← 本文件
    ├── tb.sh                    ← TensorBoard 启停（start/stop/restart/status）
    ├── commands.md  TRAINING_V.md  ARCHITECTURE.md  LGDOG_*.md
    │
    ├── scripts/                 ← 入口脚本
    │   ├── train.py             训练（rsl_rl PPO，默认 rl_cfg，HIM 用自定义 runner）
    │   ├── play.py              回放（mjlab 原生 viewer / viser）★ 不给 --checkpoint_file 会自动取最新
    │   ├── sim2sim.py           部署链路验证（普通 MuJoCo CPU + onnxruntime，50Hz PD）
    │   ├── list_envs.py         列任务（用 --keyword 过滤）
    │   ├── visualize_terrain.py 看地形生成结果
    │   ├── csv_to_npz.py        动作数据转换
    │   └── check_lgdog_*.py     HIM / TerrainV4 的自检
    │
    ├── src/
    │   ├── assets/
    │   │   ├── robots/lgdog1/   ★ 机器人本体
    │   │   │   ├── lgdog1_constants.py   KP/KD/力矩/站姿/armature/frictionloss/ACTION_SCALE
    │   │   │   ├── make_lgdog1_xml.py    ★ 改 MJCF 改这里，再重跑生成 lgdog1.xml
    │   │   │   ├── xmls/lgdog1.xml       生成产物，别手改
    │   │   │   ├── xmls/assets/*.STL     网格
    │   │   │   └── LgDog1.urdf           原始 URDF
    │   │   └── terrains/lgdog1_testmap.xml   从 LgRobotDog 提取的测试地图（191 geom）
    │   │
    │   ├── policy_history.py    ★ 部署侧观测历史契约（282 = 47×6 那套校验）
    │   │
    │   └── tasks/velocity/      ★ 任务定义
    │       ├── velocity_env_cfg.py       所有机器人的基类（reward 字典、课程、nconmax）
    │       ├── mdp/rewards.py            奖励函数实现 ★ 加新奖励写这里
    │       ├── mdp/curriculums.py        课程（terrain_levels_vel、commands_vel）
    │       ├── mdp/metrics.py            traveled_distance 等指标
    │       ├── rl/him.py                 HIMRunner / HIMActor / HIMPPO
    │       ├── rl/him_estimator.py       HIM 估计器网络
    │       └── config/lgdog1/            ★ LgDog1 专属配置
    │           ├── __init__.py               任务注册（7 个 LgDog1-* 任务）
    │           ├── env_cfgs.py               指令范围/地形/站立比例/机器人绑定
    │           ├── reward_cfg.py             ★ 奖励统一入口：Flat/Rough/TerrainV4/HIM/HIM-Standard
    │           ├── rl_cfg.py                 网络结构 + PPO 超参 + experiment_name
    │           ├── blind_terrain_env_cfg.py  TerrainV4 / HIM 的基类（盲跑 + 足端射线 + 接触传感器）
    │           ├── blind_mdp.py              blind 分支的课程/终止/指标
    │           └── him_env_cfg.py            HIM 任务配置
    │
    ├── logs/
    │   ├── rsl_rl/<experiment_name>/<时间戳>_V<n>_<描述>/
    │   │       model_<N>.pt、policy.onnx（每存盘同步导出）、params/{env,agent}.yaml、git/
    │   ├── train_V<n>_*.log      训练文字日志
    │   ├── tensorboard/          TensorBoard 自身日志 + tb.pid
    │   └── play_*.log            回放日志
    │
    ├── deploy/                  C++ 实机部署（源码在，二进制需重建）
    ├── simulate/                unitree_mujoco（需重建）
    ├── tests/                   回归测试
    └── doc/                     setup 文档、gif、license
```

### experiment 目录（按任务分，V 号只体现在 run 名里）

```
logs/rsl_rl/
├── lgdog1_velocity/               ← LgDog1-Flat 和 LgDog1-Rough 共用！（★ 见下方注意）
│   ├── 2026-10-14_19-56-16_V1_flat
│   └── 2026-10-14_22-50-35_V2_rough_sighted
├── lgdog1_velocity_blind/         ← LgDog1-Rough-Blind（V3 V4 V5）
├── lgdog1_velocity_blind_gentle/  ← LgDog1-Rough-Blind-Gentle（V6）
├── lgdog1_velocity_blind_him_v1/  ← LgDog1-Rough-Blind-HIM（V7 V7b）
├── lgdog1_velocity_blind_terrain_v4/ ← TerrainV4
└── lgdog1_velocity_blind_him_standard_v1/ ← 标准 Rough + HIM（新任务）
```

> ⚠️ **`LgDog1-Flat` 和 `LgDog1-Rough` 共用 `lgdog1_velocity` 目录**（都取 `rl_cfg.py` 的默认
> `experiment_name`）。所以这两个任务**不能用自动选 checkpoint** —— 会拿到对方任务的权重
> （观测 234 维 vs 47 维）。play 会报 size mismatch 并提示用 `--load-run`，
> 或直接 `--checkpoint_file=` 指路径。其余 5 个任务各有独立实验目录。

---

## 3. 六条常用命令

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog && conda activate unitree_rl_mjlab

# 看训练曲线（浏览器开 http://localhost:6006）
bash tb.sh start

# 训练（新开一轮）
setsid nohup python scripts/train.py LgDog1-Rough-Blind-HIM \
  --env.scene.num-envs=1024 --agent.max-iterations=16000 \
  --agent.logger=tensorboard --agent.seed=42 --agent.run-name=V8_xxx \
  --enable-nan-guard True > logs/train_V8_xxx_$(date +%m%d_%H%M).log 2>&1 < /dev/null &

# 回放（不给 checkpoint 就自动取该任务最新 run 的最新 model）
# ★ 训练跑着的时候也能开（实测 4.2G 训练 + 4 envs play 并存无冲突），
#   每次重跑这条就是当前最新进度 —— 这就是"用 play 看训练"的正确姿势
python scripts/play.py LgDog1-Rough-Blind-HIM --num_envs 4 --viewer native

# 部署链路验证（不走 mjlab，普通 MuJoCo + onnx）
python scripts/sim2sim.py LgDog1-Flat --onnx=export/.../policy.onnx --vx 0.6 --viewer

# 列任务
python scripts/list_envs.py --keyword LgDog1
```

---

## 4. 任务与观测维度（别配错 checkpoint）

| 任务 | 观测 | 地形 | `experiment_name` |
|---|---|---|---|
| `LgDog1-Flat` | 47 | 平地 | `lgdog1_velocity` ⚠️ 与 Rough 共用 |
| `LgDog1-Rough` | **234**（含 187 维 height_scan） | 崎岖 | `lgdog1_velocity` ⚠️ |
| `LgDog1-Rough-Blind` | 47 | 崎岖 | `lgdog1_velocity_blind` |
| `LgDog1-Rough-Blind-Gentle` | 47 | 温和 | `lgdog1_velocity_blind_gentle` |
| `LgDog1-Rough-Blind-HIM` | **282**（47×6 历史） | 温和 | `lgdog1_velocity_blind_him_v1` |
| `LgDog1-Rough-Blind-HIM-Standard` | **282**（47×6 历史） | 标准 Rough，无机身高度扫描 | `lgdog1_velocity_blind_him_standard_v1` |
| `LgDog1-Rough-Blind-TerrainV4` | 47 | TerrainV4 | `lgdog1_velocity_blind_terrain_v4` |

**盲走部署输入**：普通 Blind 系列 actor 使用单帧 47 维；两类 HIM 使用 282 维六帧历史，都不读取地形高度图。TerrainV4/HIM 的仿真足下射线与 critic 真值仅供训练，不需要在真机端增加对应传感器。
