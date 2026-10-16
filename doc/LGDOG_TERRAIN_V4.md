# LgDog 盲走地形训练：分析与修改

本说明记录 TerrainV4 阶段。后续新增的 HIM 实验、历史观测部署接口和最新验证结果见 `LGDOG_HIM.md`。

日期：2026-10-16。基于用户提供的 `E:\mjlab\unitree_rl_mjlab.zip`，并核对了锁定版本 mjlab 1.2.0 的管理器和传感器源码。

工作副本：`E:\mjlab\unitree_rl_mjlab_lgdog`。原 ZIP、原来的官方源码目录 `E:\mjlab\unitree_rl_mjlab` 均未被覆盖。工作副本包含源代码、机器人模型和日志文本；训练 checkpoint、ONNX、TensorBoard events 仍保存在原 ZIP 中，没有全部重复解压。`.git`、缓存和约 1.12 GB 的 VSCode 数据库也未解压。

## 从实际工程得到的结论

这已经是一套有 LgDog 本体、平地训练、盲走 v1/v2/v3、温和地形和 sim2sim 工具的工程。此次修改以这些数据为基底，没有套用此前针对 Go2 的实验配置。

| 问题 | 实际证据 | 处理 |
|---|---|---|
| 足端奖励使用世界高度 | `mdp/rewards.py::feet_clearance` 直接比较 `site_pos_w[...,2]` 与 0.15 | V4 改为足端高度减正下方地面高度，并仅约束摆动脚 |
| 盲走同时关闭 critic 地形观测 | `lgdog1/env_cfgs.py` 删除 actor、critic 的 height_scan 和扫描器 | V4 仅 actor 不读地形，critic 使用特权地形数据 |
| 扫描可能打到自身可视网格 | 原 raycast 默认包含 group 1；LgDog visual meshes 在 group 1 | V4 扫描限定 group 0 地形；机器人碰撞在 group 3 |
| 课程仍受最后一条指令影响 | 路径长度已累计，但降级阈值仍使用当前指令 × 整回合时长；指令每 3 秒切换 | V4 逐步累计指令距离和沿指令方向的进度 |
| 擦腿和机身碰撞同样终止 | 原 nonfoot 传感器包括 trunk、hip、thigh、calf，超过 10 N 即终止 | V4 拆分机身和腿部，保留持续/严重腿部碰撞终止 |
| 日志口径易被误读 | mjlab 1.2 终止日志是 reset 批次中的数量；MetricsManager 输出每步平均值 | 增加有明确单位的速度误差、整回合进度和 reset 批次失败比例 |

路径长度课程的已有修复是有价值的，不能说你们完全没有做课程。问题在于仅累计路程仍不能区分跟踪指令、侧滑和原地绕圈，也不能证明已经接触到地形障碍。

## 日志分析

下面每项均取各日志最后 100 次迭代的平均值。日志来自不同任务、阶段和奖励设置，总奖励不能直接作为横向排名。

| 日志 | 最后迭代 | 平均回合步数 | 平均地形等级 | 平均 value loss |
|---|---:|---:|---:|---:|
| 平地 4096 | 4926/5300 | 992.18 | — | 0.0116 |
| blind v1 | 7301/16000 | 714.06 | 0.00081 | 0.1695 |
| blind v2 | 11545/16000 | 796.15 | 0 | 0.1645 |
| blind v3 | 15999/16000 | 705.43 | 0.11337 | 322.1010 |
| blind gentle | 2777/16000 | 765.02 | 5.87002 | 0.2275 |
| 带扫描 rough | 15999/16000 | 811.05 | 0 | 0.0477 |

v3 最后 100 次迭代的 value loss 中位数为 **0.1973**，最大值为 **7570.7654**；action_rate_l2 奖励最小值为 **-15.2617**。平均值被少数尖峰显著拉高，需要继续定位动作输出/状态异常，而不能仅归因于学习率或地形难度。这次没有用任意奖励裁剪掩盖尖峰，也没有声称已确认尖峰根因。

温和地形已经有升级效果，不能继续沿用“所有盲走实验都停在 level 0.1”的结论。但其中有 40% 平地，平均等级不能替代按地形类型统计的通过率。

`Episode_Termination/illegal_contact=1.0` **不是 100% 回合失败**。`Episode_Metrics/traveled_distance≈0.00319` 是每步距离的回合平均值，**不是整回合只走了 3 毫米**。完整数值汇总见 `lgdog_log_summary.json`。

另一个注释问题：本地 legged_gym 的 `_prepare_reward_function` 同样会将 reward scale 乘 dt。工程中“legged_gym 不乘 dt，所以迁移到 mjlab 应除以 0.02”的通用解释不成立；不同奖励函数的单位、归约方式还需单独比较。本次保留既有主要奖励权重，避免依据错误换算继续放大惩罚。

## V4 实现

新任务：**`LgDog1-Rough-Blind-TerrainV4`**。原任务 ID 保留，可以做对照。

1. **非对称 actor / critic**：actor 仍为 47 维单帧输入，不含 height_scan；critic 恢复 187 维地形扫描，总计预期 261 维。足端高度也改为相对地面高度。训练使用射线不意味着真机需要雷达。
2. **足端离地奖励**：四个足端各用一条向下的地形射线，射线起点抬高 0.5 m，避免轻微穿透造成漏检。只对有运动指令的摆动脚施加足速加权误差，目标高度先设 0.10 m；无有效命中时不产生虚假高度惩罚，并记录射线有效率。这是竖直离地高度，不是相对斜坡法线的距离，也不预测前方台阶。
3. **按整回合跟踪表现调整地形**：累计每步指令的平移距离，并累计实际速度在指令方向上的投影，负向和超速进度不额外计分。请求距离至少 1 m 才参与升降级；成功存活至少 90% 回合、跟踪比至少 0.75、进度至少 2 m、离开出生平台区域至少 1 s 才升级；失败或跟踪比低于 0.35 则降级。其余保持。纯站立和纯转向回合不因此降级。
4. **接触恢复空间**：机身接触仍使用 10 N 阈值，原倾倒终止保留。腿部同一碰撞体超过 10 N 连续 0.10 s，或任一物理子步超过 80 N 时终止；短暂轻蹭仍受腿部接触奖励惩罚。10/80 N 和 0.10 s 是实验起点，不是已验证的机械安全极限。

V4 先沿用 Gentle 地形范围：40% 平地、各 15% 上下台阶（0–5 cm）、各 10% 坡面（坡度参数 0–0.5）、各 5% 随机起伏与波浪。初始等级最多 1。速度范围固定为 vx [-0.4,0.8]、vy [-0.3,0.3] m/s、wz [-0.6,0.6] rad/s，取消按迭代数自动跳到 2 m/s 的课程。站立比例仍为 25%。这些参数服务于这一轮地形实验，应由评估决定何时提高。

没有加入历史编码器、HIM 或 RMA。这些可以作为下一轮，但会改变部署输入或网络结构，应该在基础奖励和课程可测之后单独对照。

## 部署兼容边界

保留实际 ZIP 中的模型、惯量、关节限位、执行器顺序、默认站姿和控制链路：

- 12 关节，关节顺序为 FR / FL / RR / RL，每腿 hip / thigh / calf。
- hip/thigh kp=20、kd=1；calf kp=40、kd=2。
- 力矩上限 hip/thigh 10 Nm，calf 20 Nm；动作缩放 0.25。
- 默认角度 0 / 0 / 0.2，出生高度 0.31 m；仿真 dt=0.005、decimation=4，即 50 Hz 控制。
- 既有 actor 观测项、顺序、噪声、history_length=1 保留。

**actor 接口一致不等于旧 PPO checkpoint 可以整体恢复。** V4 critic 输入维度和含义改变，先从头训练，不要直接 `--agent.resume` 加载旧 blind checkpoint。若以后做 actor-only warm start，必须连同 actor 的观测归一化统计一起处理，并重建 critic 和 optimizer。

ZIP 中能找到 LgDog 的 Python sim2sim，但未找到 LgDog 专用的实际硬件通信控制程序。这里能核对模型与 sim2sim 的接口，不能据此宣称已核验真机电机映射、通信延迟和部署程序全部行为。

## 运行与验证

在原 Linux 训练环境里把修改包覆盖到 ZIP 对应版本的项目根目录，或者使用这个完整工作副本。继续使用原项目的 mjlab 1.2.0 / MuJoCo 3.5.0 / mujoco-warp 3.5.0 / warp-lang 1.12.0 环境。

如果切换到了新的工作目录，先在该目录执行 `python -m pip install -e . --no-deps`，让 Python 指向这份源码；不要继续引用之前安装的另一份同名 `src` 包。

```bash
# 数值逻辑测试，不需要 GPU
python -m unittest discover -s tests -v

# 配置隔离、actor 一致性、真实模型编译和 CPU 射线检查
python scripts/check_lgdog_terrain_v4.py

# 在训练机器上执行完整 manager / GPU 射线 / reset 冒烟检查
python scripts/check_lgdog_terrain_v4.py --device cuda:0

# 先短跑，检查观测维度、奖励有限性、射线命中和资源占用
python scripts/train.py LgDog1-Rough-Blind-TerrainV4 --env.scene.num-envs=64 --agent.max-iterations=2 --agent.logger=tensorboard --enable-nan-guard True

# 短跑通过后正式训练；按实际显存调整环境数
python scripts/train.py LgDog1-Rough-Blind-TerrainV4 --env.scene.num-envs=2048 --agent.max-iterations=16000 --agent.logger=tensorboard --agent.run-name=terrain_v4_seed1 --enable-nan-guard True
```

新实验目录为 `logs/rsl_rl/lgdog1_velocity_blind_terrain_v4/`。这一轮优先与 `LgDog1-Rough-Blind-Gentle` 做对照。至少使用 3 个随机种子，并固定评估指令和地形，分别报告站立漂移、平地速度误差、5 cm 上下台阶通过率、坡面成功率、摔倒率和接触冲击；不要只比较训练奖励或平均地形等级。

既有 sim2sim 可以显式加载新导出的 ONNX：

```bash
python scripts/sim2sim.py LgDog1-Rough-Blind-TerrainV4 --onnx=<新实验目录>/policy.onnx --duration 20
python scripts/sim2sim.py LgDog1-Rough-Blind-TerrainV4 --onnx=<新实验目录>/policy.onnx --vx 0.4 --duration 20
```

在 sim2sim 的 stairs 地图测试时先选择 5 cm 车道；原地图还包含 10/15/20 cm 台阶，V4 的 0–5 cm 训练范围不代表具备这些高度的通过能力。

## 文件与验证记录

- `src/tasks/velocity/config/lgdog1/blind_terrain_env_cfg.py`：V4 配置及地形射线。
- `src/tasks/velocity/config/lgdog1/blind_mdp.py`：奖励、课程、指标、持续碰撞判定。
- `src/tasks/velocity/config/lgdog1/__init__.py`：新增任务注册。
- `tests/test_lgdog_blind_mdp.py`：12 项数值回归测试。
- `scripts/check_lgdog_terrain_v4.py`：真实模型/配置/可选 GPU 冒烟检查。

已完成：12 项 CPU 数值测试、Python 语法检查、原 actor 配置一致性检查、模型编译、真实 MuJoCo 地形射线检查、完整环境的 10 个控制步及局部 reset（CPU），以及新任务训练 CLI 参数解析。实测 actor 为 47 维、critic 为 261 维；模型质量 10.055 kg。最终验证记录见 `validation_results.txt`。

另已跑通 2 个 CPU 环境、24 步 rollout、1 次 PPO 迭代（1 个 epoch、2 个 minibatch）的最小训练，以及 ONNX 导出、模型结构检查和 `[1,47] → [1,12]` 的 CPU 参考推理。射线有效率在该冒烟运行中为 1.0。冒烟权重仅保存在 `E:\mjlab\work\lgdog_v4_ppo_smoke`，不随补丁分发，也不是可部署的训练成果。

本机使用单独创建的 Python 3.12 验证环境 `E:\mjlab\work\lgdog_validation`，不改变系统 Python 和原训练环境。Warp 未检测到可用的 NVIDIA CUDA 驱动，因此这里的运行验证使用 CPU。本次没有完成新策略的长期训练或真机试验，不能把代码修改称为地形通过率已经提升。
