# LgDog：借鉴 HIMLoco 的盲走实验

> 2026-10-28更新：本文以下网络图和验证结果记录的是历史V7六帧版本。当前代码的HIM新训练默认改为3帧/141维；旧模型回放保留原6帧/282维。当前使用请先看 [HIM_HISTORY_3.md](HIM_HISTORY_3.md)，不要将下文历史维度直接用于新模型。

本文记录原 V7 温和地形 HIM。标准 Rough 地形、删除 actor/critic 机身高度扫描的新任务 `LgDog1-Rough-Blind-HIM-Standard` 见 [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md)，其奖励配置为 `him_standard`。

奖励调参入口已集中到 `src/tasks/velocity/config/lgdog1/reward_cfg.py`，HIM 对应 `REWARD_PROFILES["him"]`，操作说明见 [REWARDS.md](REWARDS.md)。

已在 `E:\mjlab\unitree_rl_mjlab_lgdog` 新增任务 **`LgDog1-Rough-Blind-HIM`**。它在上一版 TerrainV4 的相同地形、奖励、课程和本体参数上增加历史估计网络，便于比较网络结构带来的变化。

这是适配 mjlab 1.2.0 / RSL-RL 5.0.1 的 **HIM-inspired 实现**，不是官方训练结果的复现，也不是直接搬用官方 Go1/Aliengo 权重。

## 借鉴了哪些核心部分

参考本地 `E:\mjlab\HIMLoco`，提交 `ef289acaa62795009363b7b819c9186690630441`：`him_estimator.py`、`him_actor_critic.py`、`him_ppo.py`、`him_on_policy_runner.py`，以及历史观测构造代码。

方法来自 Long 等人的 [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/abs/2312.11460)，参考仓库为 [HIMLoco](https://github.com/InternRobotics/HIMLoco)。上游根目录与 rsl_rl 子目录分别带有许可证，本地均保留；本改动没有把上游旧 rsl_rl 安装进当前训练环境。

网络的数据流是：

```text
6 帧本体感觉 [6,47]
        │
        ├─ 历史编码器 282 → 128 → 64 → 19
        │                         ├─ 3 维机体速度预测
        │                         └─ 16 维归一化隐变量
        │
当前 47 维观测 + 速度 3 维 + 隐变量 16 维
        │
        └─ actor 66 → 512 → 256 → 128 → 12 关节动作

训练专用：下一步干净的 47 维观测 → target encoder → 16 维表示
          真实下一步机体速度 → 速度 MSE 监督
          source/target 表示 → 32 个原型 + Sinkhorn → swapped loss

critic：保留 TerrainV4 的 261 维特权输入。
```

隐变量学习的是有助于控制的系统响应，不应把其中某一维直接解释成“台阶高度”或“摩擦系数”。actor 和导出的 ONNX 都没有读取真实速度、高度扫描或其他训练标签。

## 与官方代码的差异

| 项目 | 本版本选择及原因 |
|---|---|
| 单帧观测 | 保留 LgDog 的 47 维，包含原有两维步态相位；官方常用 45 维。没有直接硬切到 45 维破坏本工程的观测/奖励约定 |
| 历史 | 6 帧，覆盖当前时刻至前 0.10 秒，50 Hz；按最旧到最新排列，当前帧在最后。官方对应代码按最新在前拼接，两种约定不能混用 |
| reset | 用第一帧回填全部历史槽，与 mjlab CircularBuffer 的行为一致；每个环境分别清理 |
| 归一化 | 共享一套单帧 47 维统计，按最新帧更新；历史各时间位置使用相同统计，并导出进 ONNX |
| 速度标签 | 仿真中的下一步机体坐标线速度，单位 m/s；与上游使用 successor velocity 的思路一致 |
| 下一状态目标 | 单独定义干净的 `him_target`、`him_velocity` 观测组，避免依赖 critic 中易变化的硬编码切片 |
| 终止样本 | mjlab 返回自动重置后的观测，因此掉落和超时 transitions 都从估计器学习中排除；这些样本仍正常用于 PPO。上游另有终止观测传递机制 |
| 更新顺序 | PPO 各 epoch 内冻结估计器，之后用同一 rollout 更新估计器；与上游 minibatch 内交错更新不同，需用长训练比较效果 |
| 学习率 | PPO 保留原自适应调度；估计器独立 Adam，初始 1e-3，每次 rollout 默认训练 1 epoch |
| 对比学习 | 保留 16 维 latent、32 原型、温度 3.0、Sinkhorn epsilon 0.05 / 3 次迭代，以及对 batch/prototype 两轴求平均的 loss 尺度。Sinkhorn 用 log-space 实现 |

PPO 优化器明确排除了估计器参数；actor 对速度/latent 使用 stop-gradient。估计器的 encoder、target encoder 和 prototype 仅由辅助目标训练。checkpoint 同时保存这些参数、观测归一化状态及估计器优化器状态。

当前实现只支持单设备 PPO，遇到多 GPU、RND 或 symmetry 配置会明确报错，避免未同步的辅助网络静默运行。

## 训练

使用原 Linux GPU 训练环境。从新工作目录运行时，先把 editable 安装指向这份源码：

```bash
python -m pip install -e . --no-deps
python scripts/list_envs.py --keyword LgDog1

# 小规模短跑，检查实际 GPU 路径
python scripts/train.py LgDog1-Rough-Blind-HIM --env.scene.num-envs=64 --agent.max-iterations=2 --agent.logger=tensorboard --enable-nan-guard True

# 正式训练；环境数按显存调整
python scripts/train.py LgDog1-Rough-Blind-HIM --env.scene.num-envs=2048 --agent.max-iterations=16000 --agent.logger=tensorboard --agent.seed=42 --agent.run-name=him_v1_seed42 --enable-nan-guard True
```

输出目录：`logs/rsl_rl/lgdog1_velocity_blind_him_v1/`。需要从头训练，旧 TerrainV4 / Blind / Gentle checkpoint 和官方 HIMLoco checkpoint 都不能直接加载。

重点观察 `him_velocity_mse`、`him_swap`、`him_valid_fraction`、`him_updates`，以及原有速度误差、地形进度、接触和终止指标。RSL-RL logger 会把这些辅助统计统一放在 loss 日志下；其中 valid_fraction 是比例、updates 是次数，不是损失。`him_velocity_mse` 是三维速度分量的平均平方误差，单位为 (m/s)²。

不要只看 swapped loss 下降就认为地形能力增强。建议以 TerrainV4 为基线，在相同种子、训练步数、指令和测试地形下比较；后续再加入“只堆历史、不做辅助学习”的消融实验，区分历史本身与 HIM 学习目标的贡献。保留原步态奖励和 phase 是这一轮控制变量的选择，后续可单独比较弱化步态约束。

## 部署和 sim2sim

**新 ONNX 输入是 `[1,282]`，输出仍是 `[1,12]`。不能将 47 维旧部署程序直接接上新模型。**

1. 每 20 ms 构造原有一帧 47 维观测。
2. 保留六帧完整观测，最旧在前、当前在后，按帧展开为 282 维。
3. 启动或 reset 时重复第一帧填满六个槽；上一动作清零，步态计时重置。
4. 直接将原始历史传给 ONNX；模型内部已有归一化和历史编码器，外部不要重复归一化。
5. 输出仍按原 LgDog 关节顺序和动作缩放 0.25 转换，PD、力矩上限和 50 Hz 控制参数保持原值。

实现见 `src/policy_history.py`。`scripts/sim2sim.py` 已支持新旧两种接口，HIM 会检查 ONNX 的 history_length、frame_dim、排列顺序、回填规则和控制周期 metadata；不匹配就拒绝运行。也修正了该脚本原先更新键盘指令后仍使用上一步观测、以及手动 reset 后相位继续沿用总步数的问题，并在取观测前刷新 MuJoCo 的派生量和传感器。

```bash
python scripts/sim2sim.py LgDog1-Rough-Blind-HIM --onnx=<新实验目录>/policy.onnx --duration 20 --no-keyboard
python scripts/sim2sim.py LgDog1-Rough-Blind-HIM --onnx=<新实验目录>/policy.onnx --vx 0.4 --duration 20 --no-keyboard
```

ZIP 未包含 LgDog 真机硬件通信控制程序，所以本次同步到了 Python sim2sim 与可复用的历史缓存模块。真机控制程序也必须实现同一历史协议，不能把这里的 sim2sim 验证称为已经完成硬件部署。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/check_lgdog_him.py --device cpu --output logs/him_smoke
# 在 GPU 训练机器上还应执行一次：
python scripts/check_lgdog_him.py --device cuda:0 --output logs/him_gpu_smoke
```

本次在独立 Python 3.12 环境中完成：

- 23 项测试：包括原 TerrainV4 的 12 项，以及历史顺序/reset、privileged 隔离、梯度隔离、估计器真实更新、终止样本屏蔽、全终止批次、归一化导出一致性、checkpoint/优化器恢复等 11 项新测试。
- 真实 mjlab CPU 环境的 `[N,6,47]` actor、261 维 critic、历史推进及局部 reset 检查。
- 4 环境 × 每轮 12 控制步 × 2 轮 PPO + HIM 更新；确认历史 encoder 参数发生变化。
- ONNX 结构检查、ONNX Runtime 与 PyTorch 数值对齐，包含非零随机输入和非平凡归一化状态。
- 从本次生成的 checkpoint 恢复实际 runner，验证 actor 输出及估计器优化器存在。
- 新 HIM 282 维和旧 TerrainV4 47 维模型分别通过普通 MuJoCo sim2sim 的 10 个控制步。
- 新任务训练 CLI 参数解析通过。

这些是功能与接口验证，不是长期收敛或地形通过率验证。CPU sim2sim 的 0.2 秒 smoke 也不代表站立/行走能力。冒烟权重只保存在 `E:\mjlab\work\lgdog_him_smoke`，不作为训练成果交付。本机没有可用 CUDA 驱动，GPU 运行、长期训练、多种子地形评估和真机实验尚未执行。

## 修改位置

- `src/tasks/velocity/config/lgdog1/him_env_cfg.py`：HIM 观测与学习配置。
- `src/tasks/velocity/rl/him_estimator.py`：速度预测、隐变量、target/prototype 学习。
- `src/tasks/velocity/rl/him.py`：RSL-RL actor、PPO 辅助学习、checkpoint、ONNX metadata。
- `src/policy_history.py`：轻量 NumPy 部署历史缓存。
- `scripts/sim2sim.py`：新旧策略接口与 reset/指令同步。
- `scripts/check_lgdog_him.py`、`tests/test_lgdog_him.py`：集成和回归验证。

修改包是相对你提供的 ZIP 的累计补丁，包含所需 TerrainV4 文件；使用时把内容合并到该版本工程根目录，无需安装上游 HIMLoco 的旧 rsl_rl。
