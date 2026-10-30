# LgDog 盲走：安装、训练与使用

需要“标准崎岖地形 + HIM 历史估计 + 去掉机身地形扫描”时，使用新任务 `LgDog1-Rough-Blind-HIM-Standard`，具体命令见 [标准地形 HIM 指南](LGDOG_HIM_STANDARD.md)。下面原有 HIM 命令对应 V7 温和地形任务。

需要修改奖励时，先看 [REWARDS.md](REWARDS.md)。LgDog 的统一调参入口是 `src/tasks/velocity/config/lgdog1/reward_cfg.py`；`python scripts/list_rewards.py LgDog1-Rough-Blind-HIM --term foot_clearance` 可查看实际参数和函数位置。

这份指南说明如何使用本工程中的 LgDog1 配置，训练基于 HIM 思路的盲走策略，并在 MuJoCo 中检查导出的模型。项目代码可以先在 Windows 上查看和编辑；mjlab 的 GPU 训练建议在安装好 NVIDIA 驱动和 CUDA 环境的 Linux 训练机上进行。

## 这份工程包含什么

- `LgDog1-Flat`、`LgDog1-Rough`：LgDog 的平地与崎岖地形速度跟踪任务。
- `LgDog1-Rough-Blind`：不向 actor 提供地形高度扫描的盲走基线。
- `LgDog1-Rough-Blind-TerrainV4`：改进地形课程和接触终止条件的盲走版本。
- `LgDog1-Rough-Blind-HIM`：在 TerrainV4 训练分布上增加历史观测与 HIM 风格状态估计的实验版本。
- `LgDog1-Rough-Blind-HIM-Standard`：标准 Rough 地形 + HIM，actor 和 critic 均不读取机身高度扫描。

HIM版本保留47维单帧观测和12维动作。2026-10-28起，新训练默认接收最近3帧、合计141维历史；旧6帧模型仍需282维。actor不读取地形扫描或训练用速度标签。迁移和兼容说明见 [HIM_HISTORY_3.md](HIM_HISTORY_3.md)。

本实现借鉴 HIMLoco 的训练思路，适配当前 mjlab / RSL-RL 工程，不是论文结果的复现。估计器使用速度监督和原型对比目标学习历史特征。HIM 任务需要从头训练，不能加载旧的 47 维策略权重。

## 1. 准备训练环境

在 Linux GPU 训练机上获取这份修改后的工程目录 `unitree_rl_mjlab_lgdog`，进入工程根目录。不要把原始的 `unitree_rl_mjlab` 目录误当成修改版。依照 [`doc/setup_zh.md`](setup_zh.md) 建好项目依赖与 NVIDIA 运行环境后，执行：

```bash
cd /path/to/unitree_rl_mjlab_lgdog
python -m pip install -e . --no-deps
python scripts/list_envs.py --keyword LgDog1
```

确认任务列表中有 `LgDog1-Rough-Blind-HIM`。`--no-deps` 只用于把当前工程注册为 editable 包；它不会安装项目依赖。第一次配置新机器时，请按安装文档安装依赖，再运行上面命令。

## 2. 先运行短程检查

先用较少并行环境跑两轮，确认新机器上的 CUDA、MuJoCo、配置和日志目录都正常：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM \
  --env.scene.num-envs=64 \
  --agent.max-iterations=2 \
  --agent.logger=tensorboard \
  --agent.run-name=SMOKE_him \
  --enable-nan-guard True
```

检查程序正常退出，并生成 `logs/rsl_rl/lgdog1_velocity_blind_him_v1/` 下的实验目录、checkpoint 和 `policy.onnx`。正式训练前先用 `nvidia-smi` 看可用显存；环境数应按显存调整，64 是短程检查用的设置，不保证适用于所有 GPU。

## 3. 开始正式训练

> ★ **训练轮次命名约定（2026-10-16 起）**：所有训练按 `V1 / V2 / V3 …` 编号，
> `--agent.run-name` 一律写成 `V<n>_<短描述>`，例如 `V8_him_seed42`。
> 完整轮次表、每轮结论、显存上限、开新轮的标准命令 → 见 **[TRAINING_V.md](TRAINING_V.md)**。
> 本指南下面的 `V8` / `V9` 是示例编号，实际开训前请先查登记表取下一个空号。



下面是一个示例命令。建议先用较小环境数启动并观察显存和吞吐，再决定是否增加并行环境：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --agent.run-name=V8_him_seed42 \
  --enable-nan-guard True
```

结果保存在 `logs/rsl_rl/lgdog1_velocity_blind_him_v1/` 下，具体实验目录名由启动时间和 run name 组成。定期保存的 `model_<iteration>.pt` 用于恢复训练；同目录下的 `policy.onnx` 是相应 checkpoint 导出的推理策略。训练期间可以另开终端运行：

```bash
tensorboard --logdir logs/rsl_rl/lgdog1_velocity_blind_him_v1
```

重点同时看任务表现和辅助学习指标：`him_velocity_mse`、`him_swap`、`him_valid_fraction` 和 `him_updates`。辅助损失降低本身不能证明跨越能力变好；要与 TerrainV4 基线使用相同随机种子、训练步数、指令范围和测试地形做比较。

### TerrainV4 基线命令

用同样的训练环境数、迭代数和 seed 训练基线，只替换任务名与 run name：

```bash
python scripts/train.py LgDog1-Rough-Blind-TerrainV4 \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --agent.run-name=V9_terrainv4_seed42 \
  --enable-nan-guard True
```

这样能初步比较 HIM 历史估计结构的增益。若要评估稳定性，应对多个 seed 重复训练，并在固定测试地形上统计通过率、摔倒率、速度误差和接触情况。

## 4. 在普通 MuJoCo 中检查 ONNX

训练后，先将 `EXP_DIR` 设为实际生成的实验目录，再运行 sim2sim：

```bash
EXP_DIR="logs/rsl_rl/lgdog1_velocity_blind_him_v1/你的实际实验目录"
python scripts/sim2sim.py LgDog1-Rough-Blind-HIM \
  --onnx="$EXP_DIR/policy.onnx" \
  --duration 20 --no-keyboard
```

也可以指定前进速度：

```bash
python scripts/sim2sim.py LgDog1-Rough-Blind-HIM \
  --onnx="$EXP_DIR/policy.onnx" \
  --vx 0.4 --duration 20 --no-keyboard
```

新HIM ONNX输入为 `[1,141]`，输出为 `[1,12]`。输入由每帧47维、按“最旧到最新”排列的3帧构成；旧模型继续使用其metadata中记录的6帧/282维。模型内部已有观测归一化，不要重复归一化。启动时重复第一帧填充缓存，每20ms更新一次。

`scripts/sim2sim.py`检查ONNX记录的历史长度、帧维度、顺序、reset规则与控制周期。支持141维新HIM、282维旧HIM、47维单帧PPO，各自按模型协议构造缓存。

## 5. 真机部署前需要做的工作

工程没有LgDog真机通信控制程序。真机端需要47维观测、按模型metadata决定长度的历史缓存（新模型3帧）、reset重复首帧、20ms更新周期，以及相同关节顺序和动作缩放。参考 [`src/policy_history.py`](../src/policy_history.py) 和 [`scripts/sim2sim.py`](../scripts/sim2sim.py)。CPU验证不代表完成实机部署。

新HIM模型的141维输入与旧HIM的282维、单帧PPO的47维不同；更换模型时必须匹配输入协议。

## 6. 常见问题

**任务名找不到**：在工程根目录重新运行 `python -m pip install -e . --no-deps`，再执行 `python scripts/list_envs.py --keyword LgDog1`。确认运行的 Python 环境与安装包环境一致。

**CUDA 初始化失败或显存不足**：先在目标 Python 环境检查 PyTorch / CUDA 与驱动是否匹配，并用 `nvidia-smi` 检查剩余显存。把 `--env.scene.num-envs` 降低后再启动。

**旧模型加载失败**：旧6帧HIM不能直接resume到新3帧训练，需新开实验；Play会按原checkpoint恢复帧数。TerrainV4 checkpoint也不能作为HIM恢复。

**ONNX输入尺寸错误**：核对模型shape和metadata。新3帧HIM为 `[1,141]`，旧6帧HIM为 `[1,282]`，单帧PPO为 `[1,47]`。

## 相关文档

- [HIM 方法、网络结构与实现差异](LGDOG_HIM.md)
- [TerrainV4 地形训练改动](LGDOG_TERRAIN_V4.md)
- [验证记录与当前未验证事项](../archive/HIM_VALIDATION.txt)
- [项目架构说明](ARCHITECTURE.md)
- [训练环境安装说明](setup_zh.md)
