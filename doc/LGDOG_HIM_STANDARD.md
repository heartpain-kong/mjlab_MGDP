# LgDog：mjlab + HIM 标准崎岖地形盲走

任务：`LgDog1-Rough-Blind-HIM-Standard`。

这是将当前 `unitree_rl_mjlab` 工程与 HIM 状态估计结合的新任务：保留 LgDog 本体、MuJoCo 仿真、任务管理器、奖励调参入口、PPO、ONNX 与 sim2sim 链路，使用 HIM 的历史编码器、速度预测、隐变量和辅助学习。地形使用标准 Rough 配置，不做 Gentle 难度缩减。

HIM 是策略中的状态估计与学习方法，地形生成、难度课程和奖励仍由 mjlab 配置负责。两部分的组合需要通过训练来验证表现，不能仅因更换网络就推断地形通过率提高。

## 与现有 V7 的区别

| 项目 | 原任务 `LgDog1-Rough-Blind-HIM` | 新任务 `LgDog1-Rough-Blind-HIM-Standard` |
|---|---|---|
| 地形 | Gentle 温和地形 | 标准 Rough 地形 |
| 平地比例 | 40% | 20% |
| 上/下台阶比例 | 各 15% | 各 20% |
| 台阶高度范围 | 0–0.05 m | 0–0.10 m |
| 坡度参数范围 | 0–0.5 | 0–1.0（坡度比值） |
| 随机起伏范围 | 0.02–0.06 m | 0.02–0.10 m |
| 波浪幅值范围 | 0–0.10 m | 0–0.20 m |
| actor 输入 | 6×47 帧历史 | 6×47 帧历史 |
| actor 机身地形扫描 | 无 | 无 |
| critic 机身地形扫描 | 187 维 | 无 |
| critic 总维度 | 261 | 74 |
| 奖励配置名 | `him` | `him_standard` |
| 实验目录 | `lgdog1_velocity_blind_him_v1` | `lgdog1_velocity_blind_him_standard_v1` |

地形难度通过课程逐级提升，因此台阶等范围的上限不代表开局全部使用最大难度。新任务从初始等级 0–1 开始，但没有缩小标准地形的难度范围。

新任务从 actor、critic、干净的辅助目标观测中删除机身高度扫描，场景中也不创建 `terrain_scan`。训练场景仍有四条足下垂直射线，供相对离地高度奖励与 critic 的四维足端高度使用；这些是仿真中的训练数据，不进入部署 actor。真机无需高度图、雷达或足下射线传感器。

## 两个模块各负责什么

| 模块 | 本任务中的职责 |
|---|---|
| `unitree_rl_mjlab` 工程 | LgDog 几何、质量、关节顺序、PD、动作缩放、仿真步进、观测、日志和部署框架 |
| HIMActor / HIMEstimator | 六帧历史 → 三维速度预测和 16 维隐变量；当前观测与估计特征 → 十二维动作 |
| HIMPPO | 正常 PPO 更新，以及独立的速度监督、原型对比辅助更新 |
| 地形训练配置 | 标准 Rough 地形，按累计指令距离和跟踪进度调难度，区分机身碰撞与短暂腿部擦碰 |
| `reward_cfg.py` | 18 个奖励项的权重、参数及任务间独立覆盖 |

本体 PD、力矩上限、动作缩放 0.25、仿真 dt=0.005 s 和控制周期 0.02 s 沿用当前 LgDog 配置。新任务保留相对足下地面的抬脚代价，默认目标 0.10 m；后续调目标高度应在 `him_standard` 中明确记录。

训练指令沿用现有 HIM 的范围：vx 为 (-0.4,0.8)、vy 为 ±0.3 m/s、yaw rate 为 ±0.6 rad/s；指令在 3–5 s 之间随机重采样，减少固定周期切换造成的动作尖峰。地形课程还要求完成回合的平均 XY 速度误差不超过 0.8 m/s，避免策略只靠走过距离提前进入更难地形。

当前稳定性训练默认值为：归一化动作裁剪到 ±1.0、PPO rollout 为每个环境 50 步（50 Hz 下约 1 秒）、HIM 辅助估计器每轮 5 个 epoch。它们会写入每个 run 的 `params/agent.yaml`，便于和旧的 24/48 步、无动作裁剪实验对照。

## 使用方法

在 Linux GPU 训练机的工程根目录、已安装依赖的环境中执行：

```bash
python -m pip install -e . --no-deps
python scripts/list_envs.py --keyword LgDog1
python scripts/list_rewards.py LgDog1-Rough-Blind-HIM-Standard --term foot_clearance

# 先检查目标机器的实际 GPU 路径
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=64 --agent.max-iterations=2 \
  --agent.logger=tensorboard --agent.run-name=him_standard_smoke \
  --enable-nan-guard True

# 新实验示例，环境数按实际显存调整
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=1024 --agent.max-iterations=16000 \
  --agent.logger=tensorboard --agent.seed=42 \
  --agent.run-name=V9_stability_seed42 --enable-nan-guard True
```

标准 HIM 默认直接使用 Unitree/mjlab 的 `ROUGH_TERRAINS_CFG`，10 行表示难度等级、20 列表示地形类型列，
初始环境仍只从 0–1 级开始。标准 HIM 和普通 PPO 使用同一套 LgDog 地形编辑配置；该配置保留原生生成器，
仅把台阶设为 5–15 cm、坡度梯度设为 0–0.4。LgDog 的隔离编辑入口是
`src/tasks/velocity/config/lgdog1/terrain_cfg.py`，不需要修改训练环境里安装的第三方 `mjlab` 文件。
速度课程从最终指令范围的 50% 开始，线速度跟踪分数达到 0.70 并连续两次检查后每次扩张 0.05，
连续三次低于 0.55 才回退 0.025，直到达到最终范围，偏航角速度保持
配置范围；地形升级还要
同时满足命令完成度、速度误差、存活时间和离开出生平台等条件。面板或命令行可以用正数直接改
负向下限和正向上限：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --velocity-min-abs=0.4 --velocity-max-abs=0.8
```

这套课程采用本机 `E:\mjlab\HIMLoco\legged_gym\legged_gym\envs\base\legged_robot.py`
中的“跟踪质量达标后扩大命令范围”规则，以及
`E:\mjlab\legged_wbc_mjlab-main\legged_wbc_mjlab-main\src\tasks\locomotion\ri_5036_him\mdp\curriculums.py`
中的完整回合聚合方式。地形升级保留 `TransCurriculum` 文档强调的分层难度空间，但使用本项目已有的
路径进度、命令完成度、平均速度误差、存活时长和出生平台外暴露时间作为门控。论文中的 LP 教师和
多维参数自动采样没有伪装成已实现功能；当前版本先把可复现的 HIMLoco/legged-gym 课程落地。

需要继续扩展地形时可以显式设置行数和最大
难度倍率：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --terrain-num-rows=20 --terrain-difficulty-max=1.0 --agent.seed=42
```

这里的难度倍率作用于地形生成器的 `difficulty_range`，不是直接跳到最高行。Scaling Rough
Terrain、TransCurriculum、TACL 和 PSC 都强调先扩展可探索空间，再由学习进度、成功率或安全
信号选择下一批任务；因此不建议一开始把难度倍率直接提高到 1.5。

训练脚本会扫描当前日志和 `E:\\mjlab\\数据` 归档，实验名省略时自动生成下一个
`V<n>_seed<seed>` 名称。

`V8` 这里只是新实验的命名示例，并不表示已经启动训练。新日志位于 `logs/rsl_rl/lgdog1_velocity_blind_him_standard_v1/`。

迁移到训练机时同步当前工程目录及新增文件。`archive/LgDog_HIM_patch.zip` 是早期 V7 的历史补丁，不包含这个新任务或控制面板。

先从头训练，不要用 `--agent.resume=True` 直接恢复 V7 的完整 checkpoint：新任务的 critic 从 261 维改成了 74 维，训练分布也变了。训练保存 checkpoint 时会同时导出同目录下的 `policy.onnx` 和 `policy_jit.pt`，输入 `[1,282]`，输出 `[1,12]`，历史排列和 reset 协议与已有 HIM 相同。CUDA 训练默认启用 fused Adam；CPU 或不支持该算子的 PyTorch 会自动回退到普通 Adam。速度跟踪、课程进度和指令门控中的热点张量算子已用 TorchScript kernel 脚本化。

```bash
EXP_DIR="logs/rsl_rl/lgdog1_velocity_blind_him_standard_v1/你的实际实验目录"
python scripts/sim2sim.py LgDog1-Rough-Blind-HIM-Standard \
  --onnx="$EXP_DIR/policy.onnx" --vx 0.4 --duration 20 --no-keyboard
```

上面的 sim2sim 默认使用该脚本的平地测试场景，不代表覆盖了标准 Rough 训练地形。固定台阶、坡面等场景应单独测试，并与同场景、同指令下的其他策略比较。

## 修改入口与检查

- 标准 HIM 奖励：`src/tasks/velocity/config/lgdog1/reward_cfg.py` → `REWARD_PROFILES["him_standard"]`。
- 地形/扫描选择：`him_env_cfg.py::lgdog1_him_standard_env_cfg`。
- 接触与课程机制：`blind_terrain_env_cfg.py`、`blind_mdp.py`。
- 历史长度、编码器、PPO：`him_env_cfg.py`、`src/tasks/velocity/rl/him*.py`。
- 原始动作监控：`src/tasks/velocity/rl/action_monitor.py`。训练日志新增
  `Metrics/raw_action_abs_max`、`Metrics/raw_action_abs_mean` 和
  `Metrics/action_saturation_fraction`；它们分别在裁剪前记录动作幅值和达到
  `clip_actions` 上限的比例。
- 本地面板的“动作健康”卡片会自动读取当前任务最新 TensorBoard run，显示原始最大值、
  饱和率、策略标准差和平均奖励；原始最大值 > 3、饱和率 > 5% 或策略标准差 > 3
  时显示告警。sim2sim 默认也使用 `--action-clip 1.0`，并在结束时打印相同的动作统计。
- 新训练默认使用 `learning_rate=3e-4`、`entropy_coef=0`、`init_std=0.5`，
  并对裁剪前超出 ±1 的原始动作加入轻量惩罚；这是为了阻止 Gaussian 策略把熵和
  原始动作幅值持续推高。V11/V11_c 的 checkpoint 需要从头重新训练，不能续训。

建议把 `raw_action_abs_max` 和 `action_saturation_fraction` 与已有的
`Episode_Reward/action_rate_l2` 一起看。后者只反映裁剪后的执行动作，前两者才能发现
策略输出已经发散。对当前 LgDog 默认值，连续多个迭代的饱和率超过 0.05 或原始最大值
超过 3，应暂停该轮并检查 `Policy/mean_std`、entropy、学习率和 checkpoint。

```bash
python -m unittest discover -s tests -v
python scripts/check_lgdog_him.py --task LgDog1-Rough-Blind-HIM-Standard \
  --device cpu --output logs/him_standard_cpu_smoke
# 在 GPU 训练机上再检查相同链路
python scripts/check_lgdog_him.py --task LgDog1-Rough-Blind-HIM-Standard \
  --device cuda:0 --output logs/him_standard_gpu_smoke
```

检查脚本会核对真实环境的观测形状、历史推进和 reset，执行两轮 PPO/HIM 更新，并检查 ONNX Runtime 与 PyTorch 对齐及 checkpoint 恢复。生成的 smoke 权重只用于接口检查，不能作为已训练好的行走策略使用。

后续评估重点是标准台阶/坡面的通过率、摔倒率、速度误差和动作稳定性。旧 V7 使用不同地形和 critic 信息，不能仅凭训练总回报与新任务直接排名。

2026-10-21 的训练归档还发现：课程函数中的 `velocity_error_xy_mps` 是按控制步累加的 m/s，旧实现却除以回合秒数，导致门槛值被放大约 `1 / step_dt`（当前为 50 倍）。这会把约 0.2 m/s 的真实平均误差记录成约 10，并阻止 Standard HIM 升级地形。源码已改为除以控制步数，并加入回归测试；修复前的 `level_mean=0` 不能作为地形能力结论。

## 本次已完成的验证（2026-10-21）

- 45 项 CPU 测试通过，覆盖标准地形分布、扫描删除、HIM 历史输入、独立奖励配置、任务注册与 V7 隔离，以及稳定性训练默认值和课程速度门槛。
- 真实 mjlab CPU 环境确认 actor `[4,6,47]`、critic `[4,74]`、干净目标 `[4,47]`、速度标签 `[4,3]`，历史推进及局部 reset 正确。
- 四环境、每轮十二个控制步、两轮 PPO/HIM 更新完成，估计器参数发生更新。
- ONNX 完整结构检查、ONNX Runtime/PyTorch 数值对齐和 history metadata 校验通过，输出 `[1,12]`。
- 实际 runner 从新 checkpoint 恢复 actor、估计器、归一化状态及辅助优化器通过。
- 新任务训练 CLI 参数解析通过；新导出的 ONNX 在普通 CPU MuJoCo sim2sim 中完成 10 个控制步（0.2 秒）。这是接口冒烟检查，不代表已学会站立、行走或过地形。

本机没有可用 CUDA 驱动，上述验证未包含 GPU 长训练和标准地形通过率评估。日志在 `E:\mjlab\work\lgdog_him_standard_tests.log`、`lgdog_him_standard_smoke.log`、`lgdog_him_standard_cli.log`、`lgdog_him_standard_sim2sim.log`；smoke 模型位于 `E:\mjlab\work\lgdog_him_standard_smoke`，不作为训练成果交付。
