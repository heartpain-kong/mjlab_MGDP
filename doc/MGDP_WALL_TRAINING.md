# MGDP 统一八档课程 V5（2026-11-15）

当前任务 `LgDog1-Race-MGDP-WallTraining`，工作台名称 **MGDP · 比赛尺寸强化课程**，别名 `race_mgdp_wall`，新实验目录 `logs/rsl_rl/lgdog1_velocity_race_mgdp_wall_v5/`。尺寸、落脚和身体投影规则沿用 [课程V4](MGDP_COURSE_V4.md)，新增 [V5高墙防刷分](MGDP_WALL_PROGRESS_GUARD.md)。覆盖代码后重启工作台，使任务说明与日志目录更新为V5。

共 10 类地形，等比例分配，默认 **8×20** 地图、至少 10 列，每区块 **12×8 m**。同一 `gap_steps` 区块连续放置“上楼梯→过沟→下坡→80 cm 平地→上坡→过沟→下楼梯”，不再增加 `gap_stairs` 地形列。两段在同一级使用相同的平台高度和沟宽。回合时限改为 120 秒，路线指令每 121 秒重采样，避免长路线中途重置航点。

原任务 `LgDog1-Race-MGDP` 保留旧赛道、旧梅花桩规则和旧路线，供 V7 历史复现。新的规则只接入统一八档课程。

工作台现在可直接编辑高墙相关规则，入口及生效范围见 [高墙规则工作台](MGDP_WALL_WORKBENCH.md)。下文数值是默认值；实际启动使用已保存的覆盖设置。

## 工作台感知预训练入口

强化课程使用独立感知文件`logs/mgdp/perception/mgdp_wall_v5.pt`，预训练默认64环境、8×10地图、批64。原Race任务的`mgdp_race_v3.pt`保留。预训练脚本必须支持`LgDog1-Race-MGDP-WallTraining`；旧版只接受两个任务时会参数解析失败、退出码2。

Linux覆盖、重启工作台和完整命令见 [预训练入口修复](MGDP_WALL_PRETRAIN_FIX.md)。工作台新策略、就绪检查和CLI输出已统一；已有策略续训仍恢复其原配套感知。

## 已接入的训练行为

1. **无进展失败**：有 XY 移动指令时，4 秒内没有新增累计 5 cm 的最远路线进展，就结束回合。前后晃动不能刷新；局部重置只清空自己的计时。`no_passage_progress` 按失败终止处理。绕桩对齐朝向时 XY 指令为零，不累计这个移动停滞计时。
2. **高墙八阶段**：固定 8 行、难度倍率 1，初始等级 0；到达完整出口且平均速度误差 ≤0.25 m/s 才升行，失败降行。

   | 训练行 | 墙高 | 训练墙数 |
   |---:|---:|---:|
   | 1 | 10 cm | 1 |
   | 2 | 12 cm | 1 |
   | 3 | 15 cm | 1 |
   | 4 | 18 cm | 2 |
   | 5 | 24 cm | 2 |
   | 6 | 30 cm | 3 |
   | 7 | 35 cm | 3 |
   | 8 | 40 cm | 3 |

   墙厚 8 cm，墙位相对出生点 1.3、2.7、4.1 m。播放验收在每一级都放三道墙；低阶段简化墙数的训练通过率不能代替三墙验收。
3. **每墙一次奖励**：四脚均越过墙后缘 3 cm、根部越过墙后缘 35 cm，且没有出路线或同时失败，才给该墙 +2；每回合每道墙一次。出口成功 +5 和原进展奖励保留。
4. **墙附近摆腿净空**：仅高墙列、墙前 25 cm 至墙后 8 cm 的向前摆动脚，计算低于墙顶 +3 cm 的代价，权重 −0.2。
5. **梅花桩落脚**：第6档（n−2档）固定比赛尺寸：桩面20×20 cm、桩高10 cm、横向净距20 cm、同排纵向净距40 cm；第7档18×18 cm/高12 cm，第8档16×16 cm/高15 cm，前5档逐级过渡。所有档左右固定错位 30 cm。桩区局部 X∈(2.2,5.9) m 内，任意足端出现承重地面接触即失败；入口、出口地面允许使用。接触检查覆盖每个控制步的全部四个物理子步，要求有效接触、力 ≥1 N、接触法线 |Z|>0.7 且局部接触高度 ≤8 mm。保留地板，使用规则阻止踩底板绕过桩区。
6. **绕桩身体投影**：使用真实躯干碰撞盒的水平投影（包括俯仰与横滚），与桩边至少保留 3 cm。新路线 16 个航点、约 16.701 m；绕桩中心走廊半宽 8 cm、到点半径 6 cm，转向误差大于 0.12 rad 时先原地对齐再平移。其他地形仍沿用原走廊和到点参数。

高墙列的基础正奖励已按新增最远进展门控；靠下一道未完成墙75cm内停留，宽限0.75秒后1.5秒内递增至每秒扣2分。每墙一次奖励和4秒停滞失败保留；其他地形基础奖励不变。

冻结感知编码器、FP32、PPO 熵系数和现有速度课程沿用。速度仍从 0.25–0.35 m/s 开始，本次没有打开固定 1 m/s 模式。所有比赛障碍第6档均对齐参考尺寸，第7/8档强化。几何可行和工程检查通过，不等于策略已学会新规则或40cm高墙。轮胎和砖木坑几何增加，GPU峰值显存须重测。

## 使用旧模型开始新课程

V4及更早模型的奖励、地形或终止契约与当前V5不同，必须用 `--warm-start-checkpoint` 迁移 actor、观测归一化和动作标准差。匹配的冻结感知、历史长度、精度及控制步长仍严格检查；critic、优化器、轮次、课程统计重新开始。不要对旧实验使用完整续训恢复旧难度。

上传补丁、同名校验文件、策略及对应感知文件到 `/root`：

```bash
cd /root
sha256sum -c mgdp_wall_workbench_20261115.zip.sha256 && unzip -o mgdp_wall_workbench_20261115.zip -d /root
cd /root/unitree_rl_mjlab_lgdog
conda activate /root/lgdog
source server_env.sh
lgdog_server_env "$PWD"
unset CUDA_LAUNCH_BLOCKING

MODEL=/root/model_18200.pt
PERCEPTION=/root/perception_7da635cf4a664d5ede1a876944c6af3a79d3b8ce746ab13e6306eafa0c65e3bb.pt

# GPU 工程检查：通过后再开正式实验。
python -u -m scripts.train LgDog1-Race-MGDP-WallTraining \
  --env.scene.num-envs=64 --terrain-num-rows=8 --terrain-num-cols=20 \
  --agent.max-iterations=3 --agent.run-name=wall_guard_smoke \
  --warm-start-checkpoint="$MODEL" --perception-checkpoint="$PERCEPTION"

# 正式实验；新地形和传感器的实际峰值显存需在 GPU 上观察。
python -u -m scripts.train LgDog1-Race-MGDP-WallTraining \
  --env.scene.num-envs=8192 --terrain-num-rows=8 --terrain-num-cols=20 \
  --agent.max-iterations=3000 --agent.run-name=wall_progress_guard \
  --warm-start-checkpoint="$MODEL" --perception-checkpoint="$PERCEPTION"
```

本次统一八档修正后的新 V5 检查点后续使用 `--agent.resume=True --resume-checkpoint=...` 完整续训，不再传 warm-start；两者互斥。工作台的“续训”同样只用于契约一致的实验。布局保持 8 行、难度倍率 1，默认 20 列，至少 10 列。

## 验收

观察 `Episode_Termination/no_passage_progress`、`stone_ground_contact`、`slalom_body_projection`，以及越墙奖励、摆腿净空和各地形通过率。当前课程窗口通过率不能代替固定难度、多次重复的独立验收。

新任务播放使用相同的梅花桩、身体投影规则及组合沟槽，高墙每级三道。旧 V7 复现使用原任务。应分别验证八级高墙、两段过沟、桩顶落脚、绕桩身体间距、实际速度与能力退化；先看专项效果，再决定是否重训感知或调整探索。

本地已完成奖励回归、零动作真实物理对照，以及旧第18200轮模型向高墙专项布局的CPU PPO迁移、保存、ONNX/JIT导出和完整续训。GPU 峰值显存、长期收敛和真实通过能力尚未验证。

D435i下俯20°使用新增独立入口，配置与覆盖步骤见[下俯20°配置](MGDP_D435I_DOWN20.md)。原相机入口保留。
