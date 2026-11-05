# 比赛 MGDP：HIMLoco 全强度域随机化

更新：2026-11-05。适用 `race_mgdp` / `LgDog1-Race-MGDP` 的策略训练及比赛感知预训练采集。
按用户选择启用发布代码中的完整扰动范围，不降低外力或推搡强度。

参考官方 [HIMLoco 配置](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot_config.py)
及 [执行代码](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot.py)，
固定提交 `ef289acaa62795009363b7b819c9186690630441`。这里采用发布代码默认启用的项目与范围，
不是论文表格中的另一套范围，也不是迁移 HIM 的网络或奖励。

## 范围与触发时机

| 项目 | 范围 | 抽样/执行时机 |
|---|---|---|
| 机身附加载荷 | 均匀分布 −1～+2 kg | 每个环境初始化一次，回合复位不变 |
| 机身惯量 | 与该环境机身质量同比缩放 | 随附加载荷初始化 |
| 机身质心偏移 | XYZ 各 ±0.05 m | 每个环境初始化一次 |
| 足端切向摩擦 | 0.2～1.25；同机器人四足共用 | 每次复位 |
| Kp 倍率 | 0.9～1.1；同机器人所有关节共用 | 每次复位 |
| Kd 倍率 | 0.9～1.1；与 Kp 独立抽样 | 每次复位 |
| 电机强度倍率 | 0.9～1.1；同机器人所有关节共用 | 每次复位 |
| 初始关节位置 | 名义关节位置 × 0.5～1.5 | 每次复位；各关节独立，速度为零 |
| 动作延迟 | 每个机器人独立抽取 0/1/2/3 个物理子步 | 每个策略步；即 0/5/10/15 ms |
| 外力 | 机身局部坐标系 XYZ 各 −30～+30 N | 每 8 个策略步，下一物理步施加一次 5 ms 脉冲 |
| 推搡 | 世界坐标 XY 速度各均匀设为 −1～+1 m/s | 全局每 16 s；保留 Z 速度及角速度 |

当前物理步长 5 ms，策略控制间隔 20 ms。延迟期间继续使用上一策略步的关节目标，
之后切换到新目标；这是物理子步延迟。外力每 0.16 s 触发，±30 N 是每个分量的范围。
MuJoCo 外力会保留，因此实现会在脉冲后显式清零，避免持续施加。
推搡使用绝对 XY 速度替换，不累加速度；回合复位不重启全局计时。

## LgDog 的适配

- 机身名义质量 3.487 kg，因此随机质量为 **2.487～5.487 kg**。此配置替换此前的 0.9～1.1
  机身密度倍率，两套质量扰动不叠加。腿部保持名义质量和惯量。
- 在质心坐标下将完整惯量张量与质量同比缩放，保持惯量正值及物理可行性，重新计算动力学常数。
  质心偏移仍以 LgDog 模型原有质心为基准，避免照搬原版绝对 COM 坐标造成坐标原点错误。
- 保留 LgDog 髋/大腿/小腿名义 Kp **20/20/40**、Kd **1/1/2**。
  实际 Kp = 名义 Kp × Kp 倍率 × 电机强度，Kd 同理；最终力矩上限仍为 **10/10/20 Nm**。
  MuJoCo 使用原生位置执行器，隐式积分，与 Isaac Gym 的显式 PD 不构成逐步数值等价。
- 官方代码虽然抽样 `motor_strength_factors`，但 `_compute_torques` 没有使用该变量。
  本项目让其实际作用于饱和前的 PD 力矩，并保持原有硬力矩上限，这是明确的实现补全。
- 原版 `randomize_link_mass`、`randomize_restitution` 默认关闭，本次也不启用；相机外参不增加随机化。
  既有编码器偏置 ±0.015 rad、本体观测噪声保留。零名义角度乘倍率仍为零，小腿 0.2 rad 对应 0.1～0.3 rad。
- 比赛训练和感知采集均移除基类旧的随机间隔推搡，避免与新配置叠加。
  保留定向路线、任务指令及奖励；`foot_clearance=0`，actor/critic 仍为 252/261 维。

## 配置与使用

统一入口：`src/tasks/velocity/config/lgdog1/domain_rand_cfg.py` 的 `HIMLOCO_DR`。
执行事件位于 `src/tasks/velocity/mdp/himloco_randomization.py`，子步延迟及外力位于 `himloco_actions.py`。
普通 Rough-MGDP/HIM 任务和 `play` 不启用这套新增配置，保留各自原有设置。

新创建训练环境时自动生效，已运行的进程不会热更新。建议用新 run 训练与评估；
课程检查点的配置指纹包含这些参数，不能沿用旧物理配置的通过率/速度晋级统计。
旧策略可用于初始化网络，观测维度兼容；完整课程恢复仍须配置一致。
比赛感知权重的顶层 `domain_randomization` 保存完整参数、上游提交、物理步长和脉冲持续时间。

在项目根目录、已激活训练环境后，例如：

```powershell
python -m scripts.pretrain_mgdp --task LgDog1-Race-MGDP --num-envs 64 --terrain-rows 4 --terrain-cols 10 --batch-size 64 --steps 4000 --iterations 3000
```

随后照常在工作台选择 `race_mgdp` 启动策略训练。全强度扰动会提高训练难度，
短期通过率可能下降；是否改善部署鲁棒性需要 GPU 长训和独立评估。

## 验证记录

本地检查记录位于忽略目录 `.artifacts/himloco-dr/`，不上传临时模型或日志。
回归测试保留在 `tests/test_mgdp_domain_rand.py`、`tests/test_himloco_randomization.py` 和
`tests/test_mgdp_passage.py`，覆盖质量/完整惯量张量联动、任务隔离、复位采样、执行器倍率、
硬力矩上限不变、逐物理步延迟、局部外力旋转和清零、绝对 XY 推搡及课程指纹变化。

CPU 检查验证实现和训练链路，不代表 GPU 收敛、越障成功率或实机效果。

| 检查 | 结果 |
|---|---|
| 全套回归 | 270 项，0 失败，1 跳过；`all-tests.log` |
| 实际 CPU 仿真 | 10 环境、2 行×10 列；质量/完整惯量同比、质心偏移保留、腿部不变 |
| 实际抽样 | seed42 机身质量约 3.28674～5.29079 kg，在 2.487～5.487 kg 内 |
| 复位 | 子集环境重采样 PD/强度/摩擦/关节位置，其余环境及初始化质量/惯量/质心保持 |
| 子步执行 | 直接读取 MuJoCo `ctrl` 和 `xfrc_applied`，确认 0/1/2/3 步延迟、局部外力旋转、单步后清零 |
| 力矩与推搡 | 饱和目标下仍限制 10/10/20 Nm；全局第 800 步写入绝对 XY 随机速度，观测/奖励有限；`physics.log` |
| 感知采集与训练 | 16 步采集、30 条连续 4 帧序列、反向更新 2 次并保存；`pretrain.log` |
| PPO 链路 | 实际 CPU 更新、检查点/课程保存、ONNX 导出、恢复后再更新；`ppo.log` |

检查命令：`python -B -m unittest discover -s tests`。实际 CPU 仿真使用 mjlab 1.2.0、
MuJoCo 3.5.0 和 CPU 版 PyTorch，未运行 GPU 长训或实机部署。
