# LgDog HIM Standard 训练归档分析（2026-10-22）

分析目录：`E:\mjlab\数据\lgdog_training_data\logs\rsl_rl\lgdog1_velocity_blind_him_standard_v1`。归档包含 V1、V2、两次旧 V8，以及 2026-10-21 的稳定性默认值 run。

## 本轮 run

`2026-10-21_22-44-03_V8_him_standard_seed42`

- 实际 seed：42；1024 环境；11223 个训练迭代后中断。
- `num_steps_per_env=48`、`clip_actions=1.0`、HIM `estimator_epochs=5`，这些值都与 `params/agent.yaml` 一致。
- 最后 `mean_reward=57.32`，最后 `mean_episode_length=875.67`。
- 最后 `velocity_error_xy_mps=0.1999`，后四分之一均值约 0.1791 m/s。
- 最后 `tracking_ratio=0.5647`，后四分之一均值约 0.5715。
- 最后 `reset_batch_failure_fraction=0.2427`，后四分之一均值约 0.2251。
- `action_rate_l2` 后四分之一均值约 -0.367，已经没有旧 V2/V8 中的极端动作变化率尖峰。
- HIM 速度辅助损失后四分之一均值约 0.00210，`him_updates` 稳定为 20，说明估计器更新次数已生效。

## 主要结论

这轮的跟踪、存活和动作平滑性都有改善，但总回报不能直接解释为地形能力提升。`Curriculum/terrain_levels/level_mean` 从约 0.5 降到 0，`promote_fraction` 全程为 0；同时日志中的课程速度误差达到 8.37（峰值 17.42），而回合指标只有约 0.20 m/s。这两个量相差约 50 倍，正好等于控制周期 `1 / 0.02`。

原因是 MetricsManager 累加的是每个控制步的 `velocity_error_xy_mps`，课程函数旧实现用“累计值 / 秒数”计算平均值。正确公式应为“累计值 / 控制步数”。该问题已在 `blind_mdp.py` 修复，并由测试 `test_velocity_gate_uses_control_step_mean` 锁定。修复前的 `level_mean=0` 只能说明升级门槛被错误计算，不能说明策略无法通过地形。

本轮还有一个训练稳定性信号需要处理：`Policy/mean_std` 从约 1.0 增长到 69.36，`Loss/entropy` 达到 67.90。虽然动作被裁剪后 `action_rate_l2` 变小，但策略分布已经过宽，继续训练可能依赖裁剪“掩盖”探索发散。下一轮应把它作为硬监控项，同时观察 `approx_kl`、学习率和 NaN，而不能只看 mean reward。

## 与历史 run 的比较

| run | 最后 reward | 最后速度误差 | 最后 tracking ratio | 最后失败率 | 最后 terrain level |
|---|---:|---:|---:|---:|---:|
| V1 | 26.74 | 0.226 | 0.459 | 0.500 | 0.012 |
| V2 | 49.15 | 0.201 | 0.553 | 0.215 | 0.533 |
| V8（10-18） | 25.69 | 0.226 | 0.460 | 0.396 | 0.737 |
| V8（10-19） | 28.33 | 0.210 | 0.526 | 0.299 | 1.013 |
| 本轮（10-21） | **57.32** | **0.200** | **0.565** | 0.243 | 0（门槛错误） |

V2 仍是旧数据中较稳定的地形基线；本轮的 tracking 和动作平滑性优于 V2，但地形等级因单位错误失效，不能据此排名。

## 下一轮建议

1. 从头启动 `V9_curriculum_fix_seed42`，使用修复后的课程函数；不要把本轮 level 0 当作策略状态恢复依据。
2. 先跑 3000–5000 迭代，确认 `Curriculum/terrain_levels/velocity_error_xy_mps` 回到约 0.15–0.30 的 m/s 量级，并观察 `promote_fraction` 是否出现非零值。
3. 保留 seed 42 作为修复前后的连续对照，再用 seed 123 做独立复现；实验名必须与 `agent.yaml` 的真实 seed 一致。
4. 将 `Policy/mean_std < 3` 作为预警线；若继续上升，单独做 entropy coefficient 0.01 与 0.002 的短 A/B，并同时记录 `approx_kl`、学习率和动作饱和率。
5. 通过率评估应在固定平地、台阶、坡面和随机起伏场景分别进行，不能用全局 reward 或 curriculum level 单项替代。
