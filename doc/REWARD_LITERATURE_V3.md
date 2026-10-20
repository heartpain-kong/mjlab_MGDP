# HIM 标准崎岖地形奖励 V3

这版调参针对 `2026-10-18_03-24-40_V2` 的实际日志，不覆盖旧 run，也不建议直接续训旧 checkpoint。请用新的 run name 重新训练并与 V2 做相同地图、相同指令的对比。

## 日志依据

V2 在 16000 轮结束时：

- 平均 episode length 约 872 / 1000，说明站立和存活已经明显改善。
- `Metrics/twist/error_vel_xy` 仍约 1.00 m/s，速度跟踪是主要问题。
- `Curriculum/terrain_levels/level_mean` 约 0.53，课程没有继续推进。
- `Episode_Reward/action_rate_l2` 约 -0.69，当前权重为 -0.10。
- `Episode_Reward/foot_clearance` 约 -0.20，当前目标高度为 0.20 m、权重为 -1.20。
- `Episode_Termination/sustained_leg_contact` 末段约 0.38，仍有腿部持续接触导致的失败。

## V3 修改

| 项目 | V2 | V3 | 目的 |
|---|---:|---:|---|
| `action_rate_l2` | -0.10 | -0.05 | 减少动作平滑项对速度跟踪的压制 |
| `foot_clearance` | -1.20 | -0.80 | 保留抬脚引导，避免它成为主导惩罚 |
| `foot_clearance.target_height` | 0.20 m | 0.10 m | 恢复与 LgDog 足端尺寸和原始任务一致的目标 |
| `soft_landing` | -0.002 | -0.001 | 保留落地保护，避免重复惩罚过重 |
| `stand_still` | 0 | -1.0 | 低速或零指令时约束漂移和乱动 |
| `foot_gait` | 0.50 | 0.75 | 提高接触节律信号，帮助减少持续腿部接触 |

`feet_air_time` 继续保持 0。当前日志没有证明需要同时引入第二个步态目标；先避免让固定 0.4 秒目标和 HIM 的历史估计互相竞争。`leg_contact_force`、`feet_stumble` 和 `foot_slip` 暂不改变，先观察 V3 的速度误差和持续腿接触是否下降。

## 文献依据

- HIMLoco 的核心是用历史响应估计不可观测状态，以少量本体传感器实现跨地形控制；奖励应保持任务导向，避免大量互相竞争的正则项。[HIMLoco 项目与论文](https://junfeng-long.github.io/HIMLoco/)
- 非刚性地形研究发现，足端历史和足端速度对地形适应很关键；本项目保留历史编码，并把抬脚代价按足端速度加权。[Quadruped Locomotion on Non-Rigid Terrain](https://arxiv.org/abs/2107.02955)
- Terrain-Aware Quadrupedal Locomotion 将安全落足和抬脚高度作为独立引导项；V3 保留 `foot_clearance`，但把目标和权重降回可学习的范围。[Terrain-Aware Quadrupedal Locomotion](https://arxiv.org/abs/2310.04675)
- Barrier based style rewards 的实验表明，足端高度、姿态和步态约束应作为软约束使用；V3 用适度 `foot_gait` 加上较轻的抬脚项，避免硬性追求过高足端轨迹。[Barrier-Based Style Rewards](https://arxiv.org/abs/2409.15780)

## 训练和验收

使用新的实验名从头训练：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=1024 --agent.max-iterations=16000 \
  --agent.logger=tensorboard --agent.seed=42 \
  --agent.run-name=V3_reward_literature_seed42 --enable-nan-guard True
```

先比较前 3000 轮和最后 1000 轮的以下指标：`Metrics/twist/error_vel_xy`、`Train/mean_episode_length`、`Curriculum/terrain_levels/level_mean`、`Episode_Termination/sustained_leg_contact`、`Episode_Reward/foot_clearance`、`Episode_Reward/action_rate_l2`。V3 的目标是速度误差下降、课程等级继续上升，同时存活时间不低于 V2；如果速度误差下降但腿部接触升高，再单独把 `leg_contact_force` 从 -1.0 调到 -1.25，不要一次修改多项。
