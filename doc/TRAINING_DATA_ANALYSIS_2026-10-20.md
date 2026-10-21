# 训练数据分析与优化空间

分析对象是 `lgdog1_velocity_blind_him_standard_v1` 下的五个 run。`AGENT_SMOKE` 只有 3 个点，不能用来评价策略；主要比较 V1、V2 和两次 V8。

## 现有数据

| run | 轮数 | 最后 episode length | 最后地形等级 | 最后 XY 速度误差 | 持续腿接触 |
|---|---:|---:|---:|---:|---:|
| V1 `2026-10-17_06-26-53` | 866 | 656 | 0.01 | 0.92 | 0.71 |
| V2 `2026-10-18_03-24-40` | 16000 | 872 | 0.53 | 1.00 | 0.38 |
| V8 `2026-10-18_18-30-44` | 3169 | 629 | 0.74 | 0.95 | 0.75 |
| V8 `2026-10-19_10-14-38` | 16000 | 713 | 1.01 | 1.00 | 0.67 |

V2 的存活和腿部接触最好，但速度误差仍接近 1 m/s。V8 的地形等级继续上升，说明课程机制能推进；不过存活没有跟着提高，课程有提前进入策略尚未稳定的区域的风险。

## 已发现的问题

### 1. `feet_stumble` 被关闭后，腿部接触退化

10 月 19 日的 V8 快照中 `feet_stumble.weight=0.0`，持续腿接触比保留该项的 V8 更高，episode length 也没有改善。当前源码已恢复默认 `-0.5`，下一次训练应保持这个值。

### 2. 动作变化率出现尖峰

10 月 19 日 V8 的 `Episode_Reward/action_rate_l2` 后期平均约 `-14.6`，远高于其它 run 的约 `-0.6` 到 `-0.9`。这类异常会让总 reward 失真，通常表示策略输出或相邻动作发生尖峰。当前训练快照的 `agent.clip_actions` 是 `null`。

源码已将新训练默认 `clip_actions` 设为 `1.0`；仍建议每轮记录 `max(abs(action))`、`mean(abs(action))` 和 `mean(abs(action - previous_action))`，并与旧的 `null` 实验对照。

### 3. 实验名称和真实 seed 不一致

两个目录都叫 `V8_him_standard_seed42`，但配置快照中的真实 seed 分别是 `123` 和 `120`。这会让结果追踪和复现实验出错。新的 run name 应包含配置快照中的真实 seed，例如 `V4_clip1_seed120`，不要手写一个与参数不同的 seed 名称。

### 4. 课程推进快于策略稳定

V8 的 terrain level 达到约 1.0，但 `sustained_leg_contact` 仍约 0.67，速度误差约 1.0 m/s。源码已在 blind 课程升级时加入平均 XY 速度误差 `<=0.8 m/s` 门槛，并继续使用回合完成、跟踪进度和离开出生平台条件。当前仍建议先固定 level 0–1 训练 3000–5000 轮，再观察是否需要开放更高等级。

### 5. 指令数据需要分桶统计

当前只看全局平均速度误差，无法判断问题来自前进、横移还是转向。建议把数据按以下桶分别统计：

- `|vx| < 0.1`、前进 `vx > 0.1`、后退 `vx < -0.1`
- `|vy| > 0.1`
- `|wz| > 0.1`
- 平地、台阶、坡面、随机起伏

如果只有横移或转向误差大，应分别调整指令采样比例和对应 tracking reward；如果所有桶都大，优先处理动作尖峰和课程推进。

## 下一组实验建议

先做三组短实验，每组 3000–5000 轮、相同 seed 和相同环境数：

1. **V9 stability**：当前 reward，`feet_stumble=-0.5`、`clip_actions=1.0`、rollout 48、HIM estimator epochs 5。
2. **V9 curriculum**：V9 stability + 先固定低等级地形，稳定后再开放课程。
3. **V9 reward A/B**：只改 `foot_gait` 或 `foot_clearance` 的一个参数，验证 reward 冲突是否仍存在。

每组只比较以下指标：`velocity_error_xy`、`mean_episode_length`、`sustained_leg_contact`、`terrain_levels/level_mean`、`max(abs(action))` 和 `action_rate_l2`。通过标准建议为：速度误差低于 V2、持续腿接触低于 V2、episode length 不低于 V2，并且动作统计没有尖峰。
