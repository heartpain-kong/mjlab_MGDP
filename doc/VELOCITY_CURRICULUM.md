# LgDog速度课程：按实际指令进度扩速

2026-10-27。适用于标准 Rough-HIM 与 Flat-HIM；未启用此课程的旧PPO任务仍按其原配置运行。HIM网络、PPO更新、部署观测和动作接口保持原有合同。

## 为什么增加进度门

V23截至1317轮的速度课程分数EMA约0.825，但有效指令完成度仅约0.347，地形进度门全程为0。绝对误差公式在指令0.2 m/s、实际0时仍可得到0.852分，超过原扩速阈值0.78。存活达标只能说明没有提前失败，不能证明已经按要求走动。

新版本保留绝对分数与存活/失败约束，再增加独立的实际进度条件。它阻止过早扩速，不会直接给策略增加行走奖励，也不保证地形会在固定轮数内升级。V23证据见 [分析报告](TRAINING_DATA_ANALYSIS_V23_2026-10-27.md)。

## 当前判据

每约20秒仿真控制时间检查一次，聚合这段窗口内**已经完成的回合**。开始时平面指令范围为最终范围的一半。

| 条件 | 允许扩速 | 触发回退 |
|---|---|---|
| 绝对跟踪分数EMA | ≥0.78 | <0.62 |
| 窗口有效指令进度 | **≥0.60** | **<0.40** |
| 平均存活时长 / 最大回合时长 | ≥0.80 | <0.65 |
| 非超时失败回合数 / 完成回合数 | ≤0.25 | >0.35 |

扩速还要求有移动样本，且窗口累计移动指令路程至少1米。全部扩速条件连续三次通过，扩大一次0.025；任一回退条件连续两次成立，缩小一次0.025。中间区间保持当前范围；回退不低于初始范围，扩速不超过设置的最终范围。移动距离不足时不允许扩速，也不单凭进度回退。偏航角速度仍保持配置范围。

60%/40%是针对V23问题提出的初始门限，需要固定指令回放和后续训练验证。地形升级仍使用75%完成度、2米进度、平台外暴露、存活及非失败等原条件；两条课程的门限含义不同。

## 有效进度如何计算

逐个控制步，在机身坐标系下，令平面指令为c、实际平面速度为v：

```text
指令距离增量 = ||c|| * step_dt
有效进度增量 = clamp(dot(v, c / ||c||), 0, ||c||) * step_dt
窗口进度比例 = 所有完成回合的有效进度之和 / 所有完成回合的指令距离之和
```

只计平面指令速度大于0.1 m/s的步；站立样本向分子、分母都贡献0。逆向/侧向运动不赚进度，超速不额外赚进度。窗口按**米数**聚合，避免短指令回合的高比例盖过长回合欠速；不是对每个回合比例直接取平均，也不是最终坐标离出生点的距离。

## 修改位置

在 `src/tasks/velocity/config/lgdog1/blind_terrain_env_cfg.py` 中标准HIM的 `cfg.curriculum["command_vel"].params` 修改：

```python
"progress_metric_name": "tracking_progress",
"distance_metric_name": "commanded_distance",
"min_progress_ratio": 0.60,
"regress_min_progress_ratio": 0.40,
"min_commanded_distance": 1.0,  # 整个检查窗口累计指令米数，不是单回合门限
```

Flat-HIM沿用这条速度课程。类实现位于 `src/tasks/velocity/mdp/curriculums.py::RewardBasedVelocityCurriculum`。未配置这对进度指标的旧调用者保持原有行为；配置了门限却缺少指标时直接报错，避免静默退回仅按奖励扩速。

## 工作台与TensorBoard

工作台“监控数据 → 速度追踪 / 地形课程”新增：

- 速度课程有效进度及当前门限、扩速进度门、扩速分数门。
- 地形完成度门≥75%、距离门≥2米、平台外暴露门≥1秒的独立通过率。

主要新tag为：

```text
Curriculum/command_vel/progress_ratio
Curriculum/command_vel/min_progress_ratio
Curriculum/command_vel/progress_gate_pass
Curriculum/command_vel/score_gate_pass
Curriculum/terrain_levels/ratio_gate_pass_fraction
Curriculum/terrain_levels/distance_gate_pass_fraction
Curriculum/terrain_levels/exposure_gate_pass_fraction
```

`promotion_allowed`表示进度、存活、失败率与样本量等前置条件通过；还须 `score_gate_pass=1` 并满足连续次数才真正扩速。地形的三个新增指标只负责诊断，没有修改原升级规则。旧run没有对应指标时显示“—”。

## 续训与验证

课程checkpoint状态升级为v3，保存未完成检查窗口的指令米数、进度米数和上一窗口比例。读取旧v1/v2时保留原速度范围，清空缺少进度统计的部分窗口及连续次数，再重新累计。

**保留旧范围意味着续训V23不会立即回到半速。** 新判据可能触发逐步回退；要比较修改前后效果，应新建实验、固定实际奖励和地形配置。已经运行的进程不会热更新，更新文件后需重新启动才能生效。面板的奖励覆盖也需单独核对，本次没有修改权重。

CPU回归覆盖高分但原地不动、V23式低进度、按米数加权、连续扩速/回退、全站立与少量样本、旧checkpoint迁移、窗口断点恢复和地形分门诊断。本次CPU unittest共133项，132项通过、1项跳过；工作台新旧日志、无数据状态和诊断提示优先级的渲染逻辑检查通过。GPU收敛、实际步态和地形通过率需要另行验证。
