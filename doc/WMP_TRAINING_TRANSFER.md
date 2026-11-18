# WMP（wmp_go1）可迁移到 LgDog1 MGDP 技能训练的要点

来源：`E:\足式机器人\WMP - World Model-based Perception for Visual Legged Locomotion`
（论文 ICRA 2025 + `wmp_go1-main` 训练仓库 + `LeggedSkillDeploy` 部署仓库）。

## 一、WMP 的关键数字

| 项 | 值 |
|---|---|
| 世界模型（RSSM）频率 | **10 Hz**，即每 5 个控制周期更新一次（`update_interval = 5`） |
| 深度图 | 64×64，FOV 58°，下俯 20°，far 2 m，深度归一化 [-0.5,0.5] |
| 装相机的环境数 | `camera_num_envs = 1100`（其余环境用深度预测器） |
| World Model 输入 | 本体 33 + 深度 64×64 + 最近 5 步动作 60 + 上一隐状态 + is_first |
| World Model 输出 | RSSM 512 → `wm_feature_encoder` 64→64→**32** |
| History Encoder | 5 帧 × 42 = 210 → 256 → 128 → **35**（32 历史特征 + 3 预测线速度） |
| Actor 输入 | 35 + 3（指令）+ 32（世界模型）= **70** → 256→128→64→12 |
| Critic | 非对称，额外拿特权 scandots，[512,256,128] |
| PPO | entropy 0.01、5 epochs、4 mini-batches、`vel_predict_coef=1.0` |
| 课程 | 5 个速度组独立升级；`linear_score>0.8` → 上限 +0.2 m/s，只升不降 |
| 训练规模 | 建议 >2 万轮；`max_iterations=100000`，每 1000 轮存盘 |

## 二、六个可迁移机制

### 1）分组速度课程：解决我们"高墙 0% 锁死全局速度"

WMP 把地形分成 5 组（omni / rough_flat / stairs_discrete / forward / slope_cliff），
**每组独立累计 tracking 奖励**，`linear_score > 0.8` 时该组 X 上限 +0.2 m/s，只升不降。

我们现在的 `PassageCurriculum`（`src/tasks/velocity/mdp/passage.py:310`）是**全局单一速度范围**，
要求"所有列 ≥32 回合且通过率 ≥70%"，高墙 0% 就永远不扩速——这正是 V1_seed42 全程 0.6 m/s 的原因。

**改法**：给 10 个 lesson 分 3 组，每组维护自己的 `lin_vel_x` 上限：

| 组 | 包含 | 初始 | 上限 |
|---|---|---|---|
| 平缓组 | start_flat、random_hills、mjlab_rough | 0.6 | 2.0 |
| 越障组 | gap_steps、high_wall、clearance_frame、tire_array | 0.6 | 1.0 |
| 精细组 | stepping_stones、balance_beam、slalom_brick | 0.6 | 1.0 |

高墙卡住时只冻结"越障组"，平缓组仍能扩到 1.5–2.0，训练不会整体停滞。

### 2）感知降频：我们目前每步都算深度

WMP 的世界模型 **10 Hz**（每 5 步一次），深度只有 1100 个环境渲染。
我们的深度/感知编码器现在跟随 50 Hz 控制步，每个环境每步都算一遍。

**改法**：`mgdp_perception` 每 5 个控制步更新一次 latent 并缓存，中间 4 步复用。
预计感知计算量降到 1/5；如果深度分辨率还用 64×64（我们现在是 187 条射线），还能再省。

### 3）命令采样按地形分组

WMP：全方向组允许 X/Y/heading；**只向前组固定 heading=0、vy=0**，XY 合速度 <0.2 m/s 清零，5% 概率零速。
我们目前 10 个 lesson 共用一套 `PassageCommand` 参数（`lin_vel_y=(-.1,.1)`、`ang_vel_z=(-.8,.8)`）。

**改法**：越障组固定 heading=0、vy=0；绕桩组才开放较大 yaw；零速采样 5%。

### 4）Actor 拿"预测线速度"

WMP 的 History Encoder 输出 32 维历史特征 + **3 维预测机体线速度**，一起给 Actor。
我们 HIM 历史的 actor 输入里没有显式线速度预测（只有 critic 有真值）。

**改法**：给历史编码器加 3 维线速度回归头（`vel_predict_coef=1.0`），让 actor 拿到速度估计，
对 0.6 m/s 低速段的跟踪误差（我们目前 0.158 m/s）应有帮助。

### 5）Critic 特权、Actor 只拿学生输入

我们已经是非对称 actor-critic（critic 有 height_scan/base_lin_vel），与 WMP 一致；
区别是 WMP 用 `sg()` 停梯度，避免世界模型被策略目标污染。我们的 perception encoder 是冻结的，
效果等价，可保持。

### 6）AMP 参考动作（可选）

WMP 用 AMP（`amp_reward_coef=0.5*0.02`、`task_reward_lerp=0.3`、判别器 [1024,512]）约束步态自然度。
我们的 `feet_air_time`/`foot_gait` 是手工塑形；如果步态仍然抽，可考虑引入 AMP，但工程量最大，放到最后。

## 三、建议的实施顺序

| 优先级 | 改动 | 预计收益 | 工作量 |
|---|---|---|---|
| P0 | 分组速度课程（第 1 条） | 直接解决"高墙 0% 冻住全局速度" | 中（改 `PassageCurriculum` + 每环境速度范围） |
| P1 | 感知降频到 10 Hz（第 2 条） | 采集耗时约 -20~30%（我们 7.98 s/轮主要是物理+感知） | 小（加 step 计数器缓存 latent） |
| P2 | 只向前组命令采样（第 3 条） | 越障地形收敛更快、少学横向无用动作 | 小 |
| P3 | 历史编码器加线速度预测（第 4 条） | 低速跟踪误差下降 | 中 |
| P4 | AMP 参考动作（第 6 条） | 步态自然度 | 大 |

## 四、与我们现状的对照结论

- WMP 的核心不是"世界模型本身"，而是**用历史潜变量替代特权教师 + 分组课程**；
  我们已经有 HIM 历史编码器和非对称 critic，差的是**分组课程**和**感知降频**。
- 我们 V1_seed42 的两大症结（高墙 0% 锁死全局、感知每步全算）正好对应 P0 和 P1。
- 先做 P0+P1，再重训一版，预期能看到：速度上限从 0.6 抬到 1.0+、每轮采集时间下降。

## 五、WMP 的 reward 与引导设计（`legged_gym/envs/base/legged_robot.py` + `a1_amp_config.py`）

### 奖励项与权重（`a1_amp`）

| 类别 | 项 | 权重 | 说明 |
|---|---|---:|---|
| 任务 | `tracking_lin_vel` | +1.0 | `exp(-err²/0.25)`；仅 gap/bream/continuous_gap/climb 地形做速度裁剪（`lin_vel_clip=0.1`，允许略低不罚） |
| 任务 | `tracking_ang_vel` | +0.5 | 角速度跟踪 |
| 安全 | `torques` | -0.0001 | 力矩平方 |
| 安全 | `dof_acc` | -2.5e-7 | 关节加速度 |
| 安全 | `action_rate` | -0.03 | 一阶动作差分 |
| 安全 | `smoothness` | -0.01 | **二阶动作差分**（我们只有 action_rate） |
| 安全 | `collision` | -1.0 | 大腿/小腿接触力 >0.1 N 就罚 |
| 安全 | `dof_pos_limits` / `dof_vel_limits` | -10.0 | 超出关节限制 |
| 姿态 | `lin_vel_z` | -1.0 | 机体竖直速度 |
| 足端 | `feet_air_time` | +1.0 | 只在**首次触地**给奖，目标 0.5 s，指令 <0.1 m/s 不给 |
| 足端 | `feet_stumble` | -1.0 | 水平接触力 > 4×垂直力（撞竖面） |
| 足端 | `feet_edge` | -1.0（课程） | 只在 gap/climb 地形启用，权重 0.1→1.0（第 4000–10000 轮线性） |
| 其他 | `cheat` | -1.0 | 只对 gap~crawl 地形：机头偏航 >1.0 rad 就罚（防止绕开障碍） |
| 其他 | `stuck` | -1.0 | `\|vx\|<0.1` 且 `\|cmd_x\|>0.1` |
| 其他 | `stool_base_height` | -1.0 | 只在 stool 地形，机高 >0.45 m 罚 |
| AMP | `amp_reward_coef` | 0.01（0.5×0.02） | 风格奖励与任务奖励按 `task_reward_lerp=0.3` 混合，判别器 [1024,512] |

### 引导（curriculum）设计

1. **按地形块施加奖励**：环境按地形块连续排列（`gap_start_idx..gap_end_idx` 等），
   `cheat`、`stool_base_height`、`feet_edge`、速度裁剪都只作用于对应索引区间——
   同一套 reward 对 16 种地形各管各的，不需要为每地形写一份配置。
2. **奖励课程**：`reward_curriculum=True`，`feet_edge` 从 0.1 线性升到 1.0（4000→10000 轮），
   先学走、再学"踩边缘"。
3. **5 组速度课程**：见 `doc/TERRAIN_COMMANDS.md`，每组独立按 `linear_score>0.8` 升级 +0.2 m/s，只升不降。
4. **命令采样**：5% 零速；全方向组可选极限采样；**只向前组固定 heading=0、vy=0**，XY 合速度 <0.2 m/s 清零；
   偏航 >1.0 rad 触发 `cheat` 惩罚。

### 与我们现有奖励的对照

- 我们有 `tracking_lin_vel/ang_vel`、`action_rate`、`feet_stumble`、`feet_air_time`、`dof_pos_limits`，
  但没有**二阶动作平滑**、没有**按地形块施加的 `cheat` 惩罚**、没有**奖励权重随时间爬升的课程**。
- 我们的 `wall_waiting/wall_crossing`、`stone_touchdown`、`slalom_heading_progress` 是更细的显式引导，
  比 WMP 的"块惩罚 + AMP"更依赖人工设计；WMP 用 AMP 替代了大部分步态塑形。
- 可直接借用的三项：**二阶平滑 `smoothness`**、**`cheat` 偏航惩罚**（防绕行，正好对应我们绕桩/越障的绕路问题）、
  **奖励权重课程**（例如把 `feet_air_time` 或 `passage_progress` 前 3000 轮降到 0.2 再升满）。
