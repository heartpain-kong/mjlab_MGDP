# LgDog1 训练轮次登记表（V 编号）

> 2026-10-16 起，所有训练轮次按 **V1 / V2 / V3 …** 编号，一个 V 号 = 一轮完整训练。
> 与 `LgRobotDog` 那边的 V 编号习惯保持一致。

> **下一轮候选（尚未启动）**：`LgDog1-Rough-Blind-HIM-Standard`，使用标准 Rough 地形，actor/critic 无机身高度扫描，HIM 编码器仍接收六帧本体历史。命令与新日志目录见 [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md)。可用 `V8_him_standard_seed42` 命名；本文 V7/V7b 的成绩均属于旧温和地形任务。

## 命名规则

| 项 | 规则 | 例子 |
|---|---|---|
| `--agent.run-name` | `V<n>_<短描述>` | `V7_him_seed42` |
| run 目录 | `<时间戳>_V<n>_<短描述>` | `2026-10-16_19-39-11_V7_him_seed42` |
| 文字日志 | `logs/train_V<n>_<短描述>_<月日_时分>.log` | `logs/train_V7_him_seed42_1016_1939.log` |
| experiment 目录 | 由任务的 `rl_cfg.experiment_name` 决定（按任务分），**V 号只体现在 run 名里** | `lgdog1_velocity_blind_him_v1/` |

每个 V 轮的产物（都在 run 目录下）：`model_<轮数>.pt` + `policy.onnx`（每存盘同步导出）+ `params/{env,agent}.yaml` 配置快照 + `git/` 仓库 diff。

## ★ 开新一轮的标准命令

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog && conda activate unitree_rl_mjlab

# 先看显存，再决定 --env.scene.num-envs（见下面"显存上限"表）
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

setsid nohup python scripts/train.py <任务> \
  --env.scene.num-envs=<N> \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --agent.run-name=V8_<短描述> \
  --enable-nan-guard True \
  > logs/train_V8_<短描述>_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
```

> 用 `setsid nohup ... < /dev/null &` —— 普通 `nohup ... &` 在调用 shell 退出后会被挂断（踩过）。

## 看曲线（TensorBoard）

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog
bash tb.sh start      # 已在跑则跳过
bash tb.sh status     # 看端口 + 已识别的 run 列表
bash tb.sh stop       # 停
bash tb.sh restart    # 重启
```

浏览器打开 **http://localhost:6006**（局域网内其他机器用 `http://192.168.28.158:6006`）。
TensorBoard 的日志在 `logs/tensorboard/tensorboard.log`，pid 记在 `logs/tensorboard/tb.pid`。

左侧会自动按任务分组，7 个 V 轮并排对比：

```
lgdog1_velocity/                    V1 flat, V2 rough_sighted
lgdog1_velocity_blind/              V3 blind, V4 stand25, V5 curfix
lgdog1_velocity_blind_gentle/       V6 gentle          ← 目前最好
lgdog1_velocity_blind_him_v1/       V7 him (在跑)
```

**重点看这几条：**

| 曲线 | 含义 | 期望 |
|---|---|---|
| `Train/mean_reward` | 平均回报 | 单调上升，V6 已到 ~909 |
| `Curriculum/terrain_levels` | 课程地形等级 | 必须**上升**；塌到 0 = 课程判据失效 |
| `Metrics/track_lin_vel` | 线速度跟踪 | 越高越好，V6 ~0.63 |
| `Episode/mean_length` | 回合长度 | 越接近 1000（=20s 上限）越好 |
| `Loss/value` `Loss/surrogate` | PPO 健康度 | 平稳；突然爆 → nan guard 会拦 |

> ⚠️ **V7(HIM) 的地形曲线名字不一样**。HIM 用的是 `blind_mdp.terrain_levels_tracking`
> （不是我们修的 `velocity_env_cfg.terrain_levels_vel`），记录的是**一组** tag：
> `Curriculum/terrain_levels/level_mean`、`/promote_fraction`、`/demote_fraction`、
> `/tracking_ratio`、`/episode_commanded_m`、`/episode_progress_m`。
> 看 V7 就盯 `level_mean` 和 `promote_fraction`。

### ⚠️ 别把 iter 5000 的台阶当成训练崩了

`velocity_env_cfg.py:441` 有个**指令范围课程**（`commands_vel`），在
`common_step_counter > 5000×24` 时把速度指令从 `(-0.5, 1.0)` 放宽到 `(-1.0, 2.0)`：

```
iter 4950 → 5000 → 5050 → 5100
error_vel_xy   1.58   1.52   3.31   3.95     ← 台阶
track_lin_vel  1.01   1.05   0.50   0.54     ← 台阶
```

**所有轮次都会在 iteration 5000 处出现这个断崖**，那是任务变难了，不是策略退化。
V6 的"track 0.628 最低"就是被这个台阶压下来的（5000 之前它一直在 0.96~1.05）。
跨轮次比较务必取**同区间**（要么都 `<5000`，要么都 `>5000`）。

### V7(HIM) 的课程判据比 V6 严得多

`blind_mdp.terrain_decisions()` 要求**同时**满足才升级：

```
ratio >= 0.75  且  progress >= 2.0m  且  存活 >= 18s(0.9×20s)
且  没摔倒  且  离开了出生平台 >= 1.0s
```

V6 用的 `terrain_levels_vel` 只要求 `走过的路程 > 场地半宽`，宽得多。

**实测结论：门槛不用放宽，早期 `promote_fraction = 0` 是正常的。** V7 实测爬升：

| iter | 200 | 400 | 600 | 800 | 1000 | 1500 | 2000 | 2659 |
|---|---|---|---|---|---|---|---|---|
| `level_mean` | 0.000 | 0.000 | 0.002 | 0.096 | 0.565 | 2.171 | 2.926 | **3.343** |
| `tracking_ratio` | 0.173 | 0.411 | 0.567 | 0.613 | 0.678 | 0.720 | 0.681 | 0.646 |
| `episode_progress_m` | 1.03 | 2.24 | 3.35 | 3.78 | 3.97 | 4.31 | 4.02 | 3.58 |
| `promote_fraction` | 0 | 0 | 0 | 0 | 0.125 | 0.292 | 0.042 | 0.167 |

前 ~600 轮跟踪比只有 0.17~0.57，够不到 0.75，所以升级数必然是 0 —— **这是策略还没学会跟速度，不是课程坏了**。
等跟踪比上来它自己就开始升级了。判据合理性：`ratio≥0.75` 大致对应 `track_lin_vel ≈ 0.75`，
和 V6 在易指令区间（iter<5000）的 0.96~1.05 是同一水平线。

> 教训：**别只看前几百轮就下结论**。V7 在 iter 390 时 `level_mean=0` 曾被我误判为"课程失效"，
> 实际到 iter 600 就开始升级了。

## 轮次表

> ⚠️ **必须用绝对路径 `--logdir`**。踩过的坑：TensorBoard 用相对路径启动、之后 cwd 被删，进程还在跑、端口也通，但 `/data/runs` 返回 `[]`，页面全白。
> 用 `tb.sh` 就不会有这个问题。自救：`curl -s localhost:6006/data/runs`，空的就是这个毛病。

> V6 的 tfevents 在 TensorBoard 启动日志里会有一条 `BadLengthCrc ... ChecksumError` 警告 ——
> 那是上次机器重启把进程杀掉时留下的**尾部残记录**，不影响读取，V6 数据到第 11969 步都是完整的。

## 轮次表

| V | 时间 | 任务 | envs | 轮数 | 回合长度 | track_lin_vel | 地形等级 | 结论 |
|---|---|---|---|---|---|---|---|---|
| **V1** | 10-14 19:56 | `LgDog1-Flat` | 4096 | 4926 | **993 / 1000** | **0.798** | 平地 | 平地基线，走得最顺（全程在 `<5000` 易指令区间） |
| **V2** | 10-14 22:50 | `LgDog1-Rough` | 1024 | 16000 | 852 / 1000 | 0.148 † | **0.00** ❌ | **带**地形感知；课程塌底 |
| **V3** | 10-15 10:37 | `LgDog1-Rough-Blind` | 2048 | 7301 | 780 / 1000 | 0.331 † | 0.00 ❌ | 盲跑基线，走不动 |
| **V4** | 10-15 15:47 | `LgDog1-Rough-Blind` | 2048 | 11545 | 870 / 1000 | 0.573 † | 0.00 ❌ | 加站立比例 0.25 |
| **V5** | 10-15 22:54 | `LgDog1-Rough-Blind` | 2048 | 16000 | 739 / 1000 | 0.511 † | 0.11 ❌ | 加**课程修复**（路径长度） |
| **V6** | 10-16 12:29 | `LgDog1-Rough-Blind-Gentle` | 2048 | 11969 | **909 / 1000** | 0.628 †（峰值 1.05） | **5.19** ✅ | ★ **温和地形 + 课程滞回 → 课程正常、目前最好** |
| **V7** | 10-16 19:39 | `LgDog1-Rough-Blind-HIM` | 1024 | **3982 / 16000** ⚠️ | 893 / 1000 | 1.635 † | 3.61 ↑ | HIM 历史观测；学到 25% 就被机器挂死打断 |
| **V7b** | 10-16 23:35 | `LgDog1-Rough-Blind-HIM` | 1024 | **4796**（手动停，896 轮） | 914 / 1000 | 1.623 † | 3.82 ↑ | 从 V7 的 `model_3900` 续训；22 分钟，各项小幅提升 |

> #### V7 → V7b 续训效果（同样在易指令区间，可直接比）
>
> | 指标 | V7 @3980 | **V7b @4795** | |
> |---|---|---|---|
> | Mean reward | 64.25 | **67.98** | ↑ |
> | 回合长度 | 893.55 | **913.51** | ↑ |
> | `error_vel_xy` | 0.9571 | **0.8678** | ↓ 更好 |
> | 地形等级 | 3.6071 | **3.8232** | ↑ |
> | `tracking_ratio` | 0.6963 | **0.7024** | ↑ |
> | `track_linear_velocity` | 1.6351 | 1.6227 | ≈ |
> | `fell_over` | 0.0000 | **0.0000** | 零摔倒 |
> | HIM `velocity_mse` | 0.0045 | **0.0041** | ↓ 估计器更准 |
>
> 896 轮提升不大但方向一致，说明从 V7 的 `model_3900` 继续训是有效的。
> 最后存盘 `logs/rsl_rl/lgdog1_velocity_blind_him_v1/2026-10-16_23-35-01_V7b_him_seed42/model_4700.pt`。
> **要接着训就从 `model_4700.pt` 起**（注意 rsl_rl 会把 max_iterations 累加，显示 19900 而非 16000）。

> ### ⚠️ V7 夭折记录（2026-10-16）
> 机器 **21:15 挂死** —— 日志戛然而止、无任何报错、无 traceback、`fell_over` 仍为 0，
> 是典型的被外部带走。用户 22:43 硬重启。这是 10-16 的**第 4 次**（前三次 10:38 / 12:06 / 19:17）。
> V7 死在 iteration 3982，最后一次存盘 `model_3900.pt`（`save_interval=100`，最多丢 100 轮）。
> **`crash_guard.sh` 黑匣子仍未重开**：`cd ~/rl/LgRobotDog && bash tools/crash_guard.sh start`
>
> V7 夭折时的成绩（iteration 3980）其实**全面优于 V6**：回报 64.25、`track_linear_velocity` 1.635
> （V6 同易指令区间约 1.0）、`error_vel_xy` 0.957（V6 同区间约 1.4）。
> sim2sim 平地实测 `--vx 0.6`：V7 走 **5.99 m**、跟踪误差 **0.099 m/s**；V6 只走 2.64 m、误差 0.380。
> 但 V7 在 10° 斜坡上会摔（10.26s，倾角 71°），V6 不摔 —— V7 只训了 25%，地形只到 3.61。

> 回合长度上限 = 1000 步（`episode_length_s=20` × 50 Hz）。
>
> † 这些轮次的 `track_lin_vel` 是**终值**，而 iteration 5000 处指令范围会放宽
> （`(-0.5,1.0)` → `(-1.0,2.0)`）。V6 在放宽前是 0.96~1.05，放宽后掉到 0.5~0.7，
> 是任务变难、不是策略变差。**跨轮次比这一列要取同一区间**，详见上面"别把 iter 5000 的台阶当成训练崩了"。
>
> ⚠️ **V7 不能参与这个比较**。`blind_terrain_env_cfg.py:97` 有
> `cfg.curriculum.pop("command_vel", None)` —— V7/blind-terrain 系列**没有指令范围课程**，
> 速度指令全程固定 `lin_vel_x(-0.4,0.8) / lin_vel_y(±0.3) / ang_vel_z(±0.6)`，
> 比 V6 的**易档还窄**（V6 易档 `(-0.5,1.0)`）。所以 V7 的 1.635 是在更简单的跟踪任务上拿到的，
> **不能和 V6 的 1.0 直接比**。
>
> 可比的口径只有 sim2sim：同样 `--vx 0.6`、同样平地 → V7 误差 **0.099 m/s**，V6 误差 **0.380 m/s**。
>
> 各任务的指令范围差异：
>
> | 任务 | lin_vel_x | lin_vel_y | ang_vel_z | 有指令范围课程？ |
> |---|---|---|---|---|
> | `LgDog1-Rough` / `-Blind` / `-Gentle` | (-0.5, 1.0) → 5000轮后 (-1.0, 2.0) | (±0.5) → (±1.0) | (±1.0) | ✅ 有 |
> | `LgDog1-Rough-Blind-HIM`（V7） | **(-0.4, 0.8) 固定** | **(±0.3) 固定** | **(±0.6) 固定** | ❌ 无 |

### 几个关键结论（跨轮次）

1. **地形课程曾经整体失效**（V2/V3/V4/V5 的 `terrain_levels` 全塌到 ~0）
   - 根因：课程用**净位移**判定升降级，而指令含转向时机器人走圆弧，净位移只有实际路径的 **45%**
   - 修法：改用**累计路径长度**（`src/tasks/velocity/mdp/metrics.py` 的 `traveled_distance`）
2. **只修课程还不够**（V5：路径 2.43 m 仍够不到 4.0 m 门槛，62% 回合被降级）→ 需要配合**更温和的地形**（V6）
3. **课程滞回**：降级阈值封顶在升级门槛的一半，形成 `[2.0, 4.0]` 稳定带（V6 的升降比 11.6% : 10.2%，基本平衡）
4. **盲跑（无地形感知）是可行的**：V6 观测只有 47 维、无 `height_scan`，实机**不需要雷达/高程图**

## 任务对照

| 任务 | 地形感知 | 观测维度 | 说明 |
|---|---|---|---|
| `LgDog1-Flat` | 无 | 47 | 平地 |
| `LgDog1-Rough` | **有**（187 维 height_scan） | 234 | 标准崎岖 |
| `LgDog1-Rough-Blind` | 无 | 47 | 标准崎岖，盲跑 |
| `LgDog1-Rough-Blind-Gentle` | 无 | 47 | **温和地形**（平地 40%、台阶 ≤5cm、坡度减半） |
| `LgDog1-Rough-Blind-TerrainV4` | 无 | 47 | 改进课程与接触终止 |
| `LgDog1-Rough-Blind-HIM` | 无 | **6×47=282 历史** + 估计器 | HIM 状态估计 |

## 显存上限（本机 8G 卡实测）

| 任务 | 最大 envs | 吞吐 |
|---|---|---|
| `LgDog1-Flat` | 4096 | 55.6k steps/s |
| `LgDog1-Rough`（234 维） | **1024** | 16.3k |
| `LgDog1-Rough-Blind`（47 维） | **2048** | 22.0k |
| `LgDog1-Rough-Blind-HIM`（282 维历史） | **1024** | 16.2k |

> 超了会报 `Warp CUDA error 2: out of memory`。指南里若写 2048 但你这卡跑不动，按本表降。

## 回放某一轮

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog && conda activate unitree_rl_mjlab
CKPT=$(ls -t logs/rsl_rl/<experiment>/<时间戳>_V<n>_*/model_*.pt | head -1)
python scripts/play.py <任务> --checkpoint_file="$CKPT" --num_envs=16 --viewer native
```

> `play.py` **必须**显式给 `--checkpoint_file`（这版的"自动找最新"分支是坏的）。

用 sim2sim 验（走 ONNX，与实机部署同一条链）：

```bash
python scripts/sim2sim.py <任务> \
  --onnx=logs/rsl_rl/<experiment>/<时间戳>_V<n>_*/policy.onnx \
  --map stairs --spawn-y 4.0 --vx 0.6 --viewer
```

## 续训

```bash
python scripts/train.py <任务> --env.scene.num-envs=<N> \
    --agent.resume=True --agent.logger=tensorboard \
    --agent.run-name=V<n>_<同原描述> --enable-nan-guard True
```

> 续训会新建一个 run 目录，只接权重/优化器/迭代数（`V<n>` 建议加 `b` 后缀区分，如 `V7b_...`）。

## 已知的坑（V 轮次里踩过的）

| 现象 | 原因 |
|---|---|
| `Curriculum/terrain_levels` 单调塌到 0 | 课程用净位移判定（见上）→ V5 修 |
| 原地转圈 / 站不住 | 训练里只有 5% 环境学"指令为 0 站立"→ V4 提到 0.25；sim2sim 侧用 `--heading-hold` |
| 5cm 台阶上不去 | 盲跑看不见台阶；`foot_clearance` 目标 0.10 正好等于台阶高 → 提到 0.15 |
| `Warp CUDA error 2: out of memory` | envs 超显存上限，见上表 |
| 训练跑一半消失 | 机器硬挂/重启；`crash_guard.sh` 黑匣子已停摆需重开 |
| `play.py` 报 `'PlayConfig' object has no attribute 'wandb_run_path'` | 没给 `--checkpoint_file` |
| TensorBoard 页面空白、`/data/runs` 返回 `[]` | 用相对路径 `--logdir` 启动且 cwd 后来被删 → 用 `tb.sh`（绝对路径） |
| sim2sim 报 `Policy shape [1,282] does not match 1 x 47` | 历史长度原本从 `--task` 反推；HIM 的 onnx 配 `LgDog1-Flat` 就错 → 已改为优先读 ONNX 元数据 |
| 续训报 `unrecognized arguments: --agent.checkpoint_path` | 没这个参数；正确的是 `--agent.load_run` + `--agent.load_checkpoint`（都是正则） |
