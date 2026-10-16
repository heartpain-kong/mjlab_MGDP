# unitree_rl_mjlab 常用命令速查

标准崎岖地形 + HIM 新任务：`LgDog1-Rough-Blind-HIM-Standard`。它删除 actor/critic 的机身高度扫描，详细命令见 [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md)；原来的 `LgDog1-Rough-Blind-HIM` 仍是 V7 温和地形版本。

> 本机环境已装好（conda 环境 `unitree_rl_mjlab`，DDS/SDK 在 `/usr/local`）。
> 环境怎么装的、踩过哪些版本坑 → 看 **`../DEPLOY_NOTES.md`**。
> **目录结构、训练数据流、以及「在哪里改 reward」→ 看 [`ARCHITECTURE.md`](ARCHITECTURE.md)**。
> 标 ✅ 的命令都是在本机实测跑通过的；标 📖 的是官方 README 里的写法（未逐条实测）。

---

## 0. 起手式

```bash
conda activate unitree_rl_mjlab
cd ~/mjlab/unitree_rl_mjlab_lgdog
```

⚠️ **别在别的 Python 里跑**：这个环境钉了 `mujoco==3.5.0` / `warp-lang==1.12.0`，
用系统 python 或别的 conda 环境会直接崩（原因见 DEPLOY_NOTES 的 A 节）。
换机器重装时 `pip install -e .` 之后**必须**补一句：

```bash
pip install "mujoco==3.5.0" "warp-lang==1.12.0" scipy
```

---

## 1. 看有哪些任务

```bash
python scripts/list_envs.py                 # ✅ 列出全部 31 个任务
python scripts/list_envs.py --keyword g1    # ✅ 只显示名字含 g1 的
python scripts/list_envs.py --keyword LgDog1  # ✅
```
> 过滤要写成 `--keyword xxx`，直接跟一个裸参数会被 tyro 当成「无法识别」。

当前可用任务（挑常用的）：

| 任务 ID | 说明 |
|---|---|
| `Unitree-G1-Flat` / `-Rough` | G1 人形，平地 / 崎岖 |
| `Unitree-G1-23Dof-Flat` / `-Rough` | G1 23 自由度版 |
| `Unitree-Go2-Flat` / `-Rough` | Go2 四足 |
| `Unitree-H1_2-Flat` / `-Rough` | H1_2 人形 |
| `Unitree-H2-Flat` / `-Rough` | H2 人形 |
| `Unitree-A2-Flat` / `-Rough` | A2 |
| `Unitree-As2-Flat` / `-Rough` | As2 |
| `Unitree-R1-Flat` / `-Rough` | R1 |
| **`LgDog1-Flat`** / **`-Rough`** / **`-Rough-Blind`** | **你们自研的狗（本机新移植进来的）** |
| `Unitree-G1-Tracking-No-State-Estimation` | G1 动作模仿 |
| `Unitree-G1-23Dof-Tracking-No-State-Estimation` | G1-23dof 动作模仿 |

---

## 2. 训练（速度跟踪）

### 基本用法

```bash
# ✅ 最小可跑（先拿它验证环境没问题）
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 \
    --agent.max-iterations=2 --agent.logger=tensorboard

# ✅ 正式训练（G1，实测就是这个数）
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=64 \
    --agent.logger=tensorboard

# 📖 大显存/官方示例值（本机独占 8G 也别直接上，见下面显存表）
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096 \
    --agent.logger=tensorboard

# ✅ 后台跑 + 存日志
nohup python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=64 \
    --agent.logger=tensorboard > train.log 2>&1 &
tail -f train.log
```

### ★ 显存（这台只有 8 G，实测）

| 场景 | `--env.scene.num-envs` | 实测 |
|---|---|---|
| 与那个 Isaac 训练（占 5.5 G）**同时跑** | **64 ~ 128** | 64 / 128 都跑通过；**256 会 CUDA OOM 崩掉**（`c10::Error`） |
| 独占 GPU | 从 512 往上试 | 本机没测过上限；官方示例给的是 4096（那是大卡） |

先 `nvidia-smi` 看一眼再决定，别硬来：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 多 GPU 📖

```bash
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096
```
> 本机只有 1 块卡，多卡分支没实测（`--gpu-ids 0 1` 的**参数解析**是通的）。
> 本机也可写 `--gpu-ids all`。

### 断点续训 ✅

```bash
# 自动找该实验目录下最新的 model_*.pt
python scripts/train.py LgDog1-Flat --agent.resume=True --agent.logger=tensorboard

# 指定 run / checkpoint
python scripts/train.py LgDog1-Flat --agent.resume=True \
    --agent.load-run=2026-10-14_14-27-01 --agent.load-checkpoint=model_39.pt
```
> `--agent.load-run` 只填 run 目录名（不是全路径），它在
> `logs/rsl_rl/<experiment_name>/` 底下找。

### 常用参数

| 参数 | 作用 |
|---|---|
| `--env.scene.num-envs=N` | 并行环境数（**最影响显存**） |
| `--agent.max-iterations=N` | 训练多少轮 |
| `--agent.save-interval=N` | 每 N 轮存一次 `model_*.pt`（默认 100） |
| `--agent.logger=tensorboard` | 用 tensorboard；默认是 `wandb`（要联网/登录） |
| `--agent.resume=True` | 断点续训 |
| `--agent.run-name=xxx` | 给 run 目录加后缀，好认 |
| `--agent.experiment-name=xxx` | 换实验名 → 换日志目录（不覆盖旧的） |
| `--env.episode-length-s=20` | 回合长度（秒） |
| `--env.decimation=4` | 每个控制步含几个物理步（控制频率 = 1/(dt×decimation)） |
| `--env.seed=42` | 固定随机种子 |
| `--video=True --video-interval=200` | 训练时录视频 |
| `--enable-nan-guard=True` | 开 NaN 守卫 |

> 参数名**下划线和连字符都认**（`--motion-file` ≡ `--motion_file`）。
> 完整帮助：`python scripts/train.py <任务名> --help`（很长，建议 `| less`）

### 产物位置 ✅

```
logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_run_name]/
├── model_<轮数>.pt      # checkpoint
├── policy.onnx          # ★ 实机部署用的就是这个
├── events.out.tfevents* # tensorboard 日志
├── params/{env,agent}.yaml   # 本次训练用的完整配置快照
└── git/                 # 跑的时候仓库的 diff
```

各任务的 `experiment_name`：G1 → `g1_velocity`，Go2 → `go2_velocity`，
**LgDog1 → `lgdog1_velocity`**。

---

## 3. 仿真回放（play）

> ★★ **`play.py` 可以不给 `--checkpoint_file`**（已修好）：会自动取该任务
> **最新 run 里最新的 `model_*.pt`**，并且会跳过「目录已建但还没存盘」的新 run。
> 所以**训练跑着的时候，反复重跑下面这一条就能看当前最新进度**：
>
> ```bash
> python scripts/play.py LgDog1-Rough-Blind-HIM --num_envs 4 --viewer native
> ```
>
> 训练与 play 可以并存（实测训练占 4.2G + play 4 envs 无冲突）。
> ⚠️ 例外：`LgDog1-Flat` 和 `LgDog1-Rough` 共用 `lgdog1_velocity` 目录，
> 自动选会拿错任务的权重 → 这两个任务请显式给 `--checkpoint_file` 或用 `--load-run`。
> 用下面这个一行命令拿最新 checkpoint：

```bash
# 通用写法：自动取最新 checkpoint（✅ 实测可用）
CKPT=$(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -1)
python scripts/play.py Unitree-G1-Flat --checkpoint_file="$CKPT" \
    --num_envs=16 --viewer native

# ✅ 指定 checkpoint
python scripts/play.py Unitree-G1-Flat \
    --checkpoint_file=logs/rsl_rl/g1_velocity/2026-10-14_13-57-41/model_1.pt \
    --num_envs=16 --viewer native

# ✅ LgDog1
CKPT=$(ls -t logs/rsl_rl/lgdog1_velocity/*/model_*.pt | head -1)
python scripts/play.py LgDog1-Flat --checkpoint_file="$CKPT" \
    --num_envs=4 --viewer native
```

### ★ LgDog1 各 V 轮次回放（直接抄）

**注意 `--checkpoint_file` 必须和任务名配套** —— V7(HIM) 的观测是 282 维（47×6 历史），
V6/V5 是 47 维单帧，**互不兼容**，配错会报维度错或静默跑出垃圾行为。

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog && conda activate unitree_rl_mjlab

# V7 —— HIM（观测 282 维，需要历史缓冲）★ 目前学得最好的一轮（但只训到 25%）
python scripts/play.py LgDog1-Rough-Blind-HIM \
  --checkpoint_file=logs/rsl_rl/lgdog1_velocity_blind_him_v1/2026-10-16_19-39-11_V7_him_seed42/model_3900.pt \
  --num_envs 4 --viewer native

# V6 —— 温和地形盲跑（47 维），课程已训通、地形等级 5.19，最稳
python scripts/play.py LgDog1-Rough-Blind-Gentle \
  --checkpoint_file=logs/rsl_rl/lgdog1_velocity_blind_gentle/2026-10-16_12-29-01_V6_blind_gentle/model_11800.pt \
  --num_envs 4 --viewer native

# V5 —— 标准崎岖盲跑
python scripts/play.py LgDog1-Rough-Blind \
  --checkpoint_file=logs/rsl_rl/lgdog1_velocity_blind/2026-10-15_22-54-53_V5_blind_curfix/model_15999.pt \
  --num_envs 4 --viewer native

# V1 —— 平地
python scripts/play.py LgDog1-Flat \
  --checkpoint_file=logs/rsl_rl/lgdog1_velocity/2026-10-14_19-56-16_V1_flat/model_4900.pt \
  --num_envs 4 --viewer native
```

`--num_envs` 别开大：play 会按环境数开 MuJoCo Warp 世界，本机 8G 显存
4~16 够看；开着训练时更要小（会 `Graph creation error`）。

> ⚠️ **play 模式的 `nconmax` 已修**。基类把 `nconmax` 限死在 35/世界（为训练省显存），
> 但 play 的地形是 5×5 + `border_width=10` 的大块地形，初始接触数偶尔会飙到 **159**，
> 于是随机地在启动时报 `ValueError: nconmax overflow (nconmax must be >= 159)`。
> 地形生成器 `seed=None`（每次随机）所以是**间歇性**的 —— 同一个 checkpoint 有时能起有时不能。
> 已在 `env_cfgs.py` 的 play 分支设 `cfg.sim.nconmax = None`（训练路径仍保持 35，不动显存）。

| `--viewer` | 说明 |
|---|---|
| `auto`（默认） | 有显示器用 `native`，没有就用 `viser` |
| `native` | 弹 MuJoCo 原生窗口（本机有 X，用这个） |
| `viser` | 浏览器可视化（headless/远程时用） |

其他常用：

```bash
python scripts/play.py Unitree-G1-Flat --agent zero        # 零动作（只看机器人能不能站住）
python scripts/play.py Unitree-G1-Flat --agent random      # 随机动作（看物理是否正常）
python scripts/play.py Unitree-G1-Flat --no-terminations   # 关掉终止条件（方便观察摔倒过程）
python scripts/play.py Unitree-G1-Flat --video=True --video-length=200   # 录视频
python scripts/play.py Unitree-G1-Flat --num-envs=1 --device=cpu         # 没 GPU 时
```
> `--agent zero` / `--agent random` 不需要 checkpoint。

---

## 4. 动作模仿（Tracking）

### 4.1 CSV → npz

```bash
python scripts/csv_to_npz.py \
  --robot g1 \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 \
  --output-fps 50
```

> ★ **README 这里写错了**：它说 npz 默认存到 `src/motions/g1/`，
> 实际代码里是 **`src/assets/motions/g1/`**（g1_23dof → `src/assets/motions/g1_23dof/`）。
> 生成后会打印路径，以打印为准。
> `--robot` 可选 `g1` 或 `g1_23dof`（也决定输出目录）。加 `--render=True` 会开窗口+录视频。

### 4.2 训练 📖

```bash
# 注意 num-envs：本机 8G 显存别超过 128（见第 2 节显存表）
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=128 \
  --agent.logger=tensorboard
```

### 4.3 回放

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint_file=logs/rsl_rl/g1_tracking/<时间>/model_<轮数>.pt
```

---

## 5. LgDog1（你们自研的狗）

资源文件：`src/assets/robots/lgdog1/`，任务注册：`src/tasks/velocity/config/lgdog1/`。

```bash
# ✅ 训练
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 \
    --agent.max-iterations=40 --agent.save-interval=10 --agent.logger=tensorboard

# ✅ 崎岖地形（带 187 维 height_scan 地形感知；本机显存上限 1024 envs）
python scripts/train.py LgDog1-Rough --env.scene.num-envs=1024 --agent.logger=tensorboard

# ✅ 崎岖地形【无地形感知】版 —— 关掉 187 维 height_scan，纯本体感觉，观测只有 47 维
#    好处：实机不需要高程图/地形传感器；训练也快（省掉每步 187 条射线）
#    代价：看不见前方地形，只能靠脚底反馈；10cm 台阶基本磕住
#    注意：checkpoint 存在 lgdog1_velocity_blind/，与带扫描版互不兼容
python scripts/train.py LgDog1-Rough-Blind --env.scene.num-envs=1024 --agent.logger=tensorboard

# ✅ 回放（记得给 checkpoint，play.py 不会自己找）
CKPT=$(ls -t logs/rsl_rl/lgdog1_velocity/*/model_*.pt | head -1)
python scripts/play.py LgDog1-Flat --checkpoint_file="$CKPT" --num_envs=4 --viewer native

# 改机器人本体参数（KP/KD/力矩/站姿/碰撞）→ 改这个文件
#   src/assets/robots/lgdog1/lgdog1_constants.py

# 改 MJCF → ★ 改源 XML 后重跑生成脚本，别手改 lgdog1.xml
#   python src/assets/robots/lgdog1/make_lgdog1_xml.py

# 改奖励/指令/观测范围 → src/tasks/velocity/config/lgdog1/env_cfgs.py
# 改网络/PPO 超参     → src/tasks/velocity/config/lgdog1/rl_cfg.py
```

> ⚠️ 改 MJCF 碰撞体名字时**必须保持"序号在 `_collision` 之前"**的命名
> （`FR_calf1_collision` ✅ / `FR_calf_collision_1` ❌）——写反了 mjlab 匹配不上，
> 足端会失去碰撞能力，机器人一出生就穿地自由落体。原因见 `../DEPLOY_NOTES.md` D 节。

### 5.4 看训练曲线（TensorBoard）

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog
bash tb.sh start          # 已在跑则自动跳过
bash tb.sh status         # 端口 + 当前识别到的 run 列表
bash tb.sh stop           # 停
bash tb.sh restart        # 重启
```

浏览器开 **http://localhost:6006**（局域网其他机器用 `http://192.168.28.158:6006`）。
日志 `logs/tensorboard/tensorboard.log`，pid `logs/tensorboard/tb.pid`。

左侧自动按任务分组，7 个 V 轮并排对比：`lgdog1_velocity/`（V1 V2）、
`lgdog1_velocity_blind/`（V3 V4 V5）、`lgdog1_velocity_blind_gentle/`（V6，目前最好）、
`lgdog1_velocity_blind_him_v1/`（V7）。轮次含义见 `TRAINING_V.md`。

重点曲线：

| 曲线 | 期望 |
|---|---|
| `Train/mean_reward` | 持续上升 |
| `Metrics/terrain_levels` | **必须上升**；塌到 0 = 课程判据失效（V5 踩过） |
| `Metrics/track_lin_vel` | 越高越好（V6 ≈ 0.63） |
| `Episode/mean_length` | 趋近 1000（20s 上限） |
| `Loss/value` / `Loss/surrogate` | 平稳；突爆会被 nan guard 拦下 |

> ⚠️ **`--logdir` 一定要用绝对路径**。踩过的坑：用相对路径启动、之后 cwd 被删，
> TensorBoard 进程还在、端口也通，但 `/data/runs` 返回 `[]`，页面全白。
> 自查：`curl -s localhost:6006/data/runs`，空的就是这毛病。
> 手动启动也行，但必须写绝对路径：
> ```bash
> tensorboard --logdir ~/mjlab/unitree_rl_mjlab_lgdog/logs/rsl_rl \
>   --host 0.0.0.0 --port 6006 --reload_multifile=true
> ```

---

## 5.5 sim2sim —— 把策略放进普通 MuJoCo 验证部署链路

`scripts/sim2sim.py`：不经过 mjlab 的 manager 体系、不用 mujoco_warp、不用 rsl_rl，
只用 **普通 mujoco + onnxruntime + 50 Hz PD 环**，与真机部署同一条路。
它验证的是 ONNX 接口、观测拼装、关节顺序、控制频率、PD 增益这些**实机会踩的坑**。

```bash
cd ~/mjlab/unitree_rl_mjlab_lgdog && conda activate unitree_rl_mjlab

# 站立自检
python scripts/sim2sim.py LgDog1-Flat --duration 10

# 给前进指令走（不指定 --onnx 会自动取最新 run 的 policy.onnx）
python scripts/sim2sim.py LgDog1-Flat --vx 1.0 --duration 15

# ★ 键盘遥控（弹窗口，按住即走、松手即停）
python scripts/sim2sim.py LgDog1-Flat --viewer

# ★ 与 mjlab 训练环境逐位对拍观测拼装（最严格的检查）
python scripts/sim2sim.py LgDog1-Flat --check-obs

# 指定某个导出的 onnx
python scripts/sim2sim.py LgDog1-Flat \
    --onnx=~/mjlab/export/lgdog1_flat_4900/policy.onnx --vx 0.5
```

### ★ 地图（测试地形）

> 宇树自带的地图**只有一块平地** —— `src/assets/robots/*/xmls/scene_*.xml` 的 worldbody 里
> 就一个 `<geom name="floor" type="plane">`（那些 scene 里的 box 是机器人自己的碰撞体），
> `simulate/mujoco/model/` 又是 MuJoCo 官方示例模型，也没地形。所以 sim2sim 自带了几张。

```bash
# 内置地形
python scripts/sim2sim.py LgDog1-Flat --map flat        # 默认，平地
python scripts/sim2sim.py LgDog1-Flat --map slope       # 10° 斜坡
python scripts/sim2sim.py LgDog1-Flat --map stairs      # 四组楼梯 5/10/15/20 cm
python scripts/sim2sim.py LgDog1-Flat --map obstacles   # 10~60 cm 高台列
python scripts/sim2sim.py LgDog1-Flat --map mixed       # 全都有
python scripts/sim2sim.py LgDog1-Flat --map lgdog1      # LgRobotDog 那张测试图（191 geom）

# 上某条楼梯车道：车道在 y=+4.0 / +1.0 / -2.0 / -5.0
python scripts/sim2sim.py LgDog1-Flat --map stairs --spawn-y 4.0 --vx 0.6 --viewer

# 用自己的地形 XML（只含 worldbody geom）
python scripts/sim2sim.py LgDog1-Flat --map-file=我的地形.xml
```

出生点用 `--spawn-x / --spawn-y / --spawn-yaw`（默认 0,0,0 = 原点面朝 +x）。
汇总里的 **`最终位置 x`** 是测地形的关键指标——被挡住就不会涨。

**实测（平地策略 vs 地形，vx=0.6 走 15s，不挡应该到 ~9 m）**：

| 地形 | 最终 x | 表现 |
|---|---|---|
| 平地（对照） | **+7.52 m** | 正常走 |
| 10° 斜坡 | +2.76 m | 爬上去了（z 0.32→0.50），但明显变慢 |
| 楼梯 5cm/级 | +1.29 m | 爬上了第一级（台阶从 x=1.0 起），再上不去 |
| 楼梯 10cm/级 | +1.03 m | **卡死在台阶立面** |
| 楼梯 15cm/级 | +1.03 m | 同上 |
| LgRobotDog 图 5cm 梯 | +1.52 m | 略好一点 |

> 这说明**只在平地训过的策略**（`LgDog1-Flat`，观测里也没有 height_scan）
> 能应付 10° 坡和 5cm 台阶，但 **≥10cm 的台阶上不去** —— 符合预期。
> 想上楼梯得训 `LgDog1-Rough`（带地形课程 + 高度扫描）。

### 键盘（pynput 全局钩子，不挑窗口焦点）

| 键 | 功能 | 键 | 功能 |
|---|---|---|---|
| `W` / 小键盘 `8` | 前进 | `S` / 小键盘 `2` | 后退 |
| `A` / 小键盘 `4` | 左转 | `D` / 小键盘 `6` | 右转 |
| `Q` / 小键盘 `7` | 左移 | `E` / 小键盘 `9` | 右移 |
| `空格` / `R` | 复位 | `Esc` | 退出 |

速度用 `--speed-x` / `--speed-y` / `--speed-yaw` 调（默认 1.0 / 0.5 / 1.0）。
依赖 `pynput`（已装）；`--no-keyboard` 可关掉。

**朝向保持**（`--heading-hold`，有键盘时默认开）：策略观测里没有绝对朝向，
开环下发 wz 会慢慢画圈。开了以后不转向时复刻训练的 heading 控制律
`wz = clip(0.5 × wrap(目标朝向 − 当前朝向), ±1.0)` 保持朝向。
实测直行 20s：**关 → 漂移 +5.3°；开 → +1.1°**。

参数：`--vx --vy --wz`（指令）、`--duration`（秒）、`--viewer`、`--check-obs`、
`--no-keyboard`、`--heading-hold/--no-heading-hold`、`--speed-*`、`--action-clip`、
`--map`、`--map-file`、`--spawn-x/--spawn-y/--spawn-yaw`

> ⚠️ 脚本里有两处**必须与训练强绑定**的东西，改训练配置要同步改：
> ① `CTRL_DT / PHASE_PERIOD / CMD_THRESHOLD / SPAWN_HEIGHT / GOAL_JOINT`（文件头常量）；
> ② `apply_mujoco_cfg()` 会把任务的 `SimulationCfg.mujoco` 原样写到 `MjModel.opt` ——
> **这步不能省**，裸 `MjSpec.compile()` 的默认积分器是 Euler，而训练用的是 `implicitfast`，
> 不搬过去同一个策略会当场乱晃。

---

## 6. 导出 ONNX → 实机部署

训练时**每次存 checkpoint 会同步导出 `policy.onnx`** 到同一目录，不用额外命令 ✅。

```bash
# 1) 看 ONNX 的输入输出维度（确认观测/动作对齐）
python -c "
import onnx; m = onnx.load('logs/rsl_rl/lgdog1_velocity/<时间>/policy.onnx')
for i in m.graph.input:  print('in ', i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output: print('out', o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])"
# LgDog1-Flat 实测：in obs[1,47] → out actions[1,12]

# 2) 拷到部署目录（以 G1 速度控制为例）
ONNX=$(ls -t logs/rsl_rl/g1_velocity/*/policy.onnx | head -1)
cp "$ONNX" deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx

# 3) 编译控制器（6 个机器人都能编）
cd deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make -j

# 4) 仿真部署
cd ~/mjlab/unitree_rl_mjlab_lgdog/simulate
./build/unitree_mujoco                    # ★ 必须插着手柄才起得来
cd ~/mjlab/unitree_rl_mjlab_lgdog/deploy/robots/g1/build
./g1_ctrl --network=lo

# 5) 实机部署（先按 README 把本机 IP 设成 192.168.123.222/24）
./g1_ctrl --network=enp5s0                # 网卡名用 ifconfig 查
```

现成的策略文件：只有 **g1** 有（`velocity/v0` 和 `mimic/dance1_subject2` 两个）；
其余机器人的 `policy/` 目录是空的，要自己训练后补。

---

## 7. 训练监控

```bash
conda activate unitree_rl_mjlab
tensorboard --logdir=logs/rsl_rl --port 6006
# 浏览器打开 http://localhost:6006

# 只看某一个
tensorboard --logdir=logs/rsl_rl/lgdog1_velocity --port 6007
```

**命令行看关键指标**（不用开浏览器）：

```bash
tail -f train.log | grep -E "Learning iteration|Mean reward|Mean episode length|illegal_contact"
```

### 指标怎么读

| 指标 | 正常 | 说明 |
|---|---|---|
| `Mean episode length` | 持续上升 | **最主要的学习信号**；一直卡在几十步说明没学会 |
| `Mean reward` | 上升 | 有噪声，看趋势 |
| `Episode_Reward/track_linear_velocity` | 上涨 | 速度跟踪 |
| `Episode_Termination/time_out` | 后期升高 | 熬到回合结束 = 学会了 |
| `Episode_Termination/fell_over` | 后期降低 | 摔倒 |
| `Episode_Termination/illegal_contact` | 早期高正常 | 机身/腿触地（非足端） |
| `Steps per second` | — | 吞吐，掉得厉害通常是显存/CPU 抢 |

> 判据：刚起步时 `Mean episode length` 只有几十步是**正常的**（策略还随机）；
> 撑到几百步并继续涨才是真的在学。

---

## 8. 环境依赖（出问题时先看这里）

```bash
# DDS / SDK（部署用，装在 /usr/local）
ls /usr/local/lib/libddsc.so /usr/local/lib/libddscxx.so /usr/local/lib/libunitree_sdk2.a

# 编译器/工具
cmake --version && g++ --version

# 部署依赖的 apt 包
dpkg -s libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev \
        libfmt-dev libglfw3-dev bison flex
```

装环境（新机器）：

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                    libspdlog-dev libfmt-dev libglfw3-dev bison flex libcunit1-dev python3-venv
# unitree_sdk2 自带预编译的 CycloneDDS 0.10.2（含头文件），装它就够了：
cd <unitree_sdk2>/build && sudo make install
sudo ln -sf libddsc.so /usr/local/lib/libddsc.so.0
sudo ln -sf libddscxx.so /usr/local/lib/libddscxx.so.0
sudo ldconfig
```

---

## 9. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'mujoco_warp'` 或 `warp.context` 报错 | 版本被 pip 升上去了。补跑 `pip install "mujoco==3.5.0" "warp-lang==1.12.0" scipy` |
| `AttributeError: mjtEnableBit has no attribute mjENBL_MULTICCD` | 同上，`mujoco` 版本太新 |
| `ModuleNotFoundError: No module named 'scipy'` | `pip install scipy`（mjlab 用了但没声明依赖） |
| 训练报 `c10::Error` / CUDA OOM | 显存不够。降 `--env.scene.num-envs`；先 `nvidia-smi` 看谁占着 |
| `g1_ctrl: undefined symbol: ...EntityDelegate::listener_set` | `/usr/local` 里的 CycloneDDS 版本和 unitree_sdk2 预编译库不匹配。**只用 unitree_sdk2 自带的那套 0.10.2**，别自己另装 0.10.5 |
| 仿真器起不来 `Error: Joystick open failed.` | 没插手柄。unitree_mujoco 必须先插手柄（想绕开就临时把 `simulate/config.yaml` 的 `use_joystick` 改 0） |
| `g1_ctrl` 一直 `Waiting for connection rt/lowstate` | 仿真器没起来 / `--network` 填错（仿真用 `lo`，真机填实际网卡） |
| `--gpu-ids 0` 报 tyro 解析错 | 单个值它不认成列表。省略（默认 `[0]`）或写 `--gpu-ids 0 1` / `--gpu-ids all` |
| `play.py` 报 `'PlayConfig' object has no attribute 'wandb_run_path'` | 没给 `--checkpoint_file`，而代码里残留了已被删除的 W&B 字段引用 → **已修**，现在抛明确的 `ValueError` 并列出可选 run |
| `play.py` 随机启动失败：`nconmax overflow (nconmax must be >= 159)` | 基类 `nconmax=35` 对 play 的大地形不够，地形 `seed=None` 所以**间歇性** → **已修**（play 分支设 `nconmax=None`，训练不动） |
| `list_envs.py LgDog1` 报 `Unrecognized options` | 过滤要写成 `--keyword LgDog1` |
| `LgDog1-Flat` 在 `list_envs` 里找不到 | `src/assets/robots/__init__.py` 的导出被改坏了，或 `src/tasks/velocity/config/lgdog1/` 缺文件 |
| 训练日志写进 `wandb` 而不是 tensorboard | 加 `--agent.logger=tensorboard` |
| 网络：`pip` / `conda` / GitHub 拉不动 | 见 `../DEPLOY_NOTES.md` 末尾「网络环境备忘」（清华源 + ghfast 前缀 + SJTU 的 torch CPU 轮子） |

---

## 10. 一分钟自检

新环境或改完代码，跑这一串确认没坏：

```bash
conda activate unitree_rl_mjlab && cd ~/mjlab/unitree_rl_mjlab_lgdog

python scripts/list_envs.py --keyword LgDog1            # 任务在不在
python scripts/train.py LgDog1-Flat --env.scene.num-envs=64 \
    --agent.max-iterations=2 --agent.logger=tensorboard  # 训练能不能跑
ls logs/rsl_rl/lgdog1_velocity/*/policy.onnx             # ONNX 有没有导出

CKPT=$(ls -t logs/rsl_rl/lgdog1_velocity/*/model_*.pt | head -1)
python scripts/play.py LgDog1-Flat --checkpoint_file="$CKPT" \
    --num_envs=2 --viewer native                         # 回放能不能跑
```
