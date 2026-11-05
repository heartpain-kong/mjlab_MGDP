# MGDP 迁移到 mjlab：训练说明

把 [MGDP](https://github.com/arclab-hku/MGDP)（Advanced Science 2026）的**感知部分**迁到 mjlab/LgDog1，
在 MuJoCo Warp 上训练「深度感知 + 本体历史」的足式速度策略。

> 该任务（`LgDog1-Rough-MGDP`，短名 `mgdp`）是**可选任务**，默认任务仍是 HIM 标准地形。
> 使用时显式指定别名，例如 `python lgdog.py train mgdp`；
> 旧的 `standard`、`him_flat` 等工作流完全不变。

## 1. 迁移了什么

2026-11-04：MGDP 的 reward、接触和能力门控延续 v2，`foot_clearance=0`、`feet_air_time=0.25`。
当前 `race_mgdp` 已升级为 **2027 场地图 v3**，包含双排梅花桩、坡台沟壑、L 形绕桩/砖木坑路线；
比赛感知文件改为独立的 `logs/mgdp/perception/mgdp_race_v3.pt`。
使用与验证见 [MGDP_PASSAGE.md](MGDP_PASSAGE.md)，下文比赛随机地形描述属于 v1 的历史背景。

| MGDP 原始实现 | 本仓库实现 |
|---|---|
| Isaac Gym 内置相机 / Warp 射线相机（120×120→16×16） | `RayCastSensor` 针孔图案（16×16 条射线，FOV 87°，前头顶水平朝前） |
| 深度图 + 高程图双模态编码器 | `MGPDPerception.depth_encoder` / `height_encoder` |
| 对比学习（InfoNCE）+ 双重重构 | 同一套损失，见 `mgdp_perception.py::losses` |
| 以干净深度为重构目标（去噪） | 训练时注入高斯噪声 + dropout，目标仍是干净帧 |
| GRU 聚合时序 | 连续序列上的 GRU 特征重构损失；推理状态按 episode 复位 |
| PPO actor = obs45 + 本体 token + 视觉 token | actor = 4×47 本体历史 + 64 维感知特征 |
| 地形自适应奖励 | **尚未迁移**（见第 6 节） |

感知模块保留原论文的网络宽度（32/64/96，64 维 token，GRU 64），
因此**可以直接加载官方 `wm_best.pt` 权重**（`--init-from`），已验证形状完全兼容。

## 2. 数据流

```
trunk 上的 mgdp_depth site (0.02, 0, 0.075)，前头顶、光轴水平朝前（站立时与地面平行）
        │
        ▼
RayCastSensor "depth_rays"  16×16 = 256 条射线，max_distance=5 m
        │  distances (B,256)，单位米，未命中为 -1
        ▼
归一化 clip(0,5)/5，未命中→1.0，两帧堆叠 (B,2,16,16)
        │
        ▼
MGPDPerception（冻结）：U-Net 深度编码器 → 64 维 token → GRU(64)
        │
        ├──► 观测组 "perception" (B,64) ─┐
        │                                 ├─► actor 输入 188+64=252
本体 47 维 × 4 帧历史 ──────────────────┘
        │
        ▼
PPO（rsl_rl，普通 MLP actor，无 HIM 估计器）
```

critic 仍然使用特权信息（187 维高度扫描、足端接触、线速度），不接感知特征——
和 MGDP 的非对称 actor-critic 一致。

## 3. 新增/修改的文件

| 文件 | 作用 |
|---|---|
| `src/tasks/velocity/mdp/mgdp_perception.py` | 网络、损失、检查点读写、官方权重加载 |
| `src/tasks/velocity/mdp/mgdp_observations.py` | 观测项 `mgdp_depth_feature`（读传感器 + GRU 状态管理） |
| `src/tasks/velocity/config/lgdog1/mgdp_env_cfg.py` | 机器人加深度 site、深度射线传感器、观测组 |
| `src/tasks/velocity/config/lgdog1/mgdp_rl_cfg.py` | `obs_groups` 与 PPO 配置 |
| `src/tasks/velocity/config/lgdog1/__init__.py` | 注册 `LgDog1-Rough-MGDP` |
| `scripts/pretrain_mgdp.py` | MGDP 阶段 1：采集深度/高程对并训练感知模型 |
| `tests/test_mgdp_perception.py` | CPU 单元测试（7 项） |

## 4. 在 GPU 服务器上训练

```bash
cd unitree_rl_mjlab_lgdog
source .venv/bin/activate          # 或 bash install_ubuntu.sh

# 1) 预训练感知模型（约 3000 步采集 + 3000 次迭代）
python scripts/pretrain_mgdp.py \
  --num-envs 256 --sample-envs 32 --steps 4000 \
  --iterations 3000 --batch-size 256 \
  --out logs/mgdp/perception/mgdp_perception.pt

# 2) 训练策略（加载上面冻结的感知模型）
python scripts/train.py LgDog1-Rough-MGDP \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 10000 \
  --agent.logger tensorboard \
  --gpu-ids 0

# 3) 回放
python scripts/play.py LgDog1-Rough-MGDP \
  --checkpoint_file logs/rsl_rl/lgdog1_velocity_mgdp_v1/<run>/model_XXXX.pt
```

任务名已出现在 `python scripts/list_envs.py` 的第 9 项。

也可以在面板的任务卡片里直接点 **「预训练感知模型」** 按钮，日志会进入「运行记录」。
Rough 默认 256 环境；比赛默认 64 环境、4×10 采集地图、batch 64；均采集 4000 步、训练 3000 次迭代。
已有权重时仍显示 **「重新预训练感知模型」**，完成后替换当前任务默认感知权重；正在运行或停止时禁用重复启动。
完成状态按本次运行结果判断，不会因为旧权重存在就提前显示成功；失败或停止后可以重试。
旧策略回放必须配套原感知模型。新checkpoint会绑定同run的感知快照，Play/续训自动恢复；
旧checkpoint优先用该run的 `policy.perception.pt`，不会自动混用重新预训练的公共文件。
缺失原文件及显式指定方法见 [Play感知配对说明](MGDP_CHECKPOINT_PAIRING.md)。

### 2027 比赛地形（`LgDog1-Race-MGDP`）

- 地形：当前采用 `passage_terrain_cfg.py` 的 10 类密集课程，包括限高框架、轮胎阵、
  高墙、随机起伏等，具体尺寸和路线见 [定向越障说明](MGDP_PASSAGE.md)。
  感知采集默认缩小到 **4 行×10 列**，每列固定一种障碍，每行分层覆盖难度，保留全部 10 类，
  不改变策略训练的 10×20 地图和 HIMLoco 全强度域随机化；
- 相机：前头顶 `(0.02, 0, 0.075)`，光轴水平朝前，FOV 87°（示意图见
  `E:\rl_vision\deploy\out\camera_mount_diagram.png`）；
- 训练：`python lgdog.py train race_mgdp --envs 2048 --terrain-num-rows 8 --terrain-num-cols 20 --iterations 16000`（同样需要先预训练感知）；
  初始化曾在12×20大图下触发主机RAM OOM，原因和小地图预设见 [RAM说明](RACE_MGDP_RAM_OOM.md)；
  当前保留20列的显存优化与3072试运行见 [显存说明](MGDP_VRAM_3072.md)，实际容量须在本机确认；
- 真场地复测：`python scripts/sim2sim.py LgDog1-Race-MGDP --onnx=... --map race`。

比赛预训练可直接运行（在项目根目录并激活训练环境）：

```bash
python -u -m scripts.pretrain_mgdp --task LgDog1-Race-MGDP \
  --num-envs 64 --terrain-rows 4 --terrain-cols 10 --batch-size 64 \
  --steps 4000 --iterations 3000
```

初始化、采集、释放仿真、网络训练、保存分别打印 `[init]`、`[collect]`、`[release]`、`[train]`、`[save]`。
采集环境均匀分布在所有环境编号上，避免只记录前几类障碍；采集后关闭并删除仿真对象，
回收引用并清理 PyTorch CUDA 缓存，再开始感知网络训练。验证集保留在 CPU，分批传到设备。
验证 InfoNCE 的负样本范围现在是单个验证批次，指标不能直接与旧版整集计算的数值比较。
最终先写临时文件，完成后再替换默认 `.pt`，避免保存中断损坏已有权重。

`已结束` 仅表示进程已退出；成功应看到 `[save]` 和最终 `[val ]`，且默认权重存在。
若日志无 Python 异常却突然结束，仍需退出码/系统日志判定是否被 OOM killer 终止，不能仅据此断定。
8 GB GPU 若仍提示 CUDA OOM，可先停止仍在占用显存的 Play，再尝试 32 环境、2 行、batch 32；
资源缩减未在所有 GPU 上验证，不保证特定显存容量一定可用。

2026-11-05 本地验证：全套 274 项测试，0 失败、1 跳过；实际 CPU 使用 10 环境、2×10 地图，
采集 16 步得到 30 条连续 4 帧序列，训练 2 次，原子保存后成功重新加载。
记录位于忽略目录 `.artifacts/pretrain-resources/`。这只验证链路与配置，
未复现远端 Linux 的退出原因，也未验证 RTX 4060 Laptop 的完整 3000 次训练。

### 常用可调项

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num-envs` | Rough 256 / 比赛 64 | 预训练采集环境数（与策略训练无关） |
| `--terrain-rows` | Rough 12 / 比赛 4 | 仅调整预训练地图 |
| `--terrain-cols` | Rough 原配置 / 比赛 10 | 比赛至少10列以保留全部障碍类型 |
| `--batch-size` | Rough 256 / 比赛 64 | 训练及验证的序列批大小，影响显存和 InfoNCE 负样本数 |
| `--sample-envs` | 32 | 每步记录多少个环境，控制内存 |
| `--steps` | 3000 | 采集步数；越多地形覆盖越广 |
| `--init-from` | 空 | 官方 `wm_best.pt` 路径，可直接热启动 |
| `DEPTH_HW` | `(16,16)` | 改大（如 24×24）需要同步改 `depth_hw` 并重新预训练 |

### 显存与耗时（旧版经验值，4090）

下表对应修复前的单样本预训练。连续序列训练的有效帧批量增大，GPU 显存和耗时尚未重测，应先小批量运行。

| 阶段 | 显存 | 时间 |
|---|---|---|
| 预训练（256 envs，32 采样） | < 2 GB | 采集约 1–2 min，训练 3000 iter 约 5–10 min |
| 策略训练（2048 envs） | 约 14–18 GB | 比盲走任务多约 10–20% 迭代时间 |

## 预训练与部署修复（2026-11-04）

预训练默认使用 `--sequence-length 8` 的连续序列，跨 episode 的序列整条丢弃。
保留深度/高程重构与对比损失，新增 GRU 输出对干净深度 token 的 MSE（`temporal_weight=1.0`）。
这是本工程为训练循环权重增加的辅助目标；梯度通过整个序列回传。
旧版不带 `--init-from` 的预训练没有训练 GRU，已有检查点不能因此视为已修复。
重新预训练后应启动新的策略训练；已有策略回放需要继续使用其训练时的原感知权重。

新策略导出会同时写入 `policy.onnx` 和 `policy.perception.pt`，ONNX metadata 记录
感知文件名、SHA-256、历史长度、拼接顺序及控制周期。部署时需一起复制两个文件：

```bash
python scripts/sim2sim.py LgDog1-Race-MGDP --onnx path/to/policy.onnx --map race --vx 0.4 --no-keyboard
```

CPU MuJoCo 通过同一针孔射线图案、相机 site 姿态、几何组和父 body 排除配置读取深度，
按训练侧归一化及两帧堆叠计算 GRU 特征；四帧本体历史按“逐观测项内部由旧到新”展开，
再拼接 64 维感知特征。键盘复位同时清空两帧深度、GRU 状态和本体历史。

旧 ONNX 未绑定感知权重，需要明确给出原感知检查点：

```bash
python scripts/sim2sim.py LgDog1-Race-MGDP --onnx old/policy.onnx --perception old/original_perception.pt --map race --vx 0.4 --no-keyboard
```

旧格式没有指纹，调用者需确认所给文件来自该策略训练。新格式会拒绝指纹不同的文件。

## 5. 与论文实现的差异

1. **深度来源**：论文用相机渲染，这里用射线（`RayCastSensor`）。
   少了一次渲染，4096 并行时开销小得多；代价是没有纹理/遮挡语义，只有几何深度。
2. **分辨率**：16×16，与论文一致；射线数固定为 256。
3. **感知冻结**：论文 stage 2 会联合微调感知模块；本实现先冻结（更稳、可复现），
   联合微调可作为下一步（在 runner 里加辅助损失）。
4. **没有地形自适应奖励**：这是 MGDP 的另一半贡献，依赖逐地形权重表；
   mjlab 的奖励函数签名不同，需要单独移植。
5. **没有 HIM 估计器**：用 4 帧本体历史直接进 MLP，等价于 MGDP 的
   `proprio_token` 路径，但少了显式的速度/足高监督。

## 6. 下一步

- **地形自适应奖励**：在 `reward_cfg.py` 里按 `terrain_type` 切换惩罚权重；
  需要 mjlab 暴露每个 env 当前地形类别（可从 `terrain_levels` / 地形 id 推导）。
- **感知联合微调**：在 runner 的 `learn` 循环里加一次 `losses()` 反向传播，
  用 rollout 中缓存的深度/高程对做自监督。
- **真机部署**：`policy.onnx` 只包含 actor（输入 252 维），
  64 维感知特征目前由 sim2sim 加载配套 PyTorch 感知模型计算；若实机要求纯 ONNX，还需单独导出该循环模型；
  实机深度需先转成同样的 16×16 归一化图（RealSense → 裁剪 [0,5] m → 双线性缩放）。

## 7. 已做的验证（本机 CPU，无 NVIDIA GPU）

- `tests/test_mgdp_perception.py`：7 项全部通过，包括**加载官方 `wm_best.pt`**；
- mjlab 环境在 CPU 上构建成功：`actor (2,188)`、`perception (2,64)`、`critic (2,261)`；
- 深度射线实测范围 `-1.0 ~ 4.73 m`（未命中 = -1），感知特征全部有限；
- `scripts/pretrain_mgdp.py` 冒烟：60 步采集 118 对样本，10 次迭代损失下降并保存检查点；
- 端到端 PPO 冒烟：2 个环境、2 次迭代，`obs_groups=('actor','perception')` 正常，
  actor 252 维 / critic 261 维，训练与保存链路通过。

> CPU 冒烟只验证形状与链路，不代表收敛性；正式训练必须在 CUDA 上跑。
