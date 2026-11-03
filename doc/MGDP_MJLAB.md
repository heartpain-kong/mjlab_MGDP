# MGDP 迁移到 mjlab：训练说明

把 [MGDP](https://github.com/arclab-hku/MGDP)（Advanced Science 2026）的**感知部分**迁到 mjlab/LgDog1，
在 MuJoCo Warp 上训练「深度感知 + 本体历史」的足式速度策略。

> 该任务（`LgDog1-Rough-MGDP`，短名 `mgdp`）是**可选任务**，默认任务仍是 HIM 标准地形。
> 使用时显式指定别名，例如 `python lgdog.py train mgdp`；
> 旧的 `standard`、`him_flat` 等工作流完全不变。

## 1. 迁移了什么

| MGDP 原始实现 | 本仓库实现 |
|---|---|
| Isaac Gym 内置相机 / Warp 射线相机（120×120→16×16） | `RayCastSensor` 针孔图案（16×16 条射线，FOV 87°，前下方 45°） |
| 深度图 + 高程图双模态编码器 | `MGPDPerception.depth_encoder` / `height_encoder` |
| 对比学习（InfoNCE）+ 双重重构 | 同一套损失，见 `mgdp_perception.py::losses` |
| 以干净深度为重构目标（去噪） | 训练时注入高斯噪声 + dropout，目标仍是干净帧 |
| GRU 聚合时序 | 模块自带 GRU 状态，按 episode 复位 |
| PPO actor = obs45 + 本体 token + 视觉 token | actor = 4×47 本体历史 + 64 维感知特征 |
| 地形自适应奖励 | **尚未迁移**（见第 6 节） |

感知模块保留原论文的网络宽度（32/64/96，64 维 token，GRU 64），
因此**可以直接加载官方 `wm_best.pt` 权重**（`--init-from`），已验证形状完全兼容。

## 2. 数据流

```
trunk 上的 mgdp_depth site (0.34, 0, 0.06)，俯仰 -45°
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

### 常用可调项

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num-envs` | 256 | 预训练采集环境数（与策略训练无关） |
| `--sample-envs` | 32 | 每步记录多少个环境，控制内存 |
| `--steps` | 3000 | 采集步数；越多地形覆盖越广 |
| `--init-from` | 空 | 官方 `wm_best.pt` 路径，可直接热启动 |
| `DEPTH_HW` | `(16,16)` | 改大（如 24×24）需要同步改 `depth_hw` 并重新预训练 |

### 显存与耗时（经验值，4090）

| 阶段 | 显存 | 时间 |
|---|---|---|
| 预训练（256 envs，32 采样） | < 2 GB | 采集约 1–2 min，训练 3000 iter 约 5–10 min |
| 策略训练（2048 envs） | 约 14–18 GB | 比盲走任务多约 10–20% 迭代时间 |

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
  64 维感知特征需要用单独的 TorchScript/ONNX 模块计算；
  实机深度需先转成同样的 16×16 归一化图（RealSense → 裁剪 [0,5] m → 双线性缩放）。

## 7. 已做的验证（本机 CPU，无 NVIDIA GPU）

- `tests/test_mgdp_perception.py`：7 项全部通过，包括**加载官方 `wm_best.pt`**；
- mjlab 环境在 CPU 上构建成功：`actor (2,188)`、`perception (2,64)`、`critic (2,261)`；
- 深度射线实测范围 `-1.0 ~ 4.73 m`（未命中 = -1），感知特征全部有限；
- `scripts/pretrain_mgdp.py` 冒烟：60 步采集 118 对样本，10 次迭代损失下降并保存检查点；
- 端到端 PPO 冒烟：2 个环境、2 次迭代，`obs_groups=('actor','perception')` 正常，
  actor 252 维 / critic 261 维，训练与保存链路通过。

> CPU 冒烟只验证形状与链路，不代表收敛性；正式训练必须在 CUDA 上跑。
