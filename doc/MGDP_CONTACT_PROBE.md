# 接触丢失探针：本地 vs GPUFree（race_mgdp）

## 症状

`lgdog1_velocity_race_mgdp_v3` 下所有大环境数的 run（V4_new 6144、V5 5120、V6 5120、V7 5120、V8 4096）
地形等级一直是 0，末 100 迭代：

| 指标 | 正常（MGDP_v3_V4, 1024 环境） | 失败 run（5120/6144） |
|---|---:|---:|
| `Metrics/air_time_mean` | 0.05–0.12 s | **4.5 s** |
| `Metrics/slip_velocity_mean` | 0.10 | **0.000** |
| `Metrics/landing_force_mean` | 43–55 | **0.000** |
| `Metrics/leg_contact_force_max` | 0.17 | **0.000** |
| `Episode_Metrics/velocity_tracking_score` | 0.71 | 0.003 |
| 回合长度 | 1180 → 743 步 | 388–409 步 |
| `left_lane` 终止占比 | 12.8% | 95–97% |

奖励权重不是变量：V4_new 用的就是 V4 的权重，从第 1 个迭代起接触就是 0。

## 探针

```python
# probe_one.py  n  nconmax  optimize(0/1)
import sys, torch
from mjlab.envs import ManagerBasedRlEnv
from src.sim_memory import compact_model_allocations
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg

n, ncon, opt = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
cfg = lgdog1_race_mgdp_env_cfg(with_perception=False)
cfg.scene.num_envs = n
cfg.scene.terrain.terrain_generator.num_rows = 8
cfg.sim.nconmax = None if ncon == "auto" else int(ncon)
with compact_model_allocations(opt):
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
try:
    env.reset()
    for _ in range(5):
        env.step(torch.zeros((n, env.action_space.shape[-1]), device="cuda:0"))
    d = env.scene["feet_ground_contact"].data
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    print(f"RESULT n={n} nconmax={ncon} optimize={opt} "
          f"found={d.found.float().mean().item():.3f} "
          f"force={d.force.abs().mean().item():.2f} z={z.mean().item():.3f}", flush=True)
finally:
    env.close()
```

每个配置单独起进程（同一进程里连建多个环境，显存累积会污染后续结果）。

## 结果

### GPUFree（L40 44 GiB）

| 环境数 | nconmax | optimize | found | force | z |
|---|---:|---:|---:|---:|---:|
| 4 | 35 | 1 | 0.938 | 9.34 | 0.299 |
| 64 | 35 | 1 | 1.023 | 11.32 | 0.300 |
| 512 | 35 | 1 | 1.019 | 11.41 | 0.300 |
| **1024** | 35 | 1 | **0.000** | 0.00 | **0.259** |
| 1024 | 35 | 0 | 0.000 | 0.00 | 0.258 |
| 1024 | 200 | 1 | 0.000 | 0.00 | 0.258 |
| 1024 | auto | 1 | 0.000 | 0.00 | 0.258 |
| 2048 | 35 | 1 | 0.000 | 0.00 | 0.258 |

### 本地（Windows + 8 GiB，HEAD `80afae0`，torch 2.14.1+cu130）

| 任务 | 环境数 | rows | found | force | z |
|---|---:|---:|---:|---:|---:|
| blind | 512 | 10 | 1.038 | 11.14 | 0.308 |
| blind | 1024 | 10 | 1.033 | 11.13 | 0.306 |
| blind | 1536 | 10 | 1.034 | 11.06 | 0.307 |
| race | 128 | 2 | 1.027 | 11.51 | 0.300 |
| race | 256 | 2 | 1.016 | 11.48 | 0.299 |
| race | 512 | 2 | 1.024 | 11.35 | 0.300 |
| race | 128 | 8 | 1.016 | 11.38 | 0.299 |
| race | 256 | 8 | 1.038 | 11.55 | 0.299 |
| race | 384 | 8 | 1.008 | 11.39 | 0.299 |

本地到 1536 世界都没有退化，服务器 1024 就归零 → 不是"配对分块对世界数天然失效"。

## 现象解读

1. `z = 0.258` 在 1024/2048/`optimize=0`/`nconmax=200` 四种配置下**逐位相同**，是自由下落 5 步（0.1 s）的位移（≈4.7 cm），说明物理在跑，但没有地面接触。
2. `nconmax`、`optimize`、`broadphase` 路径都改过，结果不变 → 不是这三项。
3. 服务器日志里每次 1024 构建都出现：
   ```
   [memory] reusing sensor BVH after nonvisual model-field expansion: actuator_biasprm, ... geom_friction, ...
   ```
   即**域随机化扩展模型字段后没有重建传感器上下文**。本地同代码没触发问题，但这条路径是服务器与本地最可疑的差异点之一。
4. 服务器 `found=0.000` 时 `force=0.00`，连一个匹配都没有；需要区分"物理无接触"和"传感器没匹配"。

## 待补的三条（服务器，独立进程）

```bash
# 1) 512 对照：若 512 也是 0，说明与规模无关，是环境/依赖
python /tmp/probe_one.py 512 35 1

# 2) 打印物理接触数（区分物理 vs 传感器）
CUDA_LAUNCH_BLOCKING=1 python - <<'PY'
import torch
from mjlab.envs import ManagerBasedRlEnv
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
n = 1024
cfg = lgdog1_race_mgdp_env_cfg(with_perception=False)
cfg.scene.num_envs = n
cfg.scene.terrain.terrain_generator.num_rows = 8
env = ManagerBasedRlEnv(cfg, device='cuda:0')
env.reset()
for name in ('nacon', 'ncon', 'nefc'):
    if hasattr(env.sim.wp_data, name):
        print(name, env.sim.wp_data.__getattribute__(name).numpy()[:4])
for _ in range(5):
    env.step(torch.zeros((n, env.action_space.shape[-1]), device='cuda:0'))
for name in ('nacon', 'ncon', 'nefc'):
    if hasattr(env.sim.wp_data, name):
        print('after', name, env.sim.wp_data.__getattribute__(name).numpy()[:4])
d = env.scene['feet_ground_contact'].data
print('found', d.found.float().mean().item())
env.close()
PY

# 3) 依赖版本对照
pip list | grep -iE '^(mjlab|mujoco |mujoco-warp|warp-lang|rsl-rl-lib|torch) '
git log -1 --oneline; git status --short
```

判读：

| 观察 | 结论 |
|---|---|
| 512 也是 `found=0` | 与规模无关 → 依赖版本或传感器挂载 |
| `nacon=0`（reset 后） | 宽相/配对在服务器上没产对 |
| `nacon>0` 但 `found=0` | 传感器匹配问题（`contact_sensor_maxmatch`/BVH 复用） |

## 2026-11-10 第二轮：物理侧确认

服务器 1024 世界、rows=8：

```
reset  nacon [0]   nefc [12 12 12 12]
step5  nacon [0]   nefc [12 12 12 12]
found 0.0
```

**`nacon=0` 从 reset 起就是 0** → 不是传感器匹配问题，是物理宽相没有产出任何几何体对；
`nefc=12` 说明约束数组正常，只是没有接触。

服务器依赖：`mjlab 1.2.0 / mujoco 3.5.0 / mujoco-warp 3.5.0 / warp-lang 1.12.0 /
rsl-rl-lib 5.0.1 / torch 2.7.1`（本地是 torch 2.14.1，且本地目录有 `.git`，
服务器目录是解压部署、没有 `.git`）。

### 规模阈值与可疑点

| 世界数 | 每世界 geom | 世界×geom |
|---:|---:|---:|
| 512（正常） | 27840 | **14.3 M** |
| 1024（归零） | 27840 | **28.5 M** |

14.3 M < **2^24 (16.8 M)** < 28.5 M。SAP 宽相里 `wp.utils.array_scan` 只支持
`int32`/`float32` 前缀和，排序 key 也是 int32；`nsweep = 5 * nworldgeom`。
当"世界×geom"跨过 2^24 时，前缀和/索引很可能溢出，二分查找失效 → 所有世界都拿不到配对。
这能解释为什么 512 好、1024 坏，且与 `nconmax`、`optimize_vram`、broadphase 选择都无关。

### 决定性对照（待做）

固定世界数、只改 geom 数：1024 世界 + `rows=2`（geom 约 7–8k，世界×geom ≈ 7.5 M）。
若接触恢复 → 确认是"世界×geom"乘积阈值，不是世界数本身。

### 代码一致性核对

服务器不是 git 仓库，用文件哈希对照（本地 = GitHub `3fcdbd7`）：

```
7ff157f2e2cb6c55c9acac1c1c22964f  src/_mjwarp_pairs.py
3b4013ee97628ebed4793703e1384fae  src/sim_memory.py
4557aefdf9f8330e80e76a57ddd8d5e1  src/_mjwarp_bvh.py
```

### 临时可用配置

在阈值以下训练，用步数换世界数（每轮样本量不变）。`lgdog.py train` 没有暴露步数参数，
直接用 `scripts/train.py`：

```bash
python -u scripts/train.py LgDog1-Race-MGDP \
  --env.scene.num-envs=512 --agent.max-iterations=16000 \
  --agent.logger=tensorboard --agent.seed=42 --agent.run-name=V9_seed42 \
  --enable-nan-guard=True --terrain-num-rows=8 --terrain-num-cols=10 \
  --agent.num-steps-per-env=96 --agent.algorithm.num-mini-batches=4
```

512×96 与 1024×48 的每轮样本量相同（≈49k），GPU 不会因为世界数减半而闲置。


| 版本与本地不一致（尤其 `mujoco-warp`/`warp-lang`/`torch`） | 先对齐版本再复测 |

## 临时结论

- 1024 及以下在本地完全正常，服务器上 1024 失效；
- 服务器跑 512 以下可以训练，但浪费 44 GiB 显存；
- 在拿到上面三条结果前，不建议再开 4096+ 长训练。

## 2026-11-10 第三轮：阈值定量

固定 1024 世界，只改地形行数（即每世界 geom 数）：

| rows | ngeom | nacon | found | 世界×geom |
|---:|---:|---:|---|---:|
| 2 | 6992 | 2701 | 1.020 | 7.2 M |
| 4 | 13940 | 2807 | 1.022 | 14.3 M |
| 8 | 27838 | 0 | 0.000 | 28.5 M |

结合前几轮：

| 世界数 | ngeom | 世界×geom | 结果 |
|---:|---:|---:|---|
| 512 | 27840 | 14.25 M | ✅ |
| 1024 | 13940 | 14.27 M | ✅ |
| 1024 | 27840 | 28.5 M | ❌ |
| 1536 | ~9k（本地盲走） | ~13.8 M | ✅ |
| 384 | 27840 | 10.7 M | ✅ |

**规律：`世界数 × 每世界 geom < ~1.6×10^7（≈2^24）` 时正常，超过就整场 0 接触。**

### 由此得到的容量上限

| 目标世界数 | 每世界 geom 上限 | 现在的 27838 需要 |
|---:|---:|---|
| 2048 | 8180 | 缩减到 29% |
| 4096 | 4090 | 缩减到 15% |
| 5120 | 3270 | 缩减到 12% |

4096/5120 想保留完整 8×10 地形，必须改宽相（int64 前缀和），不能靠调参。
