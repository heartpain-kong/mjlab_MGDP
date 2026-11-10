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
| 版本与本地不一致（尤其 `mujoco-warp`/`warp-lang`/`torch`） | 先对齐版本再复测 |

## 临时结论

- 1024 及以下在本地完全正常，服务器上 1024 失效；
- 服务器跑 512 以下可以训练，但浪费 44 GiB 显存；
- 在拿到上面三条结果前，不建议再开 4096+ 长训练。
