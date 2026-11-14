# MGDP 第二阶段加速：物理线程裁剪与冻结感知推理

2026-11-13 验证入口修复：`from_checkpoint(map_location=...)` 仅控制 checkpoint 张量的载入位置，构造的模型仍在 CPU。`scripts/check_mgdp_inference.py` 现显式加载到 CPU 后执行 `.to(args.device)`，再进入连续帧验证。已加入不依赖真实 GPU 的异设备迁移入口回归测试。此前服务器的 `Input type torch.cuda.FloatTensor / weight type torch.FloatTensor` 源于该入口遗漏。

## 当前实测和目标

第一版服务器 L40 已验证：5120 环境、9×10 地形、50 步，零动作采样 107.02→3.68 秒；正式 PPO 稳定后采样 4.23～4.27 秒，学习约 0.23 秒，单轮约 4.48 秒，结束时全局显存 23.45/44.39 GiB。

后续 GPU 分段诊断：50 步采样共 3.698 秒，200 次物理 step 共 1.975 秒、50 次 forward 共 0.483 秒、50 次 sense 共 0.190 秒。物理部分约占总计的三分之二；第二版优先降低物理线程开销，并优化剩余感知推理。计时是 GPU 流上的跨度，可能包含等待，不能当成独占 kernel 时间。

“2 万轮约 25 小时”是对指定更新轮数的线性预算，不是成功通过全部障碍必须训练满 2 万轮的证据。固定轮数、每秒吞吐、最终通过率要分别看；已有可用策略优先在原 checkpoint 上续训。此补丁没有减少 rollout 或 PPO 更新量，也没有验证较少轮数能达到相同通过率。

## 算法与实现

### 1. 运动几何体线程裁剪（默认启用）

原生 MJWarp 已在 make_data 初始化静态几何体变换；每步 `_geom_local_to_global` 对静态几何体立即 return。但仍以 `nworld × ngeom` 启动所有线程。5120×15677=80,266,240 个线程，LgDog 实际仅 38 个运动几何体/世界。

新实现使用原生相同的 weld/mocap 判定，预先生成运动几何体索引，只启动约 194,560 个线程。包括机器人视觉几何体和 mocap 后代，不仅限于参与碰撞的几何体。保留全部原生 body/site 内核、变换表达式、求解器和物理步长；减少的是被原生代码立即跳过的工作，绝不把线程减少比例当作整轮加速比例。

仅在静态占多数、至少256几何体且 mujoco-warp 3.5.0 的 kinematics/geom 内核源码匹配时启用。拓扑数组替换时回退。质量、摩擦、COM 随机化以及按世界展开的 geom_pos/quat 仍沿用原生索引逻辑。运行中自行修改 body_weldid/body_rootid/body_mocapid/geom_bodyid 的值时，应关闭本优化。

回退：训练 `--active-kinematics False`；检查器 `--native-kinematics`。`--optimize-vram=False` 会关闭整个适配层。

### 2. CNN 分块、GRU 整批（默认启用）

此前 5120 环境、1024 分块会运行 5 次 CNN 和 5 次 GRU。现在仍分块 CNN 控制激活显存，将每块 token 拼接后只运行一次 GRU。各环境在 batch 维独立，逐环境时间状态和重置语义不变。原权重、特征维度和传感器过滤保持不变。

### 3. 冻结 CNN CUDA Graph（显式选择）

训练增加 `--perception-execution=cuda_graph`。按完整块和尾块形状各保留一张图；在 side stream 上先预热，再捕获固定缓冲区。真实图像复制到输入后 replay，输出 clone 后再处理下一块，避免下一次 replay 覆盖前一块 token。GRU 留在图外，只运行一次，无需把不同环境的状态复制进图。

使用原生 FP32，不改变感知更新频率或引入帧缓存。CPU 自动使用 eager，GPU 捕获失败会报错，测速不会把未启动图的结果当成成功。实现遵循 [PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs) 的预热、固定地址和输入复制要求。

### 4. 可选 BF16 CNN

训练增加 `--perception-precision=bf16`。仅 CNN 使用 autocast，token 转回 FP32 后运行 GRU，隐藏状态、策略、PPO、物理和磁盘权重仍为 FP32。

这是近似推理，会改变数值，不默认开启。新模式先用相同权重对照旧推理，检查连续帧、部分重置和尾块：FP32 使用 atol=2e-5/rtol=2e-4；BF16 使用 atol=.01/rtol=.05。后者只排查实现错误，不证明长期策略质量或通过率一致。

精度设置写入 checkpoint 和 ONNX 元数据，Play 恢复精度，Sim2Sim 在 CPU 上使用相同精度设置的编码器。旧 checkpoint 默认 FP32；恢复旧策略时不允许静默切换成 BF16。BF16 应从新实验开始，对比长期学习曲线；FP32 图模式可继续原 checkpoint。

## 上传和一键对照

先保存并结束正在运行的训练。上传 `mgdp_speedup_v2_20261113.zip` 和校验文件到 `/root` 后解压覆盖。补丁累计包含第一版加速和之前 CUDA graph 工作区修复，不包含 logs 或 checkpoint。

```bash
cd /root
sha256sum -c mgdp_speedup_v2_20261113.zip.sha256 && unzip -o mgdp_speedup_v2_20261113.zip -d /root
cd /root/unitree_rl_mjlab_lgdog
conda activate /root/lgdog
source server_env.sh
lgdog_server_env "$PWD"
unset CUDA_LAUNCH_BLOCKING

python -m scripts.benchmark_mgdp_inference --device cuda:0 \
  --envs 5120 --rows 9 --cols 10 --repeats 1 --include-bf16 \
  --perception logs/mgdp/perception/mgdp_race_v3.pt
```

在独立进程中先做推理数值检查，再依次测：

| 名称 | 几何体线程 | CNN 执行 | CNN 精度 |
|---|---|---|---|
| native_eager_fp32 | 原生全表线程 | eager | FP32 |
| eager_fp32 | 仅运动几何体 | eager | FP32 |
| graph_fp32 | 仅运动几何体 | CUDA Graph | FP32 |
| graph_bf16 | 仅运动几何体 | CUDA Graph | BF16 |

所有模式都使用已验证的共享碰撞 BVH 和整批 GRU，50 步、20 步预热，GPU 初始化不计入采样时间。`comparison.json` 位于 `.artifacts/mgdp-inference/`，同时保留每次验证/运行 JSON 与完整日志。首次比较用 `--repeats 1`；正式对比建议去掉此参数，用两轮相反顺序降低缓存/时钟偏差。它是站立采样测速，不替代实际 PPO 与长期评估。

需要新分段诊断时：

```bash
python -m scripts.check_mgdp_runtime --device cuda:0 --envs 5120 \
  --rows 9 --cols 10 --steps 50 --profile-stages \
  --perception-execution cuda_graph --perception-precision fp32 \
  --perception logs/mgdp/perception/mgdp_race_v3.pt \
  --output .artifacts/mgdp-inference/profile.json
```

新增 observation/reward/termination/command/perception 计时；perception 包含于 observation，不能重复相加。Torch 算子记录当前 Torch 流，物理/sense 记录 Warp 流。未开启诊断的吞吐才用于最终比较。

## 短训与回退

GPU 数值检查和测速通过后，使用最快且误差/显存可接受的模式做3轮实际PPO。下面先验证 FP32 图模式：

```bash
python scripts/train.py LgDog1-Race-MGDP \
  --env.scene.num-envs=5120 --terrain-num-rows=9 --terrain-num-cols=10 \
  --perception-execution=cuda_graph --perception-precision=fp32 \
  --enable-nan-guard False --agent.max-iterations=3 \
  --agent.logger=tensorboard --agent.seed=42 --agent.run-name=MGDP_speed_v2
```

若图模式不快，使用 `--perception-execution=eager`；若物理线程裁剪异常，增加 `--active-kinematics False`。推理块大小可用 `--perception-batch-size=2048` 对照，代价是更多临时/图私有显存；不自动增大到4096。

本地 CPU 全套回归和 PPO→导出→续训检查见交付报告。新 CUDA 图/线程内核的 GPU 吞吐、显存和长期策略质量尚待服务器测试，不能从 CPU 结果给出第二版提速保证。

本次本地全套回归：348 passed、5 skipped、127 subtests passed。跳过项包含真实 CUDA capture/replay。新增检查包含 mocap/视觉几何体、按世界展开的几何参数、真实赛道短程关节/奖励/深度对照、连续帧与部分环境重置、BF16精度契约、四种模式的CPU端到端测速流程。CPU BF16 模式另完成实际 PPO 更新、checkpoint/ONNX 精度元数据导出和再次续训；图模式在 CPU 上使用 eager 回退。
