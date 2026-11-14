# MGDP 采样加速（2026-11-13）

用户当前 L40、5120 环境、9×10 地形日志：50 步 rollout 采样约 108.5 秒，PPO 更新约 0.229 秒，约 2354 环境步/秒。采样占约 99.8%；本次主要优化碰撞宽相和采样同步。环境数更多不等于训练轮次更快。

## 实现

1. **共享静态碰撞 BVH**：静态赛道只建一次树，每个世界只让运动几何体查询这棵树；无限平面单独处理，运动物体之间只检查允许的碰撞对。替代每个世界、每个物理步重复对整个静态赛道做 SAP 排序和扫描。保留原有稀疏碰撞资格表、球/平面/OBB 过滤、窄相、求解器、物理步长与奖励。
2. **设备端动作统计**：rollout 中的最大值、均值、饱和比例留在设备上，调用者读取标量时才转换；移除每步三次 `.item()`，保留原有裁剪与惩罚。
3. **感知推理批次 256→1024**：5120 环境由 20 个分块减少到 5 个；使用原权重和 GRU 状态，GroupNorm/LayerNorm 不混合样本。临时激活显存会增加；已有显存余量仍须用实际报告确认。
4. **工作台 NaN 诊断默认关闭**：原诊断每个物理步复制状态并同步。界面增加可选开关；异常排查仍可开启。训练 CLI 原来的默认值就是 False。未改变 NaN 诊断功能本身。

没有通过减少 rollout 步数、降低 PPO 更新次数或修改碰撞/奖励参数缩短单轮。此补丁不需要重训感知编码器；已有 checkpoint 的感知契约校验仍生效。

## 适用与回退

自动选择要求至少 256 个参与碰撞的静态几何体、静态数量至少是运动数量的 4 倍、运动几何体不超过 128 个，且紧凑查表不超过 800 万项。锁定并校验 mujoco-warp 3.5.0 原生过滤源码。

显式接触对、距离/法向/fromto 碰撞传感器、mocap、flex、负 margin、不含球过滤、不支持的动态几何布局等走原 SAP。几何数组被替换或按世界展开时也回退。静态树要求赛道几何在运行中保持不变；若自行原位修改静态地形、碰撞半径/位置/尺寸，请使用 SAP。现有 MGDP 的重置、质量/摩擦随机化不修改这些静态树边界。

构建阶段继续预分配旧 SAP 工作区与排序 scratch，以保留安全回退和之前 CUDA graph 生命周期修复。此项优化主要减少计算，不承诺减少该部分显存。

启动时出现 `[speed] shared static collision BVH: ...` 表示启用。强制回退：训练增加 `--collision-mode=sap`；检查器使用 `--collision-mode sap`。`--optimize-vram=False` 同时关闭这些后端适配。

## 服务器测速

先保存当前训练并结束该进程，在相同 Conda 环境中运行；不要与另一个训练竞争 GPU。解压更新后必须重新启动训练/工作台，已有 Python 进程不会自动换代码。

```bash
cd /root/unitree_rl_mjlab_lgdog
conda activate /root/lgdog
source server_env.sh
lgdog_server_env "$PWD"
unset CUDA_LAUNCH_BLOCKING

# 先确认小规模 GPU 图执行及接触正常
python -m scripts.check_mgdp_runtime --device cuda:0 \
  --envs 1024 --rows 9 --cols 10 --steps 20 \
  --perception logs/mgdp/perception/mgdp_race_v3.pt \
  --output .artifacts/mgdp-speed/probe1024.json

# 原规模、相同工作量，新进程依次运行；两轮交替顺序，排除初始化时间
python -m scripts.benchmark_mgdp_speed --device cuda:0 \
  --envs 5120 --rows 9 --cols 10 --steps 50 --warmup 20 \
  --perception logs/mgdp/perception/mgdp_race_v3.pt
```

把 `--perception` 换成实际使用的权重。比较结果位于 `.artifacts/mgdp-speed/comparison.json`，`speedup` 是 SAP 平均采样时间 / BVH 平均采样时间，两者均关闭 NaN 诊断。站立零动作测速包括物理、传感器和冻结感知，不包含策略/PPO 更新，也不代表收敛速度。每次保留初始化日志、显存与接触结果。

若需分段诊断，在检查命令上增加 `--profile-stages`。输出 physics step/forward/sense 的调用数与累计时间；CUDA 在 Warp 流上记录事件，末尾统一同步，无逐调用强制同步。诊断会增加开销；它们是 GPU 流上的时间跨度，不能当作独占 kernel 时间，也不应和未分析的吞吐直接混比。

GPU 检查通过后，可先用与当前运行相同的任务/权重做 3 轮训练：

```bash
python scripts/train.py LgDog1-Race-MGDP \
  --env.scene.num-envs=5120 --agent.max-iterations=3 \
  --terrain-num-rows=9 --terrain-num-cols=10 \
  --agent.logger=tensorboard --agent.seed=42 \
  --enable-nan-guard False --agent.run-name=MGDP_speed_check
```

正式续训在工作台选中原 checkpoint 和“续训”，NaN 诊断保持未勾选；原模型、日志和 TensorBoard 用法保持不变。此版本没有修改服务器上正在运行的任务。

## 验证范围

本地 CPU 验证涵盖三种过滤组合的碰撞候选对、多个世界运动、显式接触/传感器回退、接触与短轨迹、新旧真实赛道的关节状态/奖励/深度射线对照，以及动作统计无标量同步。另有 CUDA graph 首次 forward、step 和 recapture 回归测试，需在有 CUDA 的机器执行。

本地 CPU 结果不能推导 L40 的提速倍数；GPU 吞吐、显存和长程收敛均须在服务器复测。

本次全套本地回归：340 passed、3 skipped、127 subtests passed；跳过项包括实际 CUDA 执行。短轨迹对照只验证数值一致性，不代表长期任务成功率。

CPU 顺序测速（10 环境、2×10 地形、测试感知权重、预热 10 步、采样 20 步、两次交替顺序）：SAP 平均 3.086 秒，BVH 平均 1.890 秒，吞吐比约 1.63 倍。仅比较宽相后端；两者均使用新的其他设置。原始报告为本地 `.artifacts/mgdp-speed-serial/comparison.json`。该小规模 CPU 结果不预测 5120 环境的 GPU 收益。

CPU 冒烟流程还完成一次 PPO 更新、checkpoint/ONNX 感知契约检查、编码器 sidecar 导出以及加载 checkpoint 后再次更新。
