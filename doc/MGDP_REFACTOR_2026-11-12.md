# MGDP 静态地形感知与碰撞后端重构

本次针对训练记录中的初始化 RAM/CUDA OOM、大规模接触归零、共享 BVH 未生效和采样开销。
参考本机 `E:\足式机器人\MGDP - Mastering a Generalized Depth Perception Model for Quadruped Locomotion\代码\arclab-hku__MGDP`：
`warp_sensor/warp_sensor/warp_manager.py::__load_mesh_from_gym` 只构建一份地形 `wp.Mesh`，所有环境的相机共享它。
本工程采用这个资源组织方式，保留 MuJoCo 的 box/mesh/hfield 精确射线求交，未将赛道物理形状转换成凸包。

## 改动与边界

| 问题 | 现在的行为 |
| --- | --- |
| 地形扫描包含机器人，无法共享静态 BVH | 新 MGDP 配置的 `terrain_scan`、`depth_rays` 都设为 `(0,)`，只扫描地形 |
| 先分配完整叶子再缩小，仍会初始化 OOM | 在经源码验证的 context 工厂中直接分配一世界的 lower/upper/group |
| 共享后射线仍减去 world 偏移，第二个世界起可能越界 | 共享射线内核直接用局部叶子编号，仍用对应世界的射线起点和几何体姿态 |
| 每步 refit 所有世界的同一地形 | 静态共享 context 跳过 refit；机器人、相机、mocap/flex 保留原生逐世界树 |
| 稀疏表仍先在主机创建平方大小的稠密表 | 稀疏模式直接分块生成有效对，不再分配该稠密表 |
| 大部分地形对因焊接到同一 world 被过滤 | 直接枚举与运动焊接体相关的候选对，并补回全部显式接触和距离传感器对 |
| 全局稀疏表被后创建的模型覆盖 | 配对表归各自 Model 所有，设备副本也按模型分开 |
| int64 补丁退出构造上下文后可能失效 | 运行时按 Model 标记分流，关闭优化构建的模型仍走原生路径 |
| 每步重新分配 SAP 临时数组 | 缓存归各自 Data 所有，形状/设备/宽相类型变化时重建 |
| 每世界串行扫描整张地形 | 按 256 个 geom 分块做 int64 前缀和，再累加块偏移和世界偏移 |
| 扫掠游标仍为 int32 | 游标及累计工作量都用 int64 |
| 每帧 `if reset.any()` 引起 CUDA 到 CPU 同步 | 用张量 mask 复位历史帧和 GRU，保留逐环境复位语义 |
| 新约定影响旧策略回放/续训 | checkpoint 与 ONNX 保存过滤设置；旧 checkpoint 恢复旧默认过滤 |

静态共享要求所有被启用几何体都焊接到 world、group=0、无 mocap/flex、无活动渲染相机。
相机、动态几何体、视觉字段扩展均保守回退逐世界 BVH。HIMLoco 物理域随机化继续执行。
工厂替换在构造完成或异常后恢复；运行时只按模型/context 标记分流，不保留已销毁环境的 ID。
不修改 site-packages，只启用通过源码指纹检查的 mujoco-warp 3.5.0 / MJLab 1.2.0 路径。

本次保留赛道几何、10 类障碍、地形课程、1.0 m/s 课程目标、奖励和动作接口。
这次是资源组织与后端正确性重构，不等同于复现官方全部控制算法。
原论文的地形自适应奖励、残差动作等与本工程仍有差别；越墙、轮胎通过率需要独立训练评估。

## 内存量级

叶子数组原生开销为 `nworld × enabled_geoms × 28` 字节，共享后为 `enabled_geoms × 28`。
例如 4096 世界、约 27800 个启用地形几何体，仅叶子数组即可少分配约 **2.97 GiB**；
BVH 节点和构建暂存另计，不能把此数字称为整进程实测节省。

27838 个 geom 的稠密配对查找表约 **2.89 GiB**。稀疏模式在主机和设备都避免它，
仅保留有效配对的 int64 key、int32 二元值。几何体姿态、物理状态、求解器、排序暂存和 PPO 缓存仍与环境数有关。
不能由这两个数字证明某张 GPU 一定能训练 4096 环境。

服务器记录中的约 `2^24 world-geoms` 接触归零是规模相关证据，**不是已证明的 int32 上限**。
int32 上限是 `2^31-1`，原始问题与累计重叠工作量之间的关系仍需 GPU 对照。
本次消除了累计和及扫掠游标的溢出风险，并提供接触检查，不能以 CPU 小场景通过宣称服务器问题已消失。

## 新训练与旧策略

新训练推荐先重新预训练感知模型：高度监督和深度图现在都只包含地形。
已有权重不会被自动删除；先输出到新路径，保留旧策略配套编码器。

```bash
python -m scripts.pretrain_mgdp --task LgDog1-Race-MGDP \
  --num-envs 64 --terrain-rows 4 --terrain-cols 20 \
  --out logs/mgdp/perception/mgdp_race_terrain_v1.pt
```

感知权重记录 `sensing` 约定，加载已记录且不匹配的权重会报错。
旧编码器未记录约定时，新训练会提示重新预训练。
策略 checkpoint 的 `mgdp_perception.sensing` 和 ONNX 的 `mgdp_sensing` 保存实际过滤设置。
旧默认过滤恢复为 `terrain_scan=(0,1,2)`、`depth_rays=None`，包含机器人，仍用逐世界树。
旧策略由此保留原来的观测分布，也不能直接获得静态共享树的全部收益。

## 服务器检查

在项目根目录、已激活训练环境中运行。默认不加载感知网络，先确认物理与射线：

```bash
python -m scripts.check_mgdp_runtime --device cuda:0 \
  --envs 1024 --rows 8 --cols 20 \
  --output .artifacts/mgdp-runtime/1024.json
```

同样网格再检查目标环境数：

```bash
python -m scripts.check_mgdp_runtime --device cuda:0 \
  --envs 4096 --rows 8 --cols 20 \
  --perception logs/mgdp/perception/mgdp_race_terrain_v1.pt \
  --output .artifacts/mgdp-runtime/4096-perception.json
```

报告应显示 `passed=true`、`shared_bvh=true`、`sparse_pairs=true`、非零接触以及有限的物理/射线值。
`env_steps_per_second` 是整段仿真和观测吞吐，有 `--perception` 时也包括编码器推理，**不含 PPO 更新**。
CPU 和 CUDA 检查应分开解释。`--native` 对照同一传感器约定下的原生后端；
`--legacy-sensing` 改回旧过滤，但这个对照同时改变感知内容，不能单独据此归因加速。
初始化异常保留 `runtime_init` 诊断和 JSON 错误摘要。零动作探针只检查入口接触，不测试越障技能。

通过目标规模检查后，完成两轮真实 PPO 更新和保存：

```bash
python -u scripts/train.py LgDog1-Race-MGDP \
  --env.scene.num-envs=4096 --terrain-num-rows=8 --terrain-num-cols=20 \
  --agent.max-iterations=2 --agent.algorithm.num-mini-batches=8 \
  --agent.logger=tensorboard --agent.seed=42 --agent.run-name=mgdp_terrain_check \
  --perception-checkpoint=logs/mgdp/perception/mgdp_race_terrain_v1.pt
```

## 本地验证（2026-11-12，Windows CPU）

- 多环境对照原生射线距离、命中几何体与法线，覆盖 box、mesh、hfield、静态焊接体及各世界不同姿态。
- 确认共享 context 不分配完整逐世界叶子；相机/动态/mocap 场景回退。
- 分块 int64 扫描覆盖块边界、不整除尾部、零值和大于 int32 上限的累计量。
- 两个不同过滤规则的模型同时存在时，稀疏 SAP 接触与原生一致，互不污染。
- 显式静态接触和距离/法线传感器在候选剪枝后保持原生顺序与值。
- 真实 10 环境、2×10 地图 CPU 站立检查：共享 BVH、稀疏配对启用，脚底接触均值 1.0，物理和扫描均有限。
- 真实少量采集、2 步感知预训练、PPO 更新、检查点与 ONNX/编码器导出、恢复后再次 PPO 更新完成。

验证文件位于忽略目录 `.artifacts/mgdp-refactor/` 及 `.artifacts/mgdp-refactor-*.log`。
GPU 图捕获、1024/4096 环境容量、GPU 吞吐和长期越障效果尚未在本机验证。
