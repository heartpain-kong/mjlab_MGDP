# 深度传感器 BVH 初始化的显存峰值

## 已确认的失败

用户导出的工作台文本记录了6×20、2048环境的实际启动命令。
该次退出码为1，已出现以下日志：

```text
[memory] bounded host pair construction enabled (verified mujoco-warp 3.5.0)
[memory] collision pairs ready: active=437892, elapsed=4.51s; uploading model without full pair indices
Warp CUDA error 2: out of memory
RuntimeError: Failed to allocate 1540177920 bytes on device 'cuda:0'
```

堆栈位于`SensorContext -> create_render_context -> build_scene_bvh`，
失败的是为所有环境复制几何体旋转矩阵的设备数组。它约1.43GiB。
这次有明确CUDA OOM证据，与此前RAM风险、SIGKILL/−9应分别处理。
主机分块建表已成功执行，后续显存分配仍未全部完成，尚未进入PPO更新。
工作台的1.2/8GB读数来自进程退出后的状态，不能用它判断失败时峰值。

## 修复

原生mujoco-warp3.5.0使用单份MuJoCo MjData作为所有环境的初始几何体姿态，
但会先生成nworld份尺寸、位置与旋转矩阵，再计算每个环境的相同初始包围盒。

现在使用原生包围盒内核只计算一次初始bounds，再将bounds写入各环境独立的BVH叶子。
每环境的BVH、group、root和运行时原生refit保持独立；射线和相机的几何体过滤、
命中算法、深度分辨率和87度视野不变，不改变感知权重、奖励或域随机化。

20890个几何体、2048环境时，原生三份暂存设备数组约2.39GiB，
优化后尺寸/位置/旋转暂存仅一世界约1.2MiB，另有一份小型初始包围盒。
这些是按数组大小计算的逻辑分配量，并非RTX4060上测得的整体显存下降。
独立BVH叶子、节点、物理Data、完整pair-id表和PPO缓存仍占显存。

MJLab1.2.0在域随机化字段扩展后会重建SensorContext。
对于仅包含body_mass、body_inertia、body_ipos、geom_friction、
actuator_gainprm、actuator_biasprm及其质量/逆权重/肌腱参考量等物理派生字段的扩展，现复用已有Context，
避免创建第二份BVH。物理字段扩展、物理参数随机化和CUDA图重建照常执行。
相机参数、几何体字段、未知字段及未提供字段集合的调用仍走原生recreate。

采用版本和源码指纹校验：BVH构建函数及其bounds内核须匹配mujoco-warp3.5.0，
Context复用须匹配MJLab1.2.0的recreate源码。
验证不通过的优化独立跳过并打印说明；不修改site-packages。
所有临时工厂/方法替换在环境构造结束或异常时恢复。工作台排版不变。

## Linux 验证

更新代码，在已激活环境和项目根目录中复现本次规模，先完成两轮：

```bash
git pull
python -u lgdog.py train race_mgdp --envs 2048 --iterations 2 \
  --terrain-num-rows 6 --terrain-num-cols 20 --ppo-mini-batches 8
```

应出现：

```text
[memory] shared BVH initialization: geoms=20890, worlds=2048; avoids=2.39 GiB repeated pose/size staging; per-world BVH and runtime refit unchanged
[memory] reusing sensor BVH after nonvisual model-field expansion: ...
```

实际几何体数量随配置可能变化。若打印keeping native BVH staging或keeping native context recreation，
对应优化未启用，应同时检查依赖版本和源码校验，不应把它当成已优化的结果。
原生脚本`--optimize-vram False`、短命令`--no-optimize-vram`会关闭这些后端优化，
感知推理分批参数仍独立控制。

完整初始化、采样、PPO更新与保存都成功后，才能确认该机器当前配置的短程容量；
长期训练、8×20或3072环境仍需独立验证。不要用预估节省量承诺可容纳3072。

## 验证范围

- CPU比较原生与优化的初始包围盒，覆盖混合原生几何体、mesh/hfield、几何体组过滤、
  多环境和相机开关。
- 验证只上传一世界尺寸/位置/旋转暂存，不调用全世界np.tile。
- 对各环境不同自由关节姿态和几何体尺寸，执行原生refit，射线距离、几何体ID和法线与原生一致。
- 物理字段复用和相机/几何体重建分流，以及异常后的工厂恢复均有回归测试。
- 实际10环境、2×10地图的CPU仿真比较12控制步，包括轨迹、奖励、本体观测与深度射线观测。
- CPU上使用8个PPO小批次，更新、保存、ONNX导出、恢复后再次更新通过；日志确认HIMLoco
  物理字段扩展复用了现有BVH，没有重建第二份Context。
- 6项针对性测试通过；最终全量复跑301项0失败、1跳过。
  此前全量及门户单独复跑曾遇到已有HTTP测试的WinError10054连接重置，相关门户代码未修改，
  最终通过不代表该HTTP时序问题已修复。

本地没有CUDA驱动，以上不构成用户GPU上的实测通过。临时日志位于`.artifacts/vram-3072/`。
