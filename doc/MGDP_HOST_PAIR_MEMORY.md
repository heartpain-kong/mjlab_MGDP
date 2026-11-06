# 8×20 / 2048 初始化阶段的配对建表内存修复

## 本次日志说明了什么

用户运行 `LgDog1-Race-MGDP`，2048环境、8×20地图、PPO小批次数8，退出码−9。
日志最后停在地形生成完成、场景组装27838个几何体、即将编译模型和分配物理缓冲。
尚未进入PPO更新，因此修改PPO小批次数无法降低这个阶段的主机建表峰值。

−9表示SIGKILL；仅凭这条记录不能证明由OOM killer触发，需该次内核日志确认。
此前12×20失败已有内核global_oom证据。本次原生建表估算也显示较大的RAM风险：

- 几何体对数量：`27838×27837/2 = 387463203`。
- 原生两条int64索引与堆叠数组：约11.55GiB主机内存。
- 该值不含身体ID、焊接ID、掩码等多个全量中间数组，不能当成原生总峰值。
- 每环境几何体姿态总量：2048环境约2.55GiB设备存储，另有完整查找表等分配。

此前优化在原生模型上传结束后才回收未使用的几何体对表；进程可能在到达回收点前被杀。
这次改为从构建源头避免上述分配，保留8行、20列、2048环境及地图障碍密度。

## 修复方式与边界

在已验证的mujoco-warp3.5.0中，仅替换`io.put_model`的配对构建代码块，
使用每批262144对的有界临时数组，直接构建最终int32查找表和过滤后的有效对。

完整配对索引、完整几何体对表和全量过滤临时数组不再生成；
完整pair-id查找表仍保留，27838个几何体时约2.89GiB，仍为平方开销。
上传时的数组拷贝、CUDA运行时、物理Data和PPO存储不包含在这2.89GiB内。

接触掩码、焊接/父子刚体过滤、显式接触对覆盖、排除对、距离/法线/连线传感器地址顺序、
碰撞类型计数和NXN/SAP自动选择规则与原生一致。不缩减接触/约束容量，
不调整奖励、速度、域随机化、感知权重或工作台排版。

通过版本检查、精确代码块指纹、运行源码使用检查后才启用。
指纹规范化Python3.12新增的空type_params字段，兼容Linux的Python3.11。
不匹配时打印说明并保留原生路径；不能把未启用时的运行当成已应用此优化。
适配仅在环境构造期间临时替换工厂，退出或异常时恢复，不修改site-packages。

## Linux 更新后验证

```bash
git pull
# 已激活训练环境，位于项目根目录；先验证完整采样和更新链路
python -u lgdog.py train race_mgdp --envs 2048 --iterations 2 \
  --terrain-num-rows 8 --terrain-num-cols 20 --ppo-mini-batches 8
```

应看到以下阶段日志：

```text
[memory] bounded host pair construction enabled (verified mujoco-warp 3.5.0)
[init] MuJoCo model compiled; building MJWarp model with bounded pair construction...
[memory] streaming collision pairs: geoms=27838, pairs=387463203, chunk=262144; retained dense lookup=2.89 GiB; skipping unused full geom-pair table
[memory] collision pairs ready: active=..., elapsed=...s; uploading model without full pair indices
```

随后仍须完成physics-ready、runner-ready、采样、PPO更新与保存，才能确认这台机器当前配置可运行。
两轮通过仅用于容量和链路检查，不能证明稳定长训、收敛或越障效果。
若再次−9，按实际退出时间查看`journalctl -k`里的OOM及被杀PID记录；
若出现CUDA OOM，继续针对设备分配排查，不把两种故障混在一起。

原生脚本可用`--optimize-vram False`恢复原生配对构建，短命令对应`--no-optimize-vram`；
感知分批推理的开关独立，见[显存说明](MGDP_VRAM_3072.md)。

## 验证记录与限制

本地为Windows CPU环境，未连接用户的Linux RTX4060 Laptop。

- 实际8×20、2048环境的场景生成、MuJoCo编译和主机配对构建：27838个几何体，
  387463203个完整查找条目，583800个有效对；主机查找表3099705624字节，
  空闲完整几何体对表0字节；该测试进程采样RSS峰值约3.81GiB，配对过滤约22.44秒。
  **此测试未分配Warp模型、完整物理Data、感知或PPO，也不是GPU训练进程的总峰值。**
- 独立小模型比较了全部模型字段，除有意跳过的闲置完整几何体对表，其他字段与原生逐项一致。
- 覆盖显式接触覆盖、排除对、父子过滤启闭、重复距离传感器地址、空场景、单几何体、
  小批次边界和NXN/SAP_TILE/SAP_SEGMENTED三种选择。
- 10环境2×10实际CPU仿真，原生与分块版本分别使用NXN和SAP_SEGMENTED执行12控制步，
  轨迹、奖励和本体观测在rtol=1e-5、atol=1e-6内一致。
- 全套295项测试0失败、1跳过；最终传感器计数调整后，5项原生配对对比测试再次通过。
- 10环境、2×10地图、8个PPO小批次的真实CPU更新、检查点/ONNX导出、恢复后再次更新通过。

临时日志位于忽略目录`.artifacts/vram-3072/`。
