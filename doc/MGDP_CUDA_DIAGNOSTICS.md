# 8×20地图：CUDA OOM、700错误与初始化阶段记录

用户最新记录包含两段信息：

- 2048环境、8×20：明确CUDA OOM，`create_render_context`分配
  `lower=wp.zeros(nworld*bvh_ngeom, dtype=wp.vec3)`时申请683630592字节失败，约652MiB。
  这发生在调用`build_scene_bvh`之前，初始共享暂存优化尚未运行。
  因此没有shared BVH initialization消息，不能据此断定用户未更新代码。
- 1024环境、8×20：只有`wp_free_device_async`阶段大量CUDA700（illegal memory access），
  没有首个Traceback或实际失败阶段。不能凭退出清理日志认定仍是OOM，也不能确定哪个内核触发。

之前的分块建表和共享初始暂存避免了临时数组，并没有消除每环境BVH和物理状态的常驻开销。

## 当前结构的显存预算

按实际27838个几何体计算，单位GiB。BVH叶子按全部几何体纳入估算，
实际enabled_geom_groups可能排除少量几何体；用户失败的lower数组对应27817个启用几何体。

| 环境数 | 完整配对查找表 | 几何体姿态 | BVH叶子lower/upper/group | 三项小计 |
|---:|---:|---:|---:|---:|
| 512 | 2.89 | 0.64 | 0.37 | 3.90 |
| 1024 | 2.89 | 1.27 | 0.74 | 4.90 |
| 2048 | 2.89 | 2.55 | 1.49 | 6.92 |
| 3072 | 2.89 | 3.82 | 2.23 | 8.94 |

**三项小计不是总显存，也不是峰值。** 尚未计入BVH树节点及构建暂存、其他物理Data、
域随机化字段、Torch策略/感知/PPO缓存、驱动与其他进程。
配对查找表由地图规模决定，降低环境数不会使它同比缩小。
在当前8×20结构下，3072这三项就已超过8GiB，不能继续把8GB GPU的3072当作可行承诺。
2048的实际OOM也表明不能仅靠上一版临时数组优化确认容量。

## 本次诊断改动

CUDA训练初始化期间临时包围以下调用，在调用前后同步设备、输出全设备显存及Torch计数：

- model-upload：MJWarp模型上传（保留现有分块建表工厂）。
- physics-data：物理Data分配。
- simulation-graphs：Simulation.create_graph返回后同步，不在capture内部同步。
- sensor-context：深度/相机Context创建。
- environment-construction：覆盖其余环境构造步骤。

遇到Python异常或同步检查发现CUDA异常时，立即在该run目录记录：

- `init_error.json`：最先捕获的具体阶段、异常、完整Python traceback，后续外层异常不覆盖它。
- `init_stage.json`：当前阶段、最后完成的阶段、已捕获失败阶段。

若原生程序直接退出或被杀而没有Python异常，可能只有最后阶段记录；不能保证始终生成错误文件。
记录目录不可写时只打印诊断写入失败，不覆盖原始异常。
多GPU时按全局rank使用独立的`cuda_init_rankN`子目录。
CPU训练不启用这些同步钩子。训练更新阶段不增加逐步同步，不改变碰撞、传感器或CUDA图算法。
临时函数/方法替换在构造成功或失败时恢复。工作台排版和显式训练参数不变。

## 下一次先定位错误

在已激活环境、项目根目录运行。每次都是新训练进程，保留8×20地图，先使用512环境两轮：

```bash
git pull
CUDA_LAUNCH_BLOCKING=1 python -u lgdog.py train race_mgdp \
  --envs 512 --iterations 2 --terrain-num-rows 8 --terrain-num-cols 20 \
  --ppo-mini-batches 8
```

CUDA_LAUNCH_BLOCKING仅用于这次定位异步错误，会影响速度；容量与吞吐应在后续关闭该变量后另测。
该配置仍需用户GPU实测，并非已证实512一定能跑通。
若仍失败，优先查看本次run的`init_error.json`及第一条`[init:error]`，
不要用大量释放阶段700报错代替首因。若初始化成功而训练阶段出错，应查看训练阶段首次Traceback。
不要因为只看见700就要求重启整机或认定GPU硬件损坏。

## 验证

5项新增回归检查覆盖嵌套阶段错误保留、异步同步错误传播、工厂恢复、CPU不产生文件、
诊断钩子不绕过现有紧凑模型工厂，以及固定地图的三项显存预算。
全量306项中304项通过、1跳过、1项原有门户HTTP测试出现Windows连接中止（WinError10053）；
失败为test_portal.PortalHTTPTests.test_start_requires_token_and_same_origin，相关门户代码未改动。
没有在用户RTX4060 Laptop上执行本次GPU诊断运行。
