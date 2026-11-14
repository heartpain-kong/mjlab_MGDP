# 首次 reset 的 CUDA 700：共享工作区的图生命周期修复

## 服务器记录

用户于 2026-11-12 运行 1024 环境、8×20 地图的 `scripts.check_mgdp_runtime`。
环境构造完成后报告设备总使用量 2.49/44.39 GiB；首次 `env.reset()` 失败，
异步错误最终在 `prepare_rays -> _extract_yaw_rotation -> singular.any()` 被检测到。
随后大量 `wp_free_device_async` 700 是同一 CUDA 错误后的清理错误。
记录未出现 CUDA OOM，不能以这个错误推断显存不足，也不能把失败前的2.49GiB当成完整运行峰值。

## 已修正的代码缺陷

MGDP 的可复用 SAP 数组原来在 `sap_broadphase_i64` 首次执行时创建。
MJLab 1.2.0 的 `Simulation.create_graph()` 先捕获 step，再捕获 forward。
如果首次数组分配发生在 step 捕获中，分配节点属于 step 图。
而 `ManagerBasedRlEnv.reset()` 的执行顺序是 forward → sense，尚未运行过 step 图。
forward 图于是可能先访问 step 图拥有的地址。CPU 没有该捕获/回放生命周期，原有 CPU 测试无法验证它。

现在由环境构造范围内的 `put_data/make_data` 适配，在第一次 `create_graph()` 之前准备所有 SAP 数组。
CUDA segmented sort 的 stream-local CUB 暂存也通过一次纯 scratch 排序在捕获前准备。
不推进物理状态，不改变接触计数、地形、奖励、容量和域随机化。
若在图捕获期间才要求创建这类可复用缓存，将明确报错，不再创建跨图共享的图内分配。
构造工厂在退出或异常时恢复；每个 Data 仍持有自己的缓存。

这是从当前代码确认的缺陷，与服务器报错阶段吻合；服务器这次700的首发根因仍需修复版GPU复测确认。

## 新探针

- `--debug-sync` 在 reset 的 forward/sense 调用边界分别同步，避免把 forward 的异步错误归到射线准备。
- 每次生成独立 trace 子目录，避免上次的 `init_error.json` 被误认为本次结果。
- 结果增加 `phase`、共享BVH/稀疏表和 SAP 预分配状态，首次reset失败也会保留。
- 在关闭环境、释放Warp资源前先写结果，保留原始失败摘要。

将补丁 ZIP 上传到 `/root` 后：

```bash
cd /root
unzip -o mgdp_cuda_graph_fix_20261112.zip -d /root
cd /root/unitree_rl_mjlab_lgdog
conda activate /root/lgdog
source server_env.sh
lgdog_server_env "$PWD"

CUDA_LAUNCH_BLOCKING=1 python -u -m scripts.check_mgdp_runtime \
  --device cuda:0 --envs 1024 --rows 8 --cols 20 --debug-sync \
  --output .artifacts/mgdp-runtime/1024_graph_fix.json
```

应看到 `SAP workspace prepared before CUDA graph capture`，结果包含
`sap_workspace_prepared=true`、SAP_SEGMENTED时的 `sap_sort_prepared=true`，并完成接触检查。
若仍失败，以新报告的 `trace_directory` 下首个 `init_error.json` 为准。
CUDA 700 后必须使用新的Python进程复测；以上命令每次都会启动新进程。

## 验证范围

新增测试检查真实 Simulation 在首次建图前已有共享缓存、首次执行forward即有接触、重建图不更换缓存、
捕获期间拒绝首次分配，以及诊断将forward异常正确归到runtime-forward。
另有真实CUDA专项覆盖“先forward、后step、重建图再forward”；本机没有CUDA驱动，该专项会跳过。
本地验证不能代替服务器上的CUDA图回放通过。
