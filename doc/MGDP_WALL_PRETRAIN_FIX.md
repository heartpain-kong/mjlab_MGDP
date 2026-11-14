# 比赛尺寸强化课程：预训练入口修复（2026-11-15）

## 日志定位

用户日志运行路径为 `/home/heartpain/mjlab/unitree_rl_mjlab_lgdog/scripts/pretrain_mgdp.py`，其 argparse 只接受 `LgDog1-Rough-MGDP` 和 `LgDog1-Race-MGDP`，工作台却传入 `LgDog1-Race-MGDP-WallTraining`，因此在参数解析阶段以退出码2结束。仿真和感知训练都还没开始，前面的TorchScript弃用警告不是此次退出原因。

这是工作台任务与旧脚本不匹配。日志里的256环境、12行、批256是旧通用默认；统一八档课程要求8行。旧输出 `mgdp_race_v3.pt` 还与原赛道共用，需要隔离。

## 修复结果

- 预训练CLI接受WallTraining任务，实际使用其统一八档赛道。
- 工作台、命令行默认与任务提示统一为64环境、8行×10列、批大小64。
- 工作台收到非8行的强化课程请求时，直接报清晰错误，不创建必然失败的子进程；至少10列保留全部障碍类型。
- 强化课程输出改为独立的 `logs/mgdp/perception/mgdp_wall_v5.pt`，CLI默认、任务就绪检查和新策略配置均读取同一路径。
- 原Race任务仍使用 `mgdp_race_v3.pt`。旧策略完整续训仍从其检查点契约恢复配套感知；不能把新感知与旧策略混用。

配置源为 `src/config/mgdp_pretrain.py`；工作台声明在 `src/dashboard/catalog.py`，启动验证在 `routes.py`，采集在 `scripts/pretrain_mgdp.py`。

## Linux本机覆盖

累计补丁：`E:/算力自由/mgdp_wall_pretrain_fix_20261115.zip`及同名`.sha256`。它包含此前必要的课程/工作台/感知依赖，不含训练模型、日志、虚拟环境。原补丁保留；E盘镜像与完整包更新前自动备份。

将zip和.sha256两个文件复制到 **Linux的 `/home/heartpain/mjlab/`**。这是日志中的工程父目录，不能只更新Windows的E盘副本。

```bash
cd /home/heartpain/mjlab &&
sha256sum -c mgdp_wall_pretrain_fix_20261115.zip.sha256 &&
unzip -o mgdp_wall_pretrain_fix_20261115.zip -d /home/heartpain/mjlab
```

确认覆盖成功后，激活原环境并检查实际脚本支持项：

```bash
cd /home/heartpain/mjlab/unitree_rl_mjlab_lgdog
conda activate unitree_rl_mjlab
python -m scripts.pretrain_mgdp --help
```

`--task`的选项必须包含`LgDog1-Race-MGDP-WallTraining`。本次没有在该Linux主机远程执行覆盖或GPU训练。

**退出并重启工作台后端，再刷新网页。** 只刷新页面不会重新导入Python配置；旧后端仍可能发送256环境/12行。若原来是在终端运行工作台，在该终端Ctrl+C结束网页服务后，用原启动命令重启，例如：

```bash
python lgdog.py --python /home/heartpain/miniconda3/envs/unitree_rl_mjlab/bin/python ui --no-browser --port 8765
```

工作台选择“MGDP · 比赛尺寸强化课程”后点击预训练。也可直接执行：

```bash
python -u -m scripts.pretrain_mgdp \
  --task LgDog1-Race-MGDP-WallTraining \
  --device cuda:0 --num-envs 64 --terrain-rows 8 --terrain-cols 10 \
  --steps 4000 --iterations 3000 --batch-size 64 \
  --out logs/mgdp/perception/mgdp_wall_v5.pt
```

这条命令使用现有默认深度空间约定。D435i实验配置另按`doc/MGDP_D435I.md`显式选择并与配套新策略一起使用。

## 验证边界

新增回归将工作台生成的真实argv交给实际CLI解析，并检查任务提示、默认资源、八行限制、新旧输出隔离及策略配置的一致性。

CPU十环境高墙夹具：通过工作台路由生成命令，采集12步，完成1次感知优化，保存真实`mgdp_wall_v5.pt`；旧Race感知文件字节保持不变。为控制CPU开销，物理夹具只保留高墙比例，因此不声称验证了十类障碍的GPU采集吞吐、完整3000次优化或模型效果。

完整结果见 [MGDP_WALL_PRETRAIN_FIX_VALIDATION.json](MGDP_WALL_PRETRAIN_FIX_VALIDATION.json)。

完整分组回归407项通过、5项跳过、135项子测试通过；工作台/任务/WallTraining/感知入口相关模块48项通过。
