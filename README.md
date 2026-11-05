# LgDog · mjlab 训练工作台

## Ubuntu 一键安装

在 Ubuntu 22.04/24.04 的算力租借实例中，把仓库上传或克隆后执行：

```bash
cd unitree_rl_mjlab_lgdog
bash install_ubuntu.sh
source .venv/bin/activate
python scripts/verify_install.py
```

脚本会安装系统编译库、创建 `.venv`、安装锁定的 mjlab/MuJoCo/Warp 组合、以 editable 模式安装当前源码，并检查 LgDog 任务是否注册成功。默认使用 Python 3.10–3.13 和 PyPI 的 PyTorch GPU 包；只有做 CPU 导入检查时使用 `bash install_ubuntu.sh --cpu`。显卡训练仍需要 NVIDIA 驱动和可用 CUDA 驱动。

已有系统依赖或没有 sudo 权限时：

```bash
bash install_ubuntu.sh --skip-system-deps
```

需要指定 PyTorch CUDA 源时：

```bash
bash install_ubuntu.sh --torch-index https://download.pytorch.org/whl/cu128
```

详细参数和手工安装步骤见 [doc/setup_zh.md](doc/setup_zh.md)；GPUFree 平台的 CUDA、SSH 转发和显存记录见 [doc/DEPLOYMENT_GPUFREE.md](doc/DEPLOYMENT_GPUFREE.md)。

## 运行

在已安装训练依赖的 Python 环境中，进入这个目录：

```bash
python lgdog.py
```

浏览器打开 **http://127.0.0.1:8765**。选择任务后，可以启动训练、play 回放、sim2sim、查看奖励和真实日志。默认任务是 **HIM-PPO + 标准崎岖地形盲走**。

MGDP 深度感知作为独立任务（别名 `mgdp`）使用，首次训练前先跑一次感知预训练（见 [MGDP 说明](doc/MGDP_MJLAB.md)）：

```bash
python scripts/pretrain_mgdp.py --num-envs 256 --steps 4000 --iterations 3000
```

```bash
python lgdog.py tasks                      # 看全部任务的短名称（默认 standard）
python lgdog.py web                        # 总览：环境/GPU/进程/轮次/日志一页看完（端口 8770）
python lgdog.py kline                      # 曲线终端：把 TensorBoard 曲线当 K 线看（端口 8766）
python lgdog.py train standard --envs 1024 --iterations 16000
python lgdog.py play standard              # 自动选对应任务最新 checkpoint
python lgdog.py sim standard --vx 0.4      # 自动选对应任务最新 ONNX
python lgdog.py rewards standard --term foot_clearance

# MGDP 深度感知（需先预训练）
python lgdog.py train mgdp --envs 2048 --iterations 10000
# 定向越障 v2：入口→障碍→出口，按各类通过率门控扩速
python lgdog.py train race_mgdp --envs 2048 --terrain-num-rows 8 --terrain-num-cols 20 --iterations 16000
# 2027 比赛场地：python lgdog.py train race --envs 4096
```

正式训练示例：`python lgdog.py train standard --envs 1024 --iterations 16000 --run-name V8_him_standard_seed42`。环境数应按显存调整。面板本身仅用 Python 标准库；训练仍需要原来的 mjlab / MuJoCo / PyTorch 环境。

MGDP v2 关闭抬腿高度奖励，启用 `feet_air_time`，区分机身与腿部擦碰；使用新的 `_v2` 实验目录。`race_mgdp` 的方向、低速起点和通过率门控由任务管理，不能直接完整续训旧 MGDP_V3。详见 [定向越障配置与验证](doc/MGDP_PASSAGE.md)。

## 2027 比赛场地

`27rc-dog-race` 的障碍赛/任务赛场地已经接入训练和 sim2sim：

```bash
# 真场地：启动区起步，0.4 m/s 往前走
python scripts/sim2sim.py LgDog1-Race-Blind-HIM --onnx=<policy.onnx> --map race --vx 0.4 --duration 5

# 比赛地形课程：梅花桩 0.2×0.2×0.1、单木桥 0.4 宽、台阶 0.12/0.32、砖木坑、绕桩、0.3 m 高墙
python lgdog.py train race --envs 4096 --iterations 16000
```

训练地形是「按规则尺寸拼的近似」：轮胎阵和限高框架不是地面高度场，只在 sim2sim 的真场地里练。
完整对照表、坐标、换规则时怎么重新转换，见 [doc/RACE_COURSE.md](doc/RACE_COURSE.md)。

## 一个网页看全部：mjlab 总览

```bash
python lgdog.py web        # 浏览器打开 http://127.0.0.1:8770
bash run_web.sh            # Linux 等价入口（Python 解释器沿用 run_lgdog.sh 的规则）
run_web.cmd                # Windows 双击
```

**训练状态、显卡、磁盘、环境版本、相关进程、全部训练轮次、日志尾巴**收在一页，并且默认顺手带起
工作台（:8765）和曲线终端（:8766）—— 只记一个地址就够了。轮次表格直接读曲线终端解析好的指标，
点「K线」跳到对应轮次的蜡烛图。页面只读系统状态，唯一的写操作是启停**本页启动的**子服务。
细节和开机自启写法见 [doc/WEBHUB.md](doc/WEBHUB.md)。

## 看曲线：曲线战士（TensorBoard 转 K 线）

```bash
python lgdog.py kline            # 浏览器打开 http://127.0.0.1:8766
```

把 `logs/rsl_rl` 下的标量曲线按训练步切成蜡烛图：分时曲线、短/中/长/全程 K、MA 均线、波动柱、
自选轮次、最近 5 根高低、逐点流水，外加「全部指标 / 轮次对比 / 数据体检 / 指标词典 / 导出 CSV」。
只读本机 event 文件，不联网；第一次读大的 run 要十几秒，之后走 `logs/kline_cache/` 秒开。
细节见 [doc/KLINE.md](doc/KLINE.md)。

## 日常只需要找这几处

| 位置 | 用途 |
|---|---|
| `lgdog.py` | 统一入口：浏览器面板与简短命令 |
| [reward_cfg.py](src/tasks/velocity/config/lgdog1/reward_cfg.py) | 奖励权重、参数；新标准 HIM 修改 `him_standard` |
| `src/tasks/velocity/config/lgdog1/` | 地形、观测、课程、任务注册与 PPO 配置 |
| `logs/rsl_rl/` | 训练 checkpoint、ONNX、TorchScript 和配置快照 |
| `logs/dashboard/` | 面板启动的进程记录与输出日志 |
| `logs/kline_cache/` | 曲线终端的 event 解析缓存（可随时删，删了会重解析） |
| `logs/web/` | 总览拉起的三个服务的输出日志 |
| `src/kline/` | 曲线终端：event 解析、窗口聚合、只读 HTTP 服务与页面 |
| `src/portal/` | mjlab 总览：环境/GPU/进程采集、服务管理、只读 HTTP 服务与页面 |
| `src/assets/terrains/` | 地形资产：LgDog 测试图 + 2027 比赛场地（`race_*.xml`） |
| `scripts/import_race_course.py` | 把比赛场地仓库转换成工程地形 XML（规则更新时重跑） |
| [doc/](doc/INDEX.md) | 使用与技术文档 |
| `archive/` | 旧补丁、旧验证记录和历史清单 |

每个 HIM checkpoint 会额外生成 `policy_jit.pt`（TorchScript 推理模型）；`policy.onnx` 仍保留用于现有 sim2sim 和部署链路。

- [面板操作与目录迁移说明](doc/WORKBENCH.md)
- [mjlab 总览（一个网页看全部）](doc/WEBHUB.md)
- [曲线终端（TensorBoard 转 K 线）](doc/KLINE.md)
- [2027 比赛场地：训练与 sim2sim](doc/RACE_COURSE.md)
- [标准地形 HIM 训练说明](doc/LGDOG_HIM_STANDARD.md)
- [MGDP 深度感知迁移与训练](doc/MGDP_MJLAB.md)
- [奖励调参](doc/REWARDS.md)
- [环境安装](doc/setup_zh.md)
- [GPUFree Ubuntu 部署记录](doc/DEPLOYMENT_GPUFREE.md)

目录已从 `unitree_rl_mjlab_lgdog/unitree_rl_mjlab_lgdog/` 合并为一层，训练日志和模型随工程一起保留。原有 `scripts/train.py`、`play.py`、`sim2sim.py` 仍可直接运行。上游中英文 README 保存在 `doc/UPSTREAM_README*.md`。
