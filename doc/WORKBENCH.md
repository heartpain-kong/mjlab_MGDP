# 本地训练工作台

> 只想「看一眼现在什么情况」（训练在不在跑、GPU、最新轮次、日志尾巴）而不是来起任务的话，
> 用 `python lgdog.py web` 打开 [mjlab 总览](WEBHUB.md)（:8770）：它会顺手把本页和曲线终端带起来。

现在主工程只有一层目录：`E:\mjlab\unitree_rl_mjlab_lgdog`。打开面板后，可用同一套界面选择方案、训练、play、sim2sim 和查看奖励。

## 代码结构（2026-11 重构）

| 模块 | 职责 |
|---|---|
| `src/dashboard/tasks.py` | `TaskSpec`：任务能力声明（观测维度、奖励 profile、是否需要预训练、文档、是否支持自动课程） |
| `src/dashboard/catalog.py` | 任务/算法/场景/地图目录与默认任务解析 |
| `src/dashboard/routes.py` | API 路由表；新增接口 = 一个处理函数 + 一条表项，不再堆 `if path == ...` |
| `src/dashboard/server.py` | 仅 HTTP 管道：回环校验、token/Origin、静态资源白名单、JSON 编码 |
| `src/dashboard/web/js/format.js` | 纯格式化与标签（无 DOM 依赖，可在 Node 里单测） |
| `src/dashboard/web/app.js` | DOM 渲染与交互，通过 `window.LgDogFormat` 复用上面的纯函数 |
| `tests/test_tasks.py` | 任务能力、路由表、前后端契约（JS 用到的 id/API 路径是否都存在） |

**扩展任务**：在 `catalog._TASK_LIST` 里新增 `TaskSpec` 即可，面板下拉、命令预览、模型发现会自动出现；
需要预训练的任务把 `requires_pretrain` / `pretrain_command` / `pretrain_checkpoint` 写全，
命令构建会先检查检查点并给出可执行的提示。

**新增接口**：在 `routes.py` 写 `def foo(ctx, query)`，登记到 `GET_ROUTES` 或 `POST_ROUTES`；
响应形状由函数返回 `(status, payload)`，`server.py` 不再感知具体业务。

**页面布局（2026-11）**：顶部是「任务上下文」卡片——任务下拉列出**全部任务**（按算法分组），算法 / 场景只做展示并随任务自动变化，不会再因为筛选条件把 `mgdp` 等任务藏起来；卡片里同时显示能力标签（`需预训练`、`无自动课程`、`文档`）。
下面是模式页签（训练 / 回放 / Sim2Sim / 查看奖励）。主区默认「并排」：左侧运行设置（训练参数默认折叠，展开后才显示环境数/课程等长表单），右侧监控卡片；
顶部「并排 / 仅设置 / 仅监控」可切换，选择记在 localStorage（键 `lgdog-workbench-view-v2`）。
侧栏新增「曲线终端」（:8766）、「运行总览」（:8770）快捷入口和「MGDP 深度感知」文档。

## 启动

推荐在 Linux GPU 训练机上先激活已安装 mjlab 依赖的环境，再进入工程：

```bash
python -m pip install -e . --no-deps
python lgdog.py
```

默认打开 `http://127.0.0.1:8765`。面板只绑定本机地址。需要指定训练 Python 或不自动打开浏览器时：

```bash
python lgdog.py --python /path/to/training-env/bin/python ui --no-browser --port 8765
```

Windows PowerShell 也可以直接启动：

```powershell
Set-Location E:\mjlab\unitree_rl_mjlab_lgdog
python lgdog.py
# 当前 python 未装训练依赖时，显式指定你已有的训练环境：
python lgdog.py --python C:\path\to\training-env\Scripts\python.exe ui
```

示例中的训练解释器路径应替换为实际环境。启动面板不会自动开始训练；先核对页面显示的 Python 路径，再点击运行。

面板服务使用 Python 标准库，不依赖 torch、网页构建工具或 Node.js。实际操作由指定的训练 Python 执行，仍需要训练环境中的依赖。页面右侧会显示实际解释器路径。

Ubuntu/GPUFree 部署使用工程内的 `install_ubuntu.sh`、`run_lgdog.sh` 和 `requirements-ubuntu.txt`。GPUFree 的 CUDA 驱动、数据盘、SSH 转发和显存检查记录见 [DEPLOYMENT_GPUFREE.md](DEPLOYMENT_GPUFREE.md)。安装缓存放在数据盘或工程 `.cache/`，不会进入 Git。

右侧“GPU 状态”直接读取 `nvidia-smi`，显示显存、利用率、温度和功耗。Windows 没有 `nvidia-smi` 时会显示不可用，不影响面板的训练命令预览。

“策略输出 / 控制健康”把原始动作最大值和策略裁剪比例标成“观察”项；它们统计的是策略输出，不等于电机力矩饱和。强告警只针对策略标准差发散。旁边的“速度追踪 / 地形课程”显示 XY/Yaw 误差、移动指令追踪分数、移动指令比例、速度课程分数和当前 `vx` 范围，以及地形等级、tracking ratio、升级/降级、速度门和进度门通过率。速度课程追踪分数只用移动指令计算，避免站立样本把分数抬高；另显示速度课程的有效进度/门限、分数门、进度门、平均存活比例、失败率和扩速前置条件。地形组合进度门细分为完成度≥75%、距离≥2米和平台外暴露≥1秒；各项百分比表示对应门的通过率。前置条件通过仍须满足分数阈值与连续次数。旧run没有新增指标时显示“—”。地形降级多于升级时先检查课程难度和回合失败，不要只根据原始动作裁剪率判断策略失败。

## 页面怎么用

1. 先选择“算法 / 模型”和“场景”。训练任务由这两个选择自动决定；HIM-PPO 提供 Flat、标准 Rough 和温和 Rough，PPO 提供平地、温和 Rough、标准盲走 Rough、TerrainV4 和带扫描 Rough。两者共用 PPO 主优化流程，checkpoint 仍按输入合同独立保存。
2. 切换训练、Play、Sim2Sim 或奖励。点击“快速检查”只填写 64 环境/2 轮，不会自动运行。
   页面顶部的“运行设置 / 监控数据”可以切换界面：运行设置只保留任务和启动参数，监控数据单独显示策略健康、速度/地形课程、GPU、命令和地形编辑；训练参数/课程设置本身也可以折叠。
3. Play 选择 `.pt`，Sim2Sim 选择 `.onnx`。也可填写自定义路径；没有模型时不能启动回放。续训只能选择当前算法和场景的 checkpoint，Flat 模型不能直接续训崎岖或障碍场景。
4. 查看右侧的实际命令，再点击启动。页面下方显示进程 PID、运行状态和真实输出。
5. 打开“查看奖励”后，可以看到当前方案的全部 reward。直接编辑权重或参数 JSON，取消勾选会把权重设为 0，点击“保存奖励修改”即可写入 `reward_cfg.py`。保存前会生成 `reward_cfg.py.bak`，修改需要重启训练进程才会生效。
6. 奖励区支持名称筛选和“恢复基线”。黄色行表示当前 profile 有覆盖值；双击 reward 会显示实际函数源码和绑定位置。继承自 mjlab 的函数如果能在当前 Python 环境找到，也会直接显示依赖源码。每项显示权重方向：代价应为负数或 0，正代价标红且拒绝保存。旧的错误覆盖可打开修正；更新代码后点击“恢复基线”再保存，可清除当前方案的本地覆盖。

> ⚠️ **“保存奖励修改”写进的是 `DASHBOARD_REWARD_OVERRIDES`，它的优先级高于 `REWARD_PROFILES`。**
> 试出满意的值后要手工搬进 `REWARD_PROFILES[配置名]["weights"]`，再把面板覆盖清空，然后才提交 ——
> 否则提交的 profile 改动会被面板覆盖静默压掉，`test_standard_reward_tuning_does_not_change_v7` 也会失败。
> 完整流程和踩坑记录见 [REWARDS.md](REWARDS.md#★-面板调参试出来的值必须搬进-profile-再提交)。
7. 选中运行记录后点“停止任务”，后端会结束该任务及其子进程，并核验退出结果；失败会显示原因，可重试“强制结束”。停止不会补存尚未保存的训练步，已存 checkpoint 保留。
8. “删除记录”删除所选任务的面板记录、面板启动日志和历史运行条目；“清理已结束记录”批量清理已完成、失败、已停止、已结束的任务。运行中或停止中的记录禁止删除；身份未核验的旧记录可单独删除，但不会被批量清理。两个入口都需要在弹窗中确认。训练数据、TensorBoard events、checkpoint、ONNX 和奖励修改记录不会被删除。

Play 默认使用 Viser 浏览器界面，启动后打开页面中的“机器人视图”（通常 `http://127.0.0.1:8080`，以日志输出为准）。也可以切换为原生 MuJoCo 窗口。Sim2Sim 使用普通 MuJoCo，面板会显示每张内置地图的 geom 数量和推荐出生位置；填写外部 XML 路径后，会优先加载该地形，适合把测试地图移植到 sim2sim。面板还可以配置出生 x/y/朝向、朝向保持、动作裁剪和训练观测对拍。

“启动 TensorBoard”使用当前训练 Python 和绝对日志目录，稍后点击“曲线页面”（默认端口 6006）。若端口已被其他服务占用，错误会显示在任务日志中。

关闭浏览器不会停止任务。面板服务退出后，子进程仍直接向日志文件写输出；再次启动面板时，会通过操作系统的进程创建标识核验并恢复仍活着的任务控制。Linux 同时核对启动时刻与本次开机 ID，Windows 使用进程创建时间，避免 PID 被复用后误停其他进程。早期 Linux 记录只有命令时，需命令参数和工程工作目录同时匹配才会恢复；无法核验的旧记录显示“历史·未接管”并说明原因，不会直接按 PID 杀进程。已恢复任务也参与同类任务重复启动检查。

## 审计记录：改过哪些 reward、跑过哪些训练

页面右上角的“**历史记录**”汇总两类追加日志，写在 `logs/dashboard/history/` 下：

| 文件 | 内容 | 写入时机 |
|---|---|---|
| `reward_edits.jsonl` | 每次“保存奖励修改”的实际变化：reward 名、旧值 → 新值、参数变化、备份文件 | 保存成功后 |
| `train_runs.jsonl` | 每个操作一条 `run_start` 和一条 `run_end`：模式、任务、命令、PID、日志路径、状态、返回码、耗时 | 进程启动时 / 进程结束时 |

配套细节：

- 两者都是**单行 JSON 追加写**，文件长期增长也不会拖慢面板；读取只取尾部若干条（`GET /api/history?limit=80`）。
- 训练记录的 `run_start` 还会从命令行里解析出 `run_name` / `envs` / `iterations` / `seed`（回放和 sim2sim 记录 `checkpoint` / `onnx` / `map`），事后能还原当时想跑什么。
- 记录失败**不会影响**启停进程：`JobManager` 只把事件交给回调，异常被吞掉。
- 只有**真正变化**的 reward 才入账；点保存但值没动不会产生垃圾记录。
- 坏掉的行（手改文件、写到一半断电）会被跳过，不影响其它记录。
- 首次启动前文件不存在时，接口返回 `{"reward_edits": [], "runs": []}`。

想直接看文件：

```bash
tail -n 20 logs/dashboard/history/reward_edits.jsonl
tail -n 40 logs/dashboard/history/train_runs.jsonl
```

## 短命令

| 短名称 | 方案 |
|---|---|
| `standard` | HIM + 标准崎岖地形，无机身高度扫描 |
| `him_flat` | HIM + 平地训练，独立 checkpoint |
| `him` | 原 V7/V7b 温和地形 HIM |
| `terrain` | TerrainV4 |
| `gentle` | 普通 PPO 温和地形盲走 |
| `blind` | 普通 PPO 标准地形盲走 |
| `flat` | 平地 |
| `rough` | 带高度扫描的 Rough |

```bash
python lgdog.py tasks
python lgdog.py models standard
python lgdog.py train standard --envs 64 --iterations 2 --dry-run
python lgdog.py train standard --envs 1024 --iterations 16000 --run-name V8_him_standard_seed42
python lgdog.py train standard --envs 1024 --iterations 16000 --terrain-num-rows 20
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard --velocity-min-abs 0.4 --velocity-max-abs 0.8
python lgdog.py play standard --viewer viser
python lgdog.py sim standard --map stairs --vx 0.4 --spawn-y 4
python lgdog.py sim standard --map lgdog1 --map-file src/assets/terrains/lgdog1_testmap.xml --spawn-y 4
python lgdog.py rewards standard --term foot_clearance
python lgdog.py curves
```

MuJoCo 回放无需选择时间：打开窗口后持续运行，关闭窗口、窗口内按 Esc 或终端
Ctrl+C 结束；摔倒仍会停止。无窗口默认验证20秒。批量定时测试可在短命令中
显式添加 `--duration 20`。

标准 HIM 默认使用 Unitree/mjlab 原生 `ROUGH_TERRAINS_CFG`，10 行是难度等级、20 列是地形类型；面板里可以填写训练行数和最大难度倍率。
速度课程从填写范围的
50% 开始，使用移动指令的线速度追踪分数 EMA；分数达到 0.78、窗口有效指令进度 ≥60%、完成回合的平均存活比例 ≥80%、非超时失败率 ≤25% 且累计移动指令路程至少1米，连续三次检查通过才会扩大一次，单次扩大 0.025。连续两次分数低于 0.62、有效进度低于 40%（有足够移动样本时）、存活比例低于 65% 或失败率高于 35% 时，会回退 0.025，最低退回初始范围。中间区间保持，最大扩大到面板上限；偏航角速度保持配置的最终范围。面板输入正数，训练会把最低值转换为负向下限、最高值转换为正向上限，并同步横移
范围。地形课程同时使用命令完成度、跟踪误差、存活时长和出生平台外暴露时间，满足条件才升级，
摔倒或跟踪质量过低则降级。实验名称留空时，会按
已有 `V` 编号自动递增。

LgDog 使用工程内的 [terrain_cfg.py](../src/tasks/velocity/config/lgdog1/terrain_cfg.py) 保存原生地形的隔离副本和本项目编辑值。工作台的“崎岖地形配置 / 编辑”就是修改这个文件；标准 Rough/HIM 会读取它，Flat 平地任务不会读取崎岖地形生成器。
默认包含 7 类：`flat` 平地（20%）、`pyramid_stairs` 上台阶（20%，5–15 cm）、
`pyramid_stairs_inv` 下台阶（20%，5–15 cm）、`hf_pyramid_slope` 上坡（10%，坡度 0–0.4）、
`hf_pyramid_slope_inv` 下坡（10%，坡度 0–0.4）、`random_rough` 随机起伏（10%，噪声 0.02–0.10）、
`wave_terrain` 波浪（10%，振幅 0–0.20，4 个波）。
课程模式下，列分配地形类型，行表示 difficulty；修改 `proportion` 会改变列分配，修改
`step_height_range`、`slope_range`、`noise_range` 或 `amplitude_range` 会改变对应地形的难度范围。
例如把原生上台阶最大高度改成 15 cm：

```python
cfg = lgdog1_rough_terrain_cfg()
cfg.sub_terrains["pyramid_stairs"].step_height_range = (0.0, 0.15)
```

实际项目中直接编辑 `terrain_cfg.py` 的返回函数即可；修改地形后必须新开训练 run，旧 checkpoint 的课程状态不能代表新地形能力。
工作台“打开 3D 查看器”会按所选任务执行 `python -m scripts.visualize_terrain --task <任务ID>`，
并使用独立空闲端口；地址以工作台返回值/日志为准。远程使用时需转发该端口，启动后等待地图生成再刷新页面。
终端直接运行时默认端口8080，可用 `--port` 指定。查看器修改滑块只影响预览；
工作台“保存当前地形修改”编辑普通Rough的 `terrain_cfg.py`，2027 MGDP 图由 `passage_terrain_cfg.py` 定义。

`--dry-run` 只显示命令。Play/Sim2Sim 默认从该任务独立实验目录中寻找最新已有模型，跳过尚无模型的 run。Flat/Rough 共用实验目录，必须显式选择对应模型：`python lgdog.py play flat --checkpoint /path/to/model.pt`。

续训用 `python lgdog.py train standard --resume --checkpoint logs/rsl_rl/对应实验/某个run/model_100.pt`。面板会拒绝其它算法或场景的 checkpoint，避免把平地模型接到崎岖/障碍任务造成观测和动力学合同不一致。

## 目录整理说明

```text
unitree_rl_mjlab_lgdog/
├── lgdog.py                 日常统一入口
├── README.md                从这里开始
├── src/                     本体、任务和本地面板实现
│   ├── dashboard/           轻量服务、命令生成与网页
│   └── tasks/velocity/config/lgdog1/reward_cfg.py
├── scripts/                 原始训练/回放/部署脚本
├── logs/                    原有模型和日志 + dashboard 日志
├── doc/                     使用说明、架构、训练记录、安装文档
├── archive/                 历史补丁与验证清单，不是当前工程的完整打包
├── deploy/  simulate/       原有 C++ 部署与仿真代码
└── tests/                   功能与配置检查
```

内层目录的 `.git`、源码、资产、日志与模型一起移动到了上一层，未删除训练数据。旧文档和补丁保留在 `doc/` 与 `archive/`；原 README 保存在 `doc/UPSTREAM_README.md`、`doc/UPSTREAM_README_zh.md`。绝对路径包含旧内层目录的编辑器书签或命令需要更新；训练机复制工程后重新执行 editable 安装即可。

`tb.sh` 也已改为从自身位置定位工程、从当前环境寻找 tensorboard，不再写死用户名和 Conda 路径。


## 自动地形课程（可选）

训练参数区的“地形课程”包含手工门控、固定基础地形比例对照、LP稳定版v2。自动模式使用填写的固定速度包络，不执行旧地形升降或扩速；平地任务禁用。可编辑基础比例混合量（默认0.5）、单任务概率上限（默认0.25）、进度/概率平滑系数、温度、最少回合数及收集窗口。100秒窗口结束后还会等待完整回合时长才更新。基础比例来自地形编辑器，同类型内均分任务。

“监控数据”中的自动地形课程卡片可展开各任务的采样概率热图、LP、最近有效回报与累计回合数。它读取当前run的 `curriculum/terrain_sampling.json`，不把初次/缺失回报当作进步；旧run无文件时隐藏。等级均值允许主动回访下降，应结合任务统计与固定评估判断。完整操作见 [LP_ACRL_INTEGRATION.md](LP_ACRL_INTEGRATION.md)。

## 比赛尺寸强化课程：高墙规则

选择“MGDP · 比赛尺寸强化课程”，进入“查看奖励”后的“高墙与越障规则”，可编辑进展门控、等待惩罚、每墙奖励、距离奖励、出口奖励和无进展终止。保存后仅新启动的训练读取；设置备份、共享基础权重范围与续训兼容说明见 [高墙规则工作台](MGDP_WALL_WORKBENCH.md)。

D435i下俯20°使用新增独立入口，配置与覆盖步骤见[下俯20°配置](MGDP_D435I_DOWN20.md)。原相机入口保留。

定向越障使用独立速度模式：比赛尺寸强化课程新训练默认从0.6m/s渐进扩速，上限2.0m/s；可切换固定巡航。普通任务正负速度包络不会当成固定速度。运行面板按真实日志显示模式和范围，详见[固定巡航速度](MGDP_FIXED_CRUISE.md)。

比赛尺寸强化课程另有默认关闭的“轮胎内孔落脚引导”复选框：左右脚分别落进对应内孔，首次承重触地按每脚每孔每回合+0.25去重发放，预览命令包含`--tire-hole-guidance=True`。改变开关须新建实验或仅迁移旧策略，完整续训保持原设置；详见[轮胎落脚引导](MGDP_TIRE_HOLE_GUIDANCE.md)。
