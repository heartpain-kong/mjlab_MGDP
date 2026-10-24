# 本地训练工作台

现在主工程只有一层目录：`E:\mjlab\unitree_rl_mjlab_lgdog`。打开面板后，可用同一套界面选择方案、训练、play、sim2sim 和查看奖励。

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

## 页面怎么用

1. 选择训练方案，默认是 `standard`：标准 Rough + HIM，282 维历史输入。
2. 切换训练、Play、Sim2Sim 或奖励。点击“快速检查”只填写 64 环境/2 轮，不会自动运行。
3. Play 选择 `.pt`，Sim2Sim 选择 `.onnx`。也可填写自定义路径；没有模型时不能启动回放。
4. 查看右侧的实际命令，再点击启动。页面下方显示进程 PID、运行状态和真实输出。
5. 打开“查看奖励”后，可以看到当前方案的全部 reward。直接编辑权重或参数 JSON，取消勾选会把权重设为 0，点击“保存奖励修改”即可写入 `reward_cfg.py`。保存前会生成 `reward_cfg.py.bak`，修改需要重启训练进程才会生效。
6. 奖励区支持名称筛选和“恢复基线”。黄色行表示当前 profile 有覆盖值；双击 reward 会显示实际函数源码和绑定位置。继承自 mjlab 的函数如果能在当前 Python 环境找到，也会直接显示依赖源码。
7. 停止按钮仅停止本次面板启动的对应进程。停止不会补存尚未保存的训练步，已存 checkpoint 保留。

Play 默认使用 Viser 浏览器界面，启动后打开页面中的“机器人视图”（通常 `http://127.0.0.1:8080`，以日志输出为准）。也可以切换为原生 MuJoCo 窗口。Sim2Sim 使用普通 MuJoCo，勾选窗口时显示在运行服务的电脑上。

“启动 TensorBoard”使用当前训练 Python 和绝对日志目录，稍后点击“曲线页面”（默认端口 6006）。若端口已被其他服务占用，错误会显示在任务日志中。

关闭浏览器不会停止任务。面板服务退出后，子进程仍直接向日志文件写输出；再次启动面板会显示历史记录，但不会直接接管旧进程。标为“历史·状态未知”的任务需根据 PID 在系统中检查；只有本次服务启动的进程有停止按钮。避免在不确认旧任务状态时重复启动训练。

## 短命令

| 短名称 | 方案 |
|---|---|
| `standard` | HIM + 标准崎岖地形，无机身高度扫描 |
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
python lgdog.py train standard --envs 1024 --iterations 16000 --max-terrain-level 3
python lgdog.py play standard --viewer viser
python lgdog.py sim standard --map stairs --vx 0.4 --spawn-y 4
python lgdog.py rewards standard --term foot_clearance
python lgdog.py curves
```

训练面板或短命令里实验名称留空时，会按已有 `V` 编号自动递增。地形等级上限按内部
0 起始等级计数，例如 `3` 表示允许等级 0–3。

`--dry-run` 只显示命令。Play/Sim2Sim 默认从该任务独立实验目录中寻找最新已有模型，跳过尚无模型的 run。Flat/Rough 共用实验目录，必须显式选择对应模型：`python lgdog.py play flat --checkpoint /path/to/model.pt`。

续训用 `python lgdog.py train standard --resume --checkpoint logs/rsl_rl/对应实验/某个run/model_100.pt`，模型必须属于当前任务的实验目录。新标准 HIM 与 V7 的 critic 维度不同，不能通过此入口跨任务恢复完整 checkpoint。

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
