# mjlab 总览：一个本机网页看全部

打开这一个页面，就能回答日常那几个问题：**训练在跑吗？GPU 在动吗？最新一轮练到哪了？
日志里最后几行是什么？工作台 / 曲线终端 / TensorBoard 开着没有？**

## 启动

```bash
python lgdog.py web              # 浏览器打开 http://127.0.0.1:8770
bash run_web.sh                  # Linux：等价，Python 解释器沿用 run_lgdog.sh 的规则
run_web.cmd                      # Windows：双击即可
python -m src.portal             # 等价入口
```

默认会**顺手把训练工作台（:8765）和曲线终端（:8766）带起来**，所以只记一个地址。
不想自动带的话：

```bash
python lgdog.py web --no-autostart
```

点一下服务卡上的「停止」只停**本页启动的**进程；别处起的同端口服务只显示「已在运行」，没有停止按钮。
关掉总览（Ctrl+C）不会停子服务，和训练工作台的行为一致。

## 页面分区

| 区域 | 内容 |
|---|---|
| 概要四格 | 训练状态（有没有 `train.py` 进程）、显卡、磁盘、归档轮次 |
| 服务卡 | 训练工作台 / 曲线终端 / TensorBoard：状态灯、端口、打开、启动、停止 |
| 训练轮次 | 每个 run 的平均回报、迭代、地形等级、跟踪比、失败率、训练到什么时候、大小、一键跳 K 线 |
| 环境 | Python / torch / mjlab / mujoco / mujoco-warp / rsl-rl / tensorboard 版本，CPU 与内存 |
| 显卡 | `nvidia-smi` 的显存、利用率、温度、功耗；没有 NVIDIA 驱动时退回 WMI 只报型号 |
| 相关进程 | 训练 / 回放 / Sim2Sim / 工作台 / 曲线终端 / TensorBoard，带 CPU、内存、运行时长 |
| 最近的操作 | 工作台写下的 `logs/dashboard/history/train_runs.jsonl`（谁在什么时候起了什么） |
| 日志 | 服务日志（`logs/web/*.log`）和 `logs/` 下的任意 `.log`，只读尾部、可自动滚动 |

轮次指标来自曲线终端：它在跑时总览会直接读 `/api/overview`（解析过、缓存在
`logs/kline_cache/`），所以首屏就能看到数字；曲线终端没开时只显示文件级信息（名字、大小、时间）。

## 数据从哪来

按顺序找，存在的才算：

1. `<工程>/logs/rsl_rl/`
2. 往上一级 / 两级的 `数据/` 或 `data/`（这份工程的数据归档在 `E:\mjlab\数据\V1…V14/`）
3. `--logdir` 额外给的目录，可给多次：
   `python lgdog.py web --logdir D:/archive/runs --logdir ../another`

## 端口与安全

| 服务 | 端口 | 说明 |
|---|---:|---|
| 总览（本页） | 8770 | 默认只绑 `127.0.0.1` |
| 训练工作台 | 8765 | 由总览启动；也可以自己 `python lgdog.py` |
| 曲线终端 | 8766 | 由总览启动；也可以 `python lgdog.py kline` |
| TensorBoard | 6006 | 默认不自动起，点卡上的「启动」；日志目录取第一个数据源 |

- 本页**只读**：没有任何写训练产物、删文件、改奖励的接口。
- 唯一的写操作是「启动 / 停止子服务」，需要页面里的 token（`X-Mjlab-Token`）+ 同源校验，
  跨站请求会被拦掉。
- 想在手机/平板上看，用 `--host 0.0.0.0`，然后访问 `http://<这台机器的局域网 IP>:8770`。
  这样整页都对局域网可见（只读数据），启动/停止仍然要 token，但**默认不要开**：
  `--host 0.0.0.0` 只是给「自己家的局域网」用的。

## 开机自启（可选）

Windows（任务计划程序，登录时启动）：

```powershell
$action = New-ScheduledTaskAction -Execute "pythonw.exe" `
  -Argument "E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog\lgdog.py web --no-browser" `
  -WorkingDirectory "E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog"
Register-ScheduledTask -TaskName "mjlab-web" -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn)
```

Linux（systemd user unit，`~/.config/systemd/user/mjlab-web.service`）：

```ini
[Unit]
Description=mjlab overview
[Service]
WorkingDirectory=/path/to/unitree_rl_mjlab_lgdog
ExecStart=/bin/bash -lc 'bash run_web.sh --no-browser'
Restart=on-failure
[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now mjlab-web
```

## 常见问题

**「无 NVIDIA GPU / 未找到 nvidia-smi」**
这台机器没有 NVIDIA 驱动（比如 Intel Arc 核显）。mjlab 训练要 CUDA，所以本页会直接说明
「更适合看，而不是训」；曲线、轮次、日志照常可用。

**轮次全是「—」或「解析中」**
曲线终端刚开始解析（60 MB 的 event 文件第一次要十几秒，之后走缓存）。等几秒刷新即可；
`logs/kline_cache/` 可以随时删，删了只是下次重新解析。

**服务卡显示「已在别处运行」，没有停止按钮**
那个端口上是别的进程（可能是你自己在终端里起的）。停止请回它自己的窗口，本页只负责显示。

**TensorBoard 起不来**
看「日志」里选 TensorBoard 的尾巴：常见原因是 6006 端口被占、或当前 Python 没装 tensorboard。
这里的 TensorBoard 用**当前数据源的第一项**做 `--logdir`，归档目录是扁的（`数据/V14/`），
TensorBoard 会把每个 V 目录当成一个 run 展示。

**Windows 上进程列表偶尔出现两行一样的**
某些 venv 的 `python.exe` 是个「壳」，壳和真正的解释器命令行只差解释器路径。页面会把这种情况
合并成一行；如果还看到两行，那是真的有两份在跑。

## 和另外两个页面的关系

| 页面 | 干什么 |
|---|---|
| **mjlab 总览（本页，:8770）** | 看状态、看环境、起停三个服务、翻日志 |
| 训练工作台（:8765） | 真正「跑」：选任务、起训练 / 回放 / Sim2Sim、改奖励、看任务输出 |
| 曲线终端（:8766） | 看曲线：TensorBoard 转 K 线、分时、均线、轮次对比、数据体检 |

工作台和曲线终端各自也能单独启动，总览只是把「先起哪个、地址是什么」这件事收成一处。
