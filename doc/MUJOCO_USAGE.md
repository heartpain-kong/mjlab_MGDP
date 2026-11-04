# 本项目 MuJoCo 使用说明

所有命令在仓库根目录、已安装项目依赖的 Python 环境执行。Linux 常用 `source .venv/bin/activate`；训练依赖安装见 [setup_zh.md](setup_zh.md)。本项目固定的 MuJoCo 版本是 3.5.0。

## 选哪个入口

| 目标 | 入口 | 需要什么 |
|---|---|---|
| 看机器人模型和任务，不加载策略 | `scripts.play --agent zero` | 项目仿真依赖 |
| 看训练任务中的策略表现 | `scripts.play` / `lgdog.py play` | model_XXXX.pt；MGDP 还需原感知文件 |
| 用普通 CPU MuJoCo 验证导出的策略 | `scripts.sim2sim` | policy.onnx；MGDP 还需配套 policy.perception.pt |
| 网页选择任务、训练、回放 | `lgdog.py` | 工作台本身使用标准库，仿真子进程需要项目依赖 |

`play` 的 `native` 是本地 MuJoCo 窗口，`viser` 是浏览器查看器。服务器没有桌面时选 `viser`，按终端打印的地址和端口建立 SSH 转发；原生窗口需要本地显示环境。

## 1. 先看机器人

```bash
python -m scripts.play LgDog1-Flat --agent zero --num-envs 1 --viewer native --device cpu
```

这里零动作对应默认关节姿态，只用于检查模型/场景，不代表训练策略的效果。选择 Flat 可以避免 MGDP 的预训练前置条件。

## 2. 查看新 MGDP 定向训练任务的策略

训练输出位于 `logs/rsl_rl/lgdog1_velocity_race_mgdp_v3/<run>/`。自动选择对应目录的最新检查点：

```bash
python lgdog.py play race_mgdp --viewer native --envs 1
```

指定检查点（把 `<run>`、`<iteration>` 替换成实际名称）：

```bash
python -m scripts.play LgDog1-Race-MGDP --checkpoint_file "logs/rsl_rl/lgdog1_velocity_race_mgdp_v3/<run>/model_<iteration>.pt" --num-envs 1 --viewer native --device cpu
```

服务器用 `--viewer viser`。`play` 使用任务观测项加载其配置中的感知检查点（比赛v3默认 `logs/mgdp/perception/mgdp_race_v3.pt`，普通MGDP默认 `logs/mgdp/perception/mgdp_perception.pt`），必须保持与训练时一致；如果要连同导出的配套感知文件独立回放，使用下一节的 sim2sim。v3 路线任务默认从最低难度开始查看，不等于全难度评估。

## 3. 用 ONNX 跑比赛场地

新 MGDP 导出应保留两个文件：

```text
<run>/policy.onnx
<run>/policy.perception.pt
```

sim2sim 会按 ONNX 的文件名记录查找感知权重，并检查 SHA-256。先用平地、低速确认接口与姿态：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v3/<run>/policy.onnx" --map flat --viewer --vx 0.3 --no-keyboard
```

再切换完整比赛障碍 XML：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v3/<run>/policy.onnx" --map race --viewer --vx 0.3 --no-keyboard
```

- `--vx / --vy / --wz`：前进、横移、偏航速度指令；正 vx 向机器人前方走。
- `--spawn-x / --spawn-y / --spawn-yaw`：出生位置与朝向（yaw 单位为度），用于对准障碍。
- `--map race`：完整障碍赛场地；`race_task`：任务赛场地；其他可选地图有 `flat/stairs/slope/obstacles/mixed/lgdog1`。
- `--viewer`：显示窗口；省略它可在 CPU 无窗口运行。
- `--no-keyboard`：保持命令行给定指令，不监听键盘。

固定 vx 不会自动规划全赛道路线。单类训练通道的通过率、完整场地的通过情况应分别记录；“未摔倒”不等于“越过障碍”。

## 4. MuJoCo 原生窗口的键盘和鼠标

以下针对 `native` / `--viewer` 打开的 MuJoCo 窗口；先把焦点放到窗口。浏览器 `viser` 的控件另有自己的操作方式。

| 键盘/鼠标 | 功能 |
|---|---|
| F1 | 显示/隐藏帮助，直接查看当前窗口的操作表 |
| F2 | 显示/隐藏运行信息 |
| F3 / F4 | 切换性能分析图 / 传感器图；被动查看器中内容可能受外部循环限制 |
| F5 | 切换全屏 |
| F6 / F7 | 循环切换坐标系显示 / 标签显示 |
| Tab / Shift+Tab | 显示或隐藏左侧 / 右侧面板 |
| `[` / `]` | 切换模型中定义的固定相机；模型没有固定相机时无可切换项 |
| 鼠标左键拖动 | 旋转观察视角 |
| 鼠标右键拖动，配合 Shift | 平移视角，Shift 改变移动方向 |
| 滚轮 / 中键拖动 | 缩放视角 |
| 左键双击部件 | 选中该部件 |
| 右键双击部件 | 将视角中心移到该位置 |
| Ctrl+右键双击 | 跟踪选中的物体 |
| Page Up | 将选中对象切换到其父 body |
| 选中后 Ctrl+左键拖动 / Ctrl+右键拖动 | 对所选物体做旋转 / 平移扰动；配合 Shift 改变方向。运行与暂停状态下的响应不同 |
| 在 UI 区域按住右键 | 查看面板控件的快捷键 |
| 双击 UI 分区标题 | 展开/折叠各分区 |

常用抗扰动操作：双击机器人部件选中，然后按 Ctrl+右键拖拽，观察受到推拉后的恢复情况。

### `play --viewer native` 的仿真控制（mjlab 1.2.0）

该入口通过 mjlab 给被动查看器补上了以下控制，不能与下一节的 sim2sim 遥控键混用：

| 按键 | 功能 |
|---|---|
| Space | 暂停/继续策略与仿真循环 |
| Enter | 复位环境 |
| 右方向键 | 暂停后单步推进 |
| `=` / `-` | 加速 / 减速播放 |
| `,` / `.` | 切换上一个 / 下一个环境 |
| P | 显示/隐藏奖励图 |
| R | 显示/隐藏任务调试可视化 |
| A | 切换是否显示全部环境 |

MuJoCo 原生 Esc 用于回到自由相机；本项目 sim2sim 将窗口中的 Esc 作为退出键，无需启用键盘遥控。

### 独立 simulate 与 sim2sim 的区别

独立 MuJoCo simulate 自己驱动物理循环，支持 Space 暂停、`=`/`-` 调速、左右方向键查看历史/单步。本项目 sim2sim 用 Python 循环驱动物理，并使用 `launch_passive()`；原生暂停/调速按钮不会自动控制这个外部循环。当前 sim2sim 未接入暂停与单步：启用项目遥控时，**Space/R 是复位**。使用 `--no-keyboard` 会禁用全局遥控，但窗口内的 **Esc 仍可退出**。

### 回放时长

工作台已移除运行时间选项。有窗口的 sim2sim 默认持续运行，关闭窗口、窗口内按 Esc
或终端按 Ctrl+C 结束；摔倒保护仍会停止。回放按实际推理/渲染耗时补偿到 50 Hz，
计算过慢时会低于实时速度。长期回放只缓存最近 100 仿真秒的逐步数据，全程均值保留。
交互结束不会被当作完成一次定时姿态测试。

批量验证仍可显式指定 `--duration 20`。无窗口且未给时长时，默认验证 20 仿真秒后退出。

核对来源：[MuJoCo 3.5.0 simulate 源码](https://github.com/google-deepmind/mujoco/blob/3.5.0/simulate/simulate.cc)（帮助表及 passive 分支），以及当前安装的 `mjlab/viewer/native/viewer.py::_safe_key_callback`。

## 5. 项目 sim2sim 机器人遥控

键盘监听使用可选的 `pynput`（缺少时可在同一环境执行 `python -m pip install pynput`）：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v3/<run>/policy.onnx" --map race --viewer --vx 0 --speed-x 0.3 --speed-y 0.1 --speed-yaw 0.4
```

W/S 前后、A/D 转向、Q/E 横移，空格或 R 复位，Esc 退出。复位会清空本体历史、两帧深度和 GRU 状态；键盘未能初始化时终端会提示原因。

## 6. 回放旧 MGDP_V3

旧 ONNX 没有绑定感知文件，需要显式指定**当时训练所用的原感知权重**：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "/path/to/MGDP_V3/policy.onnx" --perception "/path/to/original_mgdp_perception.pt" --map race --viewer --vx 0.3 --no-keyboard
```

`E:/mjlab/数据/MGDP_V3` 的归档目前缺少该感知文件，不能用新预训练文件随意替代。旧 V3 也不能完整续训成 v2；新任务说明见 [MGDP_PASSAGE.md](MGDP_PASSAGE.md)。

## 7. 机器人相机小窗（原生 MuJoCo）

`play --viewer native` 和 `sim2sim --viewer` 默认在左下角显示机器人 RGB 相机，
主画面仍可用鼠标自由观察。点击 MuJoCo 窗口后按 **F8** 隐藏/恢复小窗。
直接打开原始资产 XML 的独立 simulate 不会自动运行这段 Python 小窗逻辑。

相机随 trunk 固定运动：局部位置 `(0.02, 0, 0.075)` m，站立基座 z=0.31 m 时
镜头约高 0.385 m；光轴沿机器人 +X 水平朝前，画面右侧为 -Y、上方为 +Z。
俯仰和侧倾会随机器人变化，不是始终锁定世界地平线。方形图像垂直/水平视场角均为 87°。
参数统一在 `src/config/lgdog1.py` 的 `FRONT_CAMERA_*` 中，属于仿真安装参数。

默认小窗 256×256、10 FPS；窗口较小时自动缩小到主视口宽高的三分之一以内。
可选 `bottom-left`、`bottom-right`、`top-left`、`top-right`，例如：

```bash
python -m scripts.play LgDog1-Flat --agent zero --num-envs 1 --viewer native --camera-inset-size 320 --camera-inset-fps 15 --camera-inset-corner top-right
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "/path/to/policy.onnx" --map race --viewer --camera-inset-corner bottom-right
```

禁用：直接调用 `scripts.play` 用 `--camera-inset False`；`scripts.sim2sim` 及短入口
`lgdog.py play/sim` 用 `--no-camera-inset`。无 viewer 的 sim2sim 不创建离屏渲染器。
Viser 暂无这个角落小窗，选择原生查看器才会显示。

小窗是便于观察的 RGB 图像，和 MGDP 深度射线使用相同安装位置、前向光轴及视场角。
小窗已将画面上方向校正为机器人 +Z；策略原有 16×16 深度图的像素排列保持不变，
不会在显示功能中旋转已有策略的输入。

## 当前 Windows 工作区的 CPU 环境

本轮已经在下列解释器中完成 CPU 验证。PowerShell 可以直接调用，避免误用系统里没有依赖的 Python：

```powershell
Set-Location 'E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog'
$env:PYTHONUTF8 = '1'
$py = 'E:\mjlab\work\lgdog_validation\Scripts\python.exe'
& $py -m scripts.play LgDog1-Flat --agent zero --num-envs 1 --viewer native --device cpu
```

该环境用于本机 CPU 验证，GPU 训练机依赖仍按工程安装脚本和锁定版本安装。
