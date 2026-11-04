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

训练输出位于 `logs/rsl_rl/lgdog1_velocity_race_mgdp_v2/<run>/`。自动选择对应目录的最新检查点：

```bash
python lgdog.py play race_mgdp --viewer native --envs 1
```

指定检查点（把 `<run>`、`<iteration>` 替换成实际名称）：

```bash
python -m scripts.play LgDog1-Race-MGDP --checkpoint_file "logs/rsl_rl/lgdog1_velocity_race_mgdp_v2/<run>/model_<iteration>.pt" --num-envs 1 --viewer native --device cpu
```

服务器用 `--viewer viser`。`play` 使用任务观测项加载其配置中的感知检查点（默认 `logs/mgdp/perception/mgdp_perception.pt`），必须保持与训练时一致；如果要连同导出的配套感知文件独立回放，使用下一节的 sim2sim。v2 定向任务默认从最低难度开始查看，不等于全难度评估。

## 3. 用 ONNX 跑比赛场地

新 MGDP 导出应保留两个文件：

```text
<run>/policy.onnx
<run>/policy.perception.pt
```

sim2sim 会按 ONNX 的文件名记录查找感知权重，并检查 SHA-256。先用平地、低速确认接口与姿态：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v2/<run>/policy.onnx" --map flat --viewer --vx 0.3 --duration 20 --no-keyboard
```

再切换完整比赛障碍 XML：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v2/<run>/policy.onnx" --map race --viewer --vx 0.3 --duration 30 --no-keyboard
```

- `--vx / --vy / --wz`：前进、横移、偏航速度指令；正 vx 向机器人前方走。
- `--spawn-x / --spawn-y / --spawn-yaw`：出生位置与朝向（yaw 单位为度），用于对准障碍。
- `--map race`：完整障碍赛场地；`race_task`：任务赛场地；其他可选地图有 `flat/stairs/slope/obstacles/mixed/lgdog1`。
- `--viewer`：显示窗口；省略它可在 CPU 无窗口运行。
- `--no-keyboard`：保持命令行给定指令，不监听键盘。

固定 vx 不会自动规划全赛道路线。单类训练通道的通过率、完整场地的通过情况应分别记录；“未摔倒”不等于“越过障碍”。

## 4. 键盘控制

键盘监听使用可选的 `pynput`（缺少时可在同一环境执行 `python -m pip install pynput`）：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "logs/rsl_rl/lgdog1_velocity_race_mgdp_v2/<run>/policy.onnx" --map race --viewer --vx 0 --speed-x 0.3 --speed-y 0.1 --speed-yaw 0.4 --duration 300
```

W/S 前后、A/D 转向、Q/E 横移，空格或 R 复位，Esc 退出。复位会清空本体历史、两帧深度和 GRU 状态；键盘未能初始化时终端会提示原因。

## 5. 回放旧 MGDP_V3

旧 ONNX 没有绑定感知文件，需要显式指定**当时训练所用的原感知权重**：

```bash
python -m scripts.sim2sim LgDog1-Race-MGDP --onnx "/path/to/MGDP_V3/policy.onnx" --perception "/path/to/original_mgdp_perception.pt" --map race --viewer --vx 0.3 --duration 20 --no-keyboard
```

`E:/mjlab/数据/MGDP_V3` 的归档目前缺少该感知文件，不能用新预训练文件随意替代。旧 V3 也不能完整续训成 v2；新任务说明见 [MGDP_PASSAGE.md](MGDP_PASSAGE.md)。

## 当前 Windows 工作区的 CPU 环境

本轮已经在下列解释器中完成 CPU 验证。PowerShell 可以直接调用，避免误用系统里没有依赖的 Python：

```powershell
Set-Location 'E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog'
$env:PYTHONUTF8 = '1'
$py = 'E:\mjlab\work\lgdog_validation\Scripts\python.exe'
& $py -m scripts.play LgDog1-Flat --agent zero --num-envs 1 --viewer native --device cpu
```

该环境用于本机 CPU 验证，GPU 训练机依赖仍按工程安装脚本和锁定版本安装。
