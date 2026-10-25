# GPUFree Ubuntu 22.04 部署记录

这份记录整理了在 GPUFree Ubuntu 22.04 GPU 容器中部署和运行 LgDog 的实测步骤。训练环境放在数据盘 `/root/gpufree-data`，避免占满 30GB 系统盘。GPUFree 实例支持 Jupyter 网页终端和 SSH；Docker 在平台容器中不可用。

## 1. 解压和环境位置

把工程放在数据盘，虚拟环境也放在数据盘：

```bash
cd /root/gpufree-data
unzip unitree_rl_mjlab_lgdog.zip -d unitree_rl_mjlab_lgdog-src
find unitree_rl_mjlab_lgdog-src -maxdepth 4 -name setup.py -print
```

进入包含 `setup.py`、`lgdog.py` 和 `src/` 的目录：

```bash
python3 -m venv /root/gpufree-data/lgdog-venv
source /root/gpufree-data/lgdog-venv/bin/activate
export PIP_CACHE_DIR=/root/gpufree-data/.cache/pip
```

压缩包只用于代码部署时，可以删除 `logs/`、`.vscode/`、`.git/` 和 `__pycache__/`。训练日志和 checkpoint 需要保留时，只保留目标实验目录。

## 2. CUDA/Warp 驱动修复

GPUFree 的 `nvidia-smi` 和 PyTorch 可以正常识别 RTX 4090，但 Warp 可能因为 `/usr/local/cuda/lib64` 中的 CUDA stub 库而报：

```text
CUDA error 34
CUDA driver not available
Invalid device identifier: cuda:0
```

在启动训练或工作台前设置真实驱动库：

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
```

检查：

```bash
ldconfig -p | grep libcuda.so.1
python - <<'PY'
import torch
import warp as wp
print("torch cuda:", torch.cuda.is_available())
wp.init()
print("warp device:", wp.get_device("cuda:0"))
PY
```

只有看到 `warp device: cuda:0` 后才开始 GPU 训练。网页工作台启动的子进程会继承工作台进程的环境变量，所以必须先设置变量，再启动工作台。

## 3. 网页工作台和 Windows 浏览器

工作台只监听本机回环地址。GPUFree 终端启动：

```bash
source /root/gpufree-data/lgdog-venv/bin/activate
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
cd /root/gpufree-data/unitree_rl_mjlab_lgdog
python lgdog.py ui --no-browser --port 8088
```

Windows PowerShell 使用 GPUFree 控制台复制的 SSH 地址建立隧道。示例：

```powershell
ssh -N -L 8088:127.0.0.1:8088 root@183.147.142.40 -p 30420
```

浏览器打开 `http://127.0.0.1:8088`。本机端口和远程工作台端口使用同一个数字，因为工作台会校验本机 Host 和监听端口。

## 4. 训练显存经验

RTX 4090 24GB 可以正常初始化 `LgDog1-Rough-Blind-HIM-Standard` 的 CUDA 环境。第一次运行会编译 Warp/MuJoCo-Warp 内核，缓存位于 `/root/.cache/warp/1.12.0`，后续启动会快很多。

8192 个环境在这套标准崎岖地形 HIM 配置下会在 CUDA Graph 创建阶段显存溢出：

```text
Warp CUDA error 2: out of memory
```

建议先用 64 个环境做链路检查，再用 2048 个环境正式训练；确认显存余量后再尝试 4096。2048 的启动示例：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --agent.run-name=V10_seed42_env2048 \
  --enable-nan-guard True
```

查看 GPU：

```bash
watch -n 1 -d nvidia-smi
```

## 5. TensorBoard 和端口转发

GPUFree 终端启动 TensorBoard：

```bash
/root/gpufree-data/lgdog-venv/bin/python -m tensorboard.main \
  --logdir /root/gpufree-data/unitree_rl_mjlab_lgdog/logs/rsl_rl \
  --host 127.0.0.1 \
  --port 6006
```

Windows PowerShell 使用没有冲突的本机端口：

```powershell
ssh -N -L 8089:127.0.0.1:6006 root@183.147.142.40 -p 30420
```

浏览器打开 `http://127.0.0.1:8089`。`TensorFlow installation not found` 不影响读取 TensorBoard 标量。旧实验的 event 文件如果出现 `BadLengthCrc`，通常是上次中断留下的不完整文件，可以用 `--load_fast=false` 或只指向当前实验目录。

## 6. 部署验收顺序

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
python scripts/list_envs.py --keyword LgDog1
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=64 \
  --agent.max-iterations=2 \
  --agent.logger=tensorboard \
  --enable-nan-guard True
```

验收时分别确认：Warp 输出 `cuda:0`、环境初始化完成、两轮 PPO 更新结束、日志目录出现 event 文件。自检通过后再扩大环境数和训练轮数。
