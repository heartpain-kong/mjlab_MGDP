# Conda 环境 `lgdog`：一键配置

服务器（Ubuntu 22.04/24.04、带 NVIDIA 驱动的租用机）上，用一条命令把训练环境配好：

```bash
cd /root/gpufree-data/unitree_rl_mjlab_lgdog
bash configure_lgdog.sh
```

它做的事：

1. 定位/加载 Conda（`/root/miniconda3`、`/opt/conda`、`$HOME/miniconda3` 都能自动认出）。
2. 安装系统依赖（`build-essential cmake git pkg-config python3-dev libegl1 libgl1 …`）；
   `apt-get update` 撞上 NVIDIA 源坏索引时，自动改用「只读 Ubuntu 清华源」的方式重试。
3. 创建 Conda 环境 **`lgdog`**，默认前缀 `/root/gpufree-data/conda/envs/lgdog`
   （数据盘不可写时退回 `$HOME/.conda/envs/lgdog`）。已存在则直接复用。
4. 安装锁定依赖：`torch>=2.7,<2.8` → `requirements-ubuntu.txt`（mjlab 1.2.0 / mujoco 3.5.0 /
   mujoco-warp 3.5.0 / warp-lang 1.12.0 / rsl-rl-lib 5.0.1 …）→ `pip install -e .`。
5. 写入激活钩子：`conda activate lgdog` 时自动导出
   - `PYTHONPATH=<项目根>`、`PYTHONUNBUFFERED=1`、`MUJOCO_GL=egl`
   - CUDA 驱动修复：`LD_LIBRARY_PATH` 加驱动目录、`LD_PRELOAD=libcuda.so.1`
     （容器里 Warp 报 `CUDA driver not available / CUDA error 34` 就是缺这个）
   - 反激活时原样恢复 `LD_PRELOAD` / `LD_LIBRARY_PATH`
6. 把解释器路径写进 `logs/server/python_path.txt`，`run_lgdog.sh` 之后自动用它。
7. 运行 `python scripts/verify_install.py --require-cuda`：检查依赖版本、任务注册、
   以及 **PyTorch + Warp 的真实 CUDA 分配**，通过才报成功。

## 常用参数

| 参数 | 作用 |
|---|---|
| `--env-dir PATH` | 指定环境前缀（默认 `/root/gpufree-data/conda/envs/lgdog`） |
| `--python-version 3.11` | 新建环境用的 Python 版本 |
| `--torch-index URL` | 指定 PyTorch 源，例如 `https://download.pytorch.org/whl/cu128` |
| `--skip-system-deps` | 跳过 apt（系统库已装好、或无 sudo 时） |
| `--skip-check` | 跳过最后的 CUDA 自检 |
| `--start --ui-port 8088` | 配置完成后直接拉起工作台 |
| `--cpu` | 装 CPU 版 PyTorch（只做导入检查用，不能训练） |

典型用法：

```bash
# 标准安装
bash configure_lgdog.sh

# NVIDIA 源坏了、系统库也齐全时
bash configure_lgdog.sh --skip-system-deps

# 指定 PyTorch CUDA 版本并顺手起工作台
bash configure_lgdog.sh --torch-index https://download.pytorch.org/whl/cu128 --start --ui-port 8088
```

## 激活与使用

```bash
conda activate /root/gpufree-data/conda/envs/lgdog
cd /root/gpufree-data/unitree_rl_mjlab_lgdog

python lgdog.py tasks
bash run_lgdog.sh server start --ui-port 8088
```

工作台、总览、TensorBoard 的端口分别是 8088 / 8770 / 6006（可用 `--ui-port`、`--portal-port`、`--tb-port` 改）。

## 常见问题

**`EnvironmentNameNotFound: Could not find conda environment: lgdog`**
环境是用 `conda create --prefix` 建的，父目录不在 conda 的 `envs_dirs` 里，按名字找不到。
先确认目录还在：

```bash
ls -d /root/gpufree-data/conda/envs/lgdog/bin/python
conda activate /root/gpufree-data/conda/envs/lgdog     # 按路径激活，立即可用
```

再让按名字激活也能用（写进 `~/.condarc`）：

```bash
conda config --append envs_dirs /root/gpufree-data/conda/envs
conda activate lgdog
```

`configure_lgdog.sh` 会自动做这一步；如果目录根本不存在，重新跑 `bash configure_lgdog.sh --skip-system-deps` 即可。

**`CUDA driver not available` / `Invalid device identifier: cuda:0`**
说明激活钩子没生效：确认是用 `conda activate <prefix>` 而不是直接调 `python`，
或手动执行 `source server_env.sh && lgdog_server_env "$PWD"` 后再启动训练。

**`apt-get update` 报 `File has unexpected size`**
属于 NVIDIA 软件源索引损坏；`configure_lgdog.sh` 会自动回退到 Ubuntu-only 源。
如果仍然失败，用 `--skip-system-deps` 跳过系统依赖（VNC/EGL 库镜像里通常已有）。

**`pip install` 太慢**
先 `export PIP_CACHE_DIR=/root/gpufree-data/.cache/pip`，或加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

**想换环境位置**
`LGDOG_ENV_DIR=/opt/lgdog bash configure_lgdog.sh`，或 `--env-dir /opt/lgdog`。
