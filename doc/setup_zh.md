# Ubuntu 安装与部署

## 推荐：一键安装

适用于 Ubuntu 22.04/24.04、x86_64 和 NVIDIA GPU 算力租借实例。将仓库上传或克隆后，在项目根目录运行：

```bash
bash install_ubuntu.sh
source .venv/bin/activate
python scripts/verify_install.py
```

脚本会依次完成：

1. 安装编译器、MuJoCo 图形运行库和项目需要的 C++ 开发库；
2. 创建项目内的 `.venv`，不污染系统 Python；
3. 安装 PyTorch 2.7、mjlab 1.2.0、MuJoCo 3.5.0、MuJoCo-Warp 3.5.0 和 Warp 1.12.0；
4. 用 editable 模式安装当前源码；
5. 导入训练依赖并确认 `LgDog1-Rough-Blind-HIM` 与标准 HIM 任务已注册。

默认会使用 `python3.10`–`python3.13` 中找到的第一个版本。训练需要 NVIDIA 驱动；脚本最后的自检只验证导入和任务注册，不会启动长时间训练。

## 算力自由平台命令

平台的系统盘空间有限，数据盘路径是 `/root/gpufree-data`。创建 Ubuntu 22.04 GPU 实例后，在 Jupyter 终端或 SSH 终端执行：

```bash
nvidia-smi
cd /root/gpufree-data
git clone https://github.com/heartpain-kong/unitree_rl_mjlab_lgdog.git
cd unitree_rl_mjlab_lgdog
bash install_ubuntu.sh --env-dir /root/gpufree-data/lgdog-venv
source /root/gpufree-data/lgdog-venv/bin/activate
python scripts/verify_install.py
```

平台镜像选择开发类 Ubuntu 22.04 镜像，并按显卡驱动选择 CUDA 版本。训练目录、虚拟环境、pip 缓存和 `logs/` 都放在 `/root/gpufree-data`；平台内禁止 Docker，直接使用上述虚拟环境即可。

### 常用参数

```bash
# 已有系统依赖或没有 sudo 权限
bash install_ubuntu.sh --skip-system-deps

# 把环境放到数据盘
bash install_ubuntu.sh --env-dir /data/lgdog-venv

# 指定 PyTorch CUDA wheel 源
bash install_ubuntu.sh --torch-index https://download.pytorch.org/whl/cu128

# 没有 GPU，只做 CPU 导入和代码检查
bash install_ubuntu.sh --cpu
```

如果平台使用非默认 Python，可显式指定：

```bash
bash install_ubuntu.sh --python /usr/bin/python3.11
```

## 安装后运行

```bash
source .venv/bin/activate
python lgdog.py                         # 浏览器工作台
python lgdog.py tasks
python scripts/list_envs.py --keyword LgDog1
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard
```

也可以直接使用包装脚本：

```bash
bash run_lgdog.sh tasks
```

显存不足时降低 `--env.scene.num-envs`。训练日志和模型默认写入 `logs/`，建议把该目录放在平台的数据盘或持久化卷中。

## 手工安装

一键脚本不可用时，按下面步骤执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
  python3-venv python3-dev libegl1 libgl1 libglib2.0-0 \
  libx11-6 libxext6 libxrender1 libyaml-cpp-dev libboost-all-dev \
  libeigen3-dev libspdlog-dev libfmt-dev

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch>=2.7.0,<2.8.0'
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements-ubuntu.txt
python -m pip install --no-deps -e .
python scripts/verify_install.py
```

版本锁定集中在 [requirements-ubuntu.txt](../requirements-ubuntu.txt)。不要只安装 `mjlab` 而让 pip 自由选择 MuJoCo 和 Warp 版本，否则可能出现 API 不兼容。
