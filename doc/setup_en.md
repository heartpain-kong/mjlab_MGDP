# Ubuntu Installation and Deployment

## Recommended: one-command installation

This flow targets Ubuntu 22.04/24.04, x86_64, and NVIDIA GPU rental instances. Upload or clone the repository, then run from its root:

```bash
bash install_ubuntu.sh
source .venv/bin/activate
python scripts/verify_install.py
```

The installer installs system build and MuJoCo runtime libraries, creates an isolated `.venv`, installs the pinned PyTorch/mjlab/MuJoCo/Warp stack, installs this checkout in editable mode, and verifies LgDog task registration. Training requires an NVIDIA driver; the final check only imports the runtime and does not start a long training run.

For GPUFree instances, keep the checkout and environment on the persistent data disk:

```bash
nvidia-smi
cd /root/gpufree-data
git clone https://github.com/heartpain-kong/unitree_rl_mjlab_lgdog.git
cd unitree_rl_mjlab_lgdog
bash install_ubuntu.sh --env-dir /root/gpufree-data/lgdog-venv
source /root/gpufree-data/lgdog-venv/bin/activate
python scripts/verify_install.py
```

Choose a development Ubuntu 22.04 image whose CUDA version matches the instance driver. GPUFree instances do not provide Docker; use the virtual environment directly.

Useful options:

```bash
bash install_ubuntu.sh --skip-system-deps
bash install_ubuntu.sh --env-dir /data/lgdog-venv
bash install_ubuntu.sh --torch-index https://download.pytorch.org/whl/cu128
bash install_ubuntu.sh --cpu
bash install_ubuntu.sh --python /usr/bin/python3.11
```

## Run after installation

```bash
source .venv/bin/activate
python lgdog.py
python lgdog.py tasks
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard
```

`bash run_lgdog.sh tasks` is a wrapper that uses `.venv/bin/python` without requiring activation. Store `logs/` on the rental platform's persistent data volume if the instance is ephemeral.

## Manual installation

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

The compatibility pins live in [requirements-ubuntu.txt](../requirements-ubuntu.txt). Keep the MuJoCo and Warp versions together with this code.
