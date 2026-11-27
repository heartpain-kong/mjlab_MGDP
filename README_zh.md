# LgDog 使用入口

## Ubuntu 算力平台一键部署

```bash
cd unitree_rl_mjlab_lgdog
bash install_ubuntu.sh
source .venv/bin/activate
python scripts/verify_install.py
```

安装脚本会自动处理 Ubuntu 系统库、独立 Python 环境、PyTorch、锁定的 mjlab/MuJoCo/Warp 依赖和当前源码安装。无 sudo 但镜像已有依赖时，加 `--skip-system-deps`；只做 CPU 导入检查时，加 `--cpu`。完整参数见 [安装说明](doc/setup_zh.md)，GPUFree 的实测部署记录见 [DEPLOYMENT_GPUFREE.md](doc/DEPLOYMENT_GPUFREE.md)。

算力自由平台请把仓库和环境放在 `/root/gpufree-data` 数据盘：

```bash
cd /root/gpufree-data
git clone https://github.com/heartpain-kong/unitree_rl_mjlab_lgdog.git
cd unitree_rl_mjlab_lgdog
bash install_ubuntu.sh --env-dir /root/gpufree-data/lgdog-venv
source /root/gpufree-data/lgdog-venv/bin/activate
```

## 常用入口

```bash
python tools/lgdog.py                 # 浏览器工作台
python tools/lgdog.py tasks            # 列出 LgDog 任务
python scripts/train.py LgDog1-Rough-Blind-HIM-Standard --env.scene.num-envs=1024
```

- [面板使用说明](doc/WORKBENCH.md)
- [奖励调参指南](doc/REWARDS.md)
- [标准地形 HIM](doc/LGDOG_HIM_STANDARD.md)
- [原项目中文 README](doc/UPSTREAM_README_zh.md)
