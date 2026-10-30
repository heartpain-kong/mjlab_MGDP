# LgDog 服务器一键部署

当前导出包含本地未提交的新代码，不依赖GitHub旧版本：HIM默认3帧/141维、LP稳定版v2、连续摆动净空函数、工作台、曲线终端、总览。`base_height`已撤回，抬脚权重默认仍0。代码包不含训练数据、模型、虚拟环境、Git私有仓库元数据或本机密钥。

## 服务器上运行

适用于Ubuntu 22.04/24.04 NVIDIA GPU镜像，服务器需要能访问PyPI、NVIDIA和PyTorch下载源。不使用Docker，也不需要把Windows环境复制过去。

将 `unitree_rl_mjlab_lgdog_server.zip` 上传到数据盘，在服务器终端执行：

```bash
cd /root/gpufree-data
python3 -m zipfile -e unitree_rl_mjlab_lgdog_server.zip .
cd unitree_rl_mjlab_lgdog
bash deploy_server.sh
```

首次解压请选没有同名工程的目录。已有服务器工程需要更新时，先停止其网页服务并备份本地reward/地形覆盖；推荐上传到新的release目录，保留旧源码与日志。下方PowerShell上传脚本自动创建独立release，不覆盖旧目录。

一条部署命令会：

1. 校验导出文件哈希（存在SOURCE_MANIFEST.json时），准备Linux脚本执行权限。
2. 创建或复用 `/root/gpufree-data/lgdog-venv`，pip缓存也优先放数据盘；没有该数据盘时使用工程 `.venv`。
3. 安装固定兼容组合：mjlab1.2.0、Torch2.7.x、MuJoCo/MuJoCo-Warp3.5.0、Warp1.12.0、RSL-RL5.0.1，以及工作台/导出依赖。默认Torch CUDA12.8索引，其他驱动镜像可显式选择cu126。
4. 在子进程环境中查找真实 `libcuda.so.1`，优先真实驱动目录，过滤CUDA stubs路径。不会修改系统库或用户.bashrc；特殊平台可以设置 `LGDOG_CUDA_DRIVER=/真实路径/libcuda.so.1`。
5. 验证任务注册、3帧配置、PyTorch CUDA分配、Warp CUDA分配。失败就停止，不把CPU导入通过当作GPU部署成功。
6. 后台启动工作台、曲线终端、TensorBoard和总览，均监听127.0.0.1，不依赖DISPLAY。默认不启动训练。

安装版本来源：[mjlab1.2.0包](https://pypi.org/project/mjlab/1.2.0/)、[PyTorch2.7官方wheel说明](https://pytorch.org/get-started/previous-versions/#v2-7-1)。固定版本仍需在实际服务器验证驱动兼容性。

## 本机一条命令上传并部署

Windows PowerShell中，在导出目录执行以下命令；IP和SSH端口使用租借平台当前给你的值，不要照搬历史实例地址：

```powershell
cd E:\算力自由
.\上传到服务器.ps1 -ServerHost 你的服务器IP -Port 你的SSH端口
```

脚本使用系统OpenSSH，不保存账号密码；可使用机器已有SSH key或SSH提示输入密码。它上传压缩包到 `/root/gpufree-data/lgdog-releases/时间戳_随机后缀/`，检查压缩包SHA256，解压后运行部署。可用 `-UserName 用户名 -RemoteBase /你的数据盘目录` 更换位置。平台已预装系统库时可加 `-SkipSystemDeps`；显式加 `-Smoke` 才会执行下方小训练。

如果原服务占用默认端口，部署会报错并保留原服务；不会主动杀掉旧实例或训练。安装阶段可能复用数据盘venv，因此涉及变更依赖时仍应选择空闲维护时段，或用独立 `--env-dir`。

## 打开服务器网页

部署完成后，在本机另开PowerShell窗口，保持SSH隧道运行：

```powershell
ssh -N -p 你的SSH端口 `
  -L 8765:127.0.0.1:8765 `
  -L 8766:127.0.0.1:8766 `
  -L 8770:127.0.0.1:8770 `
  -L 6006:127.0.0.1:6006 root@你的服务器IP
```

| 页面 | 本机地址 |
|---|---|
| 训练工作台 | http://127.0.0.1:8765/ |
| 曲线终端 | http://127.0.0.1:8766/ |
| 总览 | http://127.0.0.1:8770/ |
| TensorBoard | http://127.0.0.1:6006/ |

工作台使用Host/Origin校验，隧道本机端口与服务器端口应保持相同。本机端口有冲突时，部署时一起换服务器端口，再使用对应同号隧道，例如：

```bash
bash deploy_server.sh --ui-port 8865 --kline-port 8866 --portal-port 8870 --tb-port 6106
```

## 启停、日志和重复部署

服务器工程根目录：

```bash
bash run_lgdog.sh server status
bash run_lgdog.sh server stop
bash run_lgdog.sh server start
```

自定义端口的再次start需要传入同样四个端口参数。status/stop根据PID记录读取实际端口。停止只操作本工程记录且PID、启动时间、命令行、工作目录一致的网页进程，不递归停止训练任务；训练请在工作台中单独结束。关闭SSH连接后后台网页进程继续运行；实例/容器重启后需再次start，不是系统开机自启。

部署输出保存在 `logs/server/deploy.log`，各网页服务日志为同目录下的dashboard.log、kline.log、tensorboard.log、portal.log，PID记录是services.json，安装依赖快照是installed-packages.txt。原有logs和checkpoint不会被安装或start/stop删除。

同一目录重复start会复用已运行服务；更新源码后要stop再start才能让Python进程加载新版本。只复用已安装环境可用 `bash deploy_server.sh --skip-install`，仍执行GPU检查。`--no-start`仅安装和检查，不启动网页。

## 小检查与正式训练

显式运行链路小检查：

```bash
bash deploy_server.sh --skip-install --smoke
```

只跑64环境、2轮更新，保存到独立的 `logs/rsl_rl/lgdog_deploy_smoke/`，避免被标准任务误当作最新训练模型。小检查权重不具有行走能力，不能部署到实机。首次Warp编译可能较久。

通过后再手动开正式训练，例如：

```bash
bash run_lgdog.sh train standard --envs 1024 --iterations 16000 \
  --terrain-curriculum lp_acrl --velocity-min-abs 0.3 --velocity-max-abs 0.5
```

环境数需要根据实际显存和地形配置逐档验证；不要自动照搬历史4090或本地游戏本上限。新训练是3帧，不能直接resume旧6帧checkpoint。抬脚参数见 `doc/CLEARANCE_HEIGHT_LP_TUNING.md`，默认权重0，是否开启由你在工作台选择。

## 常见问题

- Warp驱动错误：查看 `LGDOG_CUDA_DRIVER`、`LD_LIBRARY_PATH` 和部署日志；nvidia-smi成功不等于Warp成功。直接执行 `bash run_lgdog.sh tasks` 只验证轻量入口，完整检查使用虚拟环境Python运行 `scripts/verify_install.py --require-cuda`，并先source server_env.sh。
- 系统依赖已装但apt不可用：`--skip-system-deps`；没有虚拟环境模块时仍需安装python3-venv。
- 镜像驱动不支持默认CUDA wheel：使用 `--torch-index https://download.pytorch.org/whl/cu126`，以实际GPU检查为准；不要删除平台真实驱动。
- 端口被旧面板占用：在旧工程中停止其网页服务，或选择一组新端口；部署不会使用pkill扫杀进程。
- 没有图形桌面：用工作台/Viser或无viewer的sim2sim；不要在没有DISPLAY的SSH会话里开启原生GLFW窗口。

本机已验证导出哈希/压缩包、脚本语法及CPU服务控制；未连接实际服务器执行apt/CUDA训练。以服务器部署日志和GPU检查结果为最终验收。
