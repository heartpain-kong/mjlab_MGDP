# D435i 下俯20°配置

工作台新增独立入口：**MGDP · 比赛尺寸强化课程 · D435i下俯20°**。
任务ID：`LgDog1-Race-MGDP-WallTraining-D435i20`；别名：`race_mgdp_wall_d435i20`。
原相机任务保留，已有策略请继续选择原入口及检查点配套的感知文件。

| 项目 | 新入口 |
|---|---|
| 安装角 | 相对机身水平向下20°，随机身姿态运动 |
| 安装位置 | 机身坐标系前向0.02m、横向0、向上0.075m |
| 深度视场 | 水平87°、垂直58° |
| 机身水平时垂直覆盖 | 上边界高于水平9°，下边界低于水平49°；近处仍有盲区 |
| 仿真采样 | 32×32射线，2×2最近有效深度池化至16×16 |
| 深度 | 光轴Z深度0.2–3m，归一化除以3；无效点按远处处理 |
| 感知文件 | `logs/mgdp/perception/mgdp_wall_d435i20.pt` |
| 策略实验目录 | `logs/rsl_rl/lgdog1_velocity_race_mgdp_wall_d435i20_v6` |

这里配置的是仿真相机和模型输入契约。848×480是RGB调试视图的宽高设置，不是每环境渲染848×480深度图。射线按控制步更新，当前控制频率50Hz；没有接入实物D435i驱动，也没有模拟30/60FPS异步采样和完整相机延迟。安装位置沿用原机身挂点，不把示意图中的0.38m写死为世界高度。实物部署仍应测量外参、读取内参并验证实际深度流。

预训练、新建策略、完整续训、Play和Sim2Sim共用该配置。感知文件和策略检查点记录相机位置、四元数、视场、深度处理方式；加载时校验，防止外观相同的16×16张量掩盖相机变化。ONNX导出也携带相机元数据。摄像头小窗中的RGB视角跟随同一个下俯光轴，RGB投影视场与射线深度图不保证逐像素一致。

该入口沿用八档比赛尺寸强化课程及高墙奖励，与原强化课程共用高墙规则配置文件。改变这些奖励会影响两个入口之后启动的任务。新相机须重新预训练感知并新建策略，不直接替换旧策略正在使用的感知权重。

## 本地Linux操作

把累计补丁及同名`.sha256`文件放到`/home/heartpain/mjlab`，执行：

```bash
cd /home/heartpain/mjlab
sha256sum -c mgdp_d435i_down20_20261115.zip.sha256 && unzip -o mgdp_d435i_down20_20261115.zip -d /home/heartpain/mjlab
```

重启工作台Python后端，然后选择新增入口，先点预训练，成功后新建策略训练。仅刷新网页不能重载Python配置。

CLI等价预训练命令：

```bash
cd /home/heartpain/mjlab/unitree_rl_mjlab_lgdog
conda activate unitree_rl_mjlab
python -m scripts.pretrain_mgdp --task LgDog1-Race-MGDP-WallTraining-D435i20 --num-envs 64 --terrain-rows 8 --terrain-cols 10 --batch-size 64 --steps 4000 --iterations 3000
```

该任务自动选择`d435i_edge16_down20`并保存到独立感知文件。正式训练环境数请按显存和速度实测选择；此变更不代表高墙通过率已经提高。

## 配置位置与验证范围

- `src/config/mgdp_camera.py`：新任务、文件名、机身挂点及旋转计算。
- `src/mgdp_depth.py`：视场、深度处理、32×32采样及相机契约。
- `src/tasks/velocity/config/lgdog1/mgdp_env_cfg.py`：实际深度挂点与RGB相机下俯；新任务沿用原八档强化课程。
- `src/dashboard/catalog.py`：工作台入口、预训练命令和独立实验目录。

CPU短流程使用10环境、8×10地图、高墙单地形夹具，采集12步，完成1次感知优化、1轮PPO和1轮完整续训，并检查ONNX相机元数据和JIT文件。另以真实MuJoCo射线检查首次恢复挂点后的深度，以实际离屏渲染检查256×145的摄像头小窗。以上检查验证接线与恢复流程，不是正式感知训练或比赛赛道验收；GPU、实物与收敛效果尚未验证。分组测试与短流程记录见`MGDP_D435I_DOWN20_VALIDATION.json`。
