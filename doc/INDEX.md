# 文档索引

日常操作从根目录运行 `python lgdog.py`。根目录说明见 [README](../README.md)。

| 想做什么 | 文档 |
|---|---|
| MuJoCo 模型查看、策略回放、比赛场地与键盘控制 | [MUJOCO_USAGE.md](MUJOCO_USAGE.md) |
| 使用浏览器面板、短命令、了解目录迁移 | [WORKBENCH.md](WORKBENCH.md) |
| MGDP 定向越障、reward、通过率门控与接触规则 | [MGDP_PASSAGE.md](MGDP_PASSAGE.md) |
| MGDP 深度感知迁移、预训练与训练 | [MGDP_MJLAB.md](MGDP_MJLAB.md) |
| 一个网页看全部：环境 / GPU / 进程 / 轮次 / 日志（总览） | [WEBHUB.md](WEBHUB.md) |
| 把训练曲线当 K 线看（曲线终端） | [KLINE.md](KLINE.md) |
| 2027 比赛场地：训练地形 + sim2sim 真场地 | [RACE_COURSE.md](RACE_COURSE.md) |
| LP-ACRL地形自动采样、参数与对照实验 | [LP_ACRL_INTEGRATION.md](LP_ACRL_INTEGRATION.md) |
| HIM三帧历史、训练开销与旧模型兼容 | [HIM_HISTORY_3.md](HIM_HISTORY_3.md) |
| 抬腿异常、连续摆动净空与LP参数 | [CLEARANCE_HEIGHT_LP_TUNING.md](CLEARANCE_HEIGHT_LP_TUNING.md) |
| 速度扩张条件、进度门与旧 checkpoint 迁移 | [VELOCITY_CURRICULUM.md](VELOCITY_CURRICULUM.md) |
| V23地形为0：进度门与扩速判据诊断 | [TRAINING_DATA_ANALYSIS_V23_2026-10-27.md](TRAINING_DATA_ANALYSIS_V23_2026-10-27.md) |
| 调整奖励、查看最终权重和函数位置 | [REWARDS.md](REWARDS.md) |
| V21/V19 行走退化、奖励方向与速度课程门限 | [TRAINING_DATA_ANALYSIS_2026-10-27.md](TRAINING_DATA_ANALYSIS_2026-10-27.md) |
| V2/V8 训练 run、课程和动作数据分析 | [TRAINING_DATA_ANALYSIS_2026-10-20.md](TRAINING_DATA_ANALYSIS_2026-10-20.md) |
| V2 日志分析与 V3 文献调参 | [REWARD_LITERATURE_V3.md](REWARD_LITERATURE_V3.md) |
| 标准 Rough + HIM 新任务 | [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md) |
| 原 V7 温和地形 HIM 方法与实现 | [LGDOG_HIM.md](LGDOG_HIM.md) |
| LgDog 训练、导出与 sim2sim | [LGDOG_GUIDE.md](LGDOG_GUIDE.md) |
| 项目源码结构 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 详细命令速查 | [commands.md](commands.md) |
| V1–V7b 历史训练记录 | [TRAINING_V.md](TRAINING_V.md) |
| TerrainV4 分析 | [LGDOG_TERRAIN_V4.md](LGDOG_TERRAIN_V4.md) |
| 安装训练依赖 | [setup_zh.md](setup_zh.md) |
| GPUFree Ubuntu 22.04 部署、CUDA 修复、SSH 转发和显存经验 | [DEPLOYMENT_GPUFREE.md](DEPLOYMENT_GPUFREE.md) |
| 服务器一键安装、自检、后台网页和Windows上传 | [SERVER_DEPLOY.md](../SERVER_DEPLOY.md) |
| 原项目说明 | [中文](UPSTREAM_README_zh.md) / [English](UPSTREAM_README.md) |

`archive/` 内为历史补丁和验证记录，当前功能请同步完整工程。模型与训练快照仍在 `logs/rsl_rl/`。旧记录里提到的 Linux 机器外部 `DEPLOY_NOTES.md`、`export/` 不属于这份工程的随附文件。
