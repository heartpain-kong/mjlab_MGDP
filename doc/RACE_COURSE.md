# 2027 比赛场地：训练与 sim2sim 接入

**当前 MGDP 训练主线为 v3**：按用户提供场地图修复双排六桩、实心坡台沟壑、L形矮桩/砖木坑、
连续限高框架和轮胎阵，并使用分段路线。见 [训练说明](MGDP_PASSAGE.md) 与
[前后对照及验证](analysis/terrain_2027_v3/report.md)。
训练实现入口为 `passage_terrain_cfg.py`；下文 HIM 和早期深度训练配置是历史/近似方案，
不应当作当前 `race_mgdp` 的地图定义。

场地来源：[`27rc-dog-race`](https://github.com/uwvwko-zzz/27rc-dog-race) —— 依据《2027 年足式机器人挑战赛比赛规则》
建模的 MJCF 场地（障碍赛按 **V2.0**，任务赛按 **V1.0**）。它本身是「规则尺寸的训练/验证用实现」，
障碍摆位以赛事公布为准。

这份工程做了两件事：

1. **sim2sim 直接跑真场地**：`--map race` / `--map race_task`，几何和比赛一模一样。
2. **训练用规则尺寸的近似地形**：新任务 `LgDog1-Race-Blind-HIM`，把能落到高度场的障碍
   （梅花桩、单木桥、沟壑台阶、砖木坑、绕桩、高墙）按规则尺寸拼成课程地形。

```bash
# 真场地：站在启动区，0.4 m/s 往前走
python scripts/sim2sim.py LgDog1-Race-Blind-HIM --onnx=<policy.onnx> --map race --vx 0.4 --duration 5

# 训练：比赛地形课程（观测/奖励与标准 HIM 相同）
python lgdog.py train race --envs 4096 --iterations 16000
```

## 文件

| 文件 | 说明 |
|---|---|
| `scripts/import_race_course.py` | 把 `27rc-dog-race` 的 `maps/*.xml` 转成工程地形；规则更新时重跑 |
| `src/assets/terrains/race_obstacle_2027.xml` | 障碍赛场地（984 geom，14×6 m） |
| `src/assets/terrains/race_task_2027.xml` | 任务赛内场（105 geom，含 8 个动态物资箱） |
| `src/assets/terrains/race_task_labels/*.png` | 物资箱标识（食品/工具/仪器/药品） |
| `src/tasks/velocity/config/lgdog1/race_terrain_cfg.py` | 训练地形：按规则尺寸拼的 8 种子地形 |
| `src/tasks/velocity/config/lgdog1/him_env_cfg.py` → `lgdog1_race_env_cfg()` | 新任务的环境配置 |
| `tests/test_race_course.py` | 地形比例、任务注册、场地资产的回归测试 |

## 为什么不能直接把比赛 XML 当 `--map-file`

1. **角度单位**：场地是 `<compiler angle="degree">`，工程是弧度。`MjSpec.attach` 会按目标单位解释拷过来的
   `euler`，844 个 euler 会全部错位。转换脚本先用 MuJoCo 加载再 `mj_saveLastXML`，把姿态烘成 `quat`、
   compiler 统一成 `radian`。
2. **`<default>` 会污染机器人**：场地 default 里有 `friction=0.9 / solref / solimp`，整块 attach 会把机器人
   geom 的默认值一起改掉。转换时把这些值逐 geom 烘进去，然后删掉 `default`。
3. **`<option>`/灯光/相机会混进场景**：训练/回放用的是任务自己的 `SimulationCfg`，同样一并删除。

转换后的文件可以直接 attach，也可以单独用 MuJoCo 打开预览。

## 场地 ↔ 训练地形对照

| 比赛障碍 | 规则尺寸 | 训练地形用什么 | 覆盖 |
|---|---|---|---|
| 梅花桩 `stepping_stones` | 0.2×0.2×0.1，净距 0.2 / 0.4，两端 0.8×0.8 平台 | `BoxSteppingStonesTerrainCfg`（同尺寸） | ✅ |
| 单木桥 `balance_beam` | 3×0.4×0.1 | `BoxNarrowBeamsTerrainCfg`（宽 0.4、高 0.1） | ✅ |
| 沟壑组合 `gap_assembly` | 平台 0.8×0.8、台阶 0.12 高 / 0.32 深、20° 坡 | `BoxOpenStairsTerrainCfg`（0.10–0.14 台阶） | ✅（坡用台阶近似） |
| 砖木坑 `brick_bark_pit` | L 形 2×2，碎砖 10–30 mm | `BoxRandomSpreadTerrainCfg`（0.1–0.2 m 碎块，0–6 cm） | ✅（近似） |
| 绕桩 `slalom` | 桩径 0.1、高 0.1、中心距 0.9 | `BoxRandomSpreadTerrainCfg` 细高柱 | ✅（近似，不带必达区） |
| 高墙 `high_wall` | 厚 0.08、高 0.3 | `BoxRandomSpreadTerrainCfg` 薄板 | ✅ |
| 轮胎阵 `tire_array` | 2×4 轮胎，外径 0.55 | 无（不是高度场） | ⚠️ 只在 sim2sim |
| 限高框架 `clearance_frame` | 横梁下表面 0.29 | 无（头顶障碍） | ⚠️ 只在 sim2sim |
| 启动区 | 0.8×0.8 平地 | `BoxFlatTerrainCfg` | ✅ |

轮胎和限高框架是「头顶/单独刚体」类障碍，**HIM 的高度场地形表达不了** —— 它们在 `--map race`
里是真几何，HIM 训练里只能靠 `foot_clearance` 类奖励和 sim2sim 复测兜底；
深度感知的 `race_mgdp` 任务则用 `race_depth_terrain_cfg.py` 把这两类障碍按比赛尺寸真正建进了训练地形。

## 场地与坐标（来自建模说明）

| 场地 | X 范围 | Y 范围 | 出生点 |
|---|---|---|---|
| 障碍赛 | [-7, 7] | [-3, 3] | 启动区 1：`(0, 0, 0.4)`，朝 +X |
| 障碍赛启动区 2 | 同上 | 同上 | `(-5.95, -1.9, 0.76)`（平台顶 0.36） |
| 任务赛内场 | [-2, 2] | [-3, 3] | 启动区内 `(0, -2.6, 0.31)`，朝 +Y |

> 任务赛的南侧启动垫只有 1.0×0.8 m，在 `(0, -3.4)` 出生时机器人会压到垫子边缘、姿态歪掉；
> 实测 `(0, -2.6)` 出生最稳，需要从门外进场的可以自己调 `--spawn-y`。
> 任务赛目前只做加载与行走验证，**没有判分**（搬箱、归位、挡车杆计数都未实现）。

## 换成下一版规则

1. 拿到新的场地仓库/压缩包，覆盖原目录；
2. 重跑转换：
   ```bash
   python scripts/import_race_course.py /path/to/27rc-dog-race-main
   ```
3. 如果障碍尺寸变了，同步改 `race_terrain_cfg.py` 里对应拼块的参数
   （文件头有「规则尺寸 → 拼块参数」的对照表），跑 `python -m unittest tests.test_race_course`。

## 深度感知版本：`LgDog1-Race-MGDP`

高度扫描表达不了头顶横梁和轮胎阵，但深度相机可以，所以另开了一份地形
`race_depth_terrain_cfg.py`（只在深度任务里用，HIM 的 `race` 任务不变），
两类障碍都按比赛场地的真实尺寸建：

| 新增障碍 | 几何（与比赛一致） | 说明 |
|---|---|---|
| 限高框架 `clearance_frame` | 0.8×0.8×0.3 m 笼子：6 根立柱 + 3 根横梁 + 2 根纵梁；每格 **2 个**，梁底 0.34→0.28 m | 机器人需要低头/趴下通过 |
| 轮胎阵 `tire_array` | **2×4 共 8 个**平放轮胎，中心距 0.55 m，外径 0.549 m、内径 0.329 m、高 0.165 m | 与障碍赛 `tire_array` 排布一致 |

- 相机：前头顶 `(0.02, 0, 0.075)`，光轴水平朝前，FOV 87°、16×16 射线；
- 任务：`python lgdog.py train race_mgdp --envs 4096 --iterations 16000`
  （需要先跑 `scripts/pretrain_mgdp.py`）；
- 真场地复测仍用 `--map race`，几何与比赛一致。

## 许可与来源

- 场地几何、物资箱标识（食品/工具/仪器/药品）来自 `27rc-dog-race`，依据比赛规则 PDF 建模；
  规则原文与赛事标识属于第三方素材。
- 该仓库未指定开源许可证；本工程只在本地训练/验证中使用，不随发布物分发规则原文。
- 训练/仿真代码与机器人模型仍是本工程（Unitree RL mjlab，Apache-2.0）的部分。
