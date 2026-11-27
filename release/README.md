# 开源策略：LgDog1 MGDP 比赛课程策略（第 12400 轮）

本目录附带一份**可直接回放**的深度感知策略与它配套的感知权重。

| 文件 | SHA-256 | 大小 |
|---|---|---|
| `policy/model_12400.pt` | `4d5556fe80f7d135c43dae09ecf7f8777a64c18573fcca24b24a0de9f3b92745` | 7.2 MB |
| `policy/perception_3cca....pt` | `94d610032715b4f60d1e973e4d50470eba8f3973e81ab4f473ff4d3700001c41` | 7.1 MB |

## 这份策略是什么

| 项 | 值 |
|---|---|
| 任务 | `LgDog1-Race-MGDP-WallTraining-D435i20`（2027 比赛场景，深度感知） |
| 训练代数 | 第 **12400** 轮 |
| 技能引导 | V5（渐进短课程 + 最终参数 + 阶段诊断） |
| 速度 | 0.6 m/s 巡航 |
| 观测 | 本体 4 帧 × 47 维（188）+ 深度特征 64 维 = **252 维** |
| 深度输入 | **16×16**（射线 16×16，FOV 87°，量程 5 m，相机挂载 (0.02, 0, 0.075)、下俯 20°） |
| 动作 | 12 维关节位置偏移，scale 0.5 |
| 合同快照 | `mgdp_skill_training = {'version': 5}`、`mgdp_course_contract = {'action_scale': 0.5, 'stone_min_feet': 2}` |

## ⚠️ 必须用对应代码版本运行

主干（`main`）已经把深度改成 **32×24** + stride 卷积编码器、地形与奖励也做了多轮修改，**与这份策略不兼容**。

为此仓库打了标签 **`policy-v1`**（提交 `46d4d0b`），它就是这份策略训练/回放时代的代码：

```bash
git checkout policy-v1
```

## 回放

```bash
# 1. 切到匹配的代码版本
git checkout policy-v1
python -m pip install -e .

# 2. 复制策略文件到仓库外任意位置（示例用 /tmp/lgdog_policy）
cp -r <本目录>/policy /tmp/lgdog_policy

# 3. 回放（十类地形各跑一次，自动录视频）
python scripts/play.py LgDog1-Race-MGDP-WallTraining-D435i20 \
  --checkpoint-file /tmp/lgdog_policy/model_12400.pt \
  --perception-checkpoint /tmp/lgdog_policy/perception_3cca6497b8e9f392ad5e47e61ea1a6e03e324d80979d8cca1a4518ad79e46e23.pt
```

也可以直接跑 `bash release/play_policy.sh /tmp/lgdog_policy`。

## 已测到的效果（0.6 m/s 巡航）

十类地形单次回放（详见 `eval_results.md`）：

| 地形 | 结果 | 用时 |
|---|---|---|
| 平地 | ✅ | 8.0 s |
| 梅花桩 | ✅ | 9.4 s |
| 单木桥 | ✅ | 8.4 s |
| 沟壑台阶 | ✅ | 16.7 s |
| 高墙 | ❌ 8 秒无新增进展截断 | 10.6 s |
| 限高笼 | ✅ | 9.6 s |
| 轮胎阵 | ✅ | 8.5 s |
| 随机起伏 | ✅ | 8.5 s |
| 随机粗糙 | ✅ | 9.2 s |
| 绕桩接砖木坑 | ✅ | 8.4 s |

合计 **9/10**。比赛中（第 6 档完整赛道）实测 **5/10**，未过的课目是沟壑台阶、高墙、限高笼、轮胎阵与绕桩超时——这份策略的短板就是"高墙"和"绕桩"。

## 说明

- 训练数据全部来自仿真；**未在实机验证**。
- 高墙课目 0% 成功率是已知问题，主干为此加了 `wall_climb` 三段式奖励、`progress_curriculum` 相对进展课程、教师-学生蒸馏脚本，见仓库 `doc/`。
- 深度编码器的输入是 16×16，与主干当前的 32×24 不同，**不要混用**。

## 媒体

本目录下的 `media/` 包含这批策略的真实回放素材（约 22 MB）：

| 文件 | 说明 |
|---|---|
| `media/preview_ten_terrain.png` | 十类地形回放总览 |
| `media/videos/*.mp4` | 十段逐课目前视角回放（0.6 m/s） |
| `media/model_12400_十类地形_0p6mps_同步前视.mp4` | 十段合成整片 |
| `media/train_curve.png`、`media/level_distribution.png` | 训练曲线与环境分布 |
| `media/results.json` | 每段实测用时/速度/进度/终止原因 |

导出方法见 `media/README.md`，一键脚本是 `scripts/record_ten_terrain.py`：

```bash
git checkout policy-v1
python scripts/record_ten_terrain.py \
  --checkpoint release/policy/model_12400.pt \
  --perception release/policy/perception_3cca6497b8e9f392ad5e47e61ea1a6e03e324d80979d8cca1a4518ad79e46e23.pt \
  --output out/videos --max-seconds 20 --speed 0.6
```
