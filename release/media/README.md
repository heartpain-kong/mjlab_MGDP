# 媒体：回放视频与汇总图

本目录是 `release/policy/model_12400.pt` 的**实测回放素材**，全部由仓库脚本真实推理生成，无插帧、无瞬移、无合成成功。

| 文件 | 说明 |
|---|---|
| `preview_ten_terrain.png` | 十类地形单次回放总览（每格：HUD + 第 1 视角缩略） |
| `train_curve.png` | 第 12400 轮训练过程曲线 |
| `level_distribution.png` | 训练环境在各课目/各难度档的分布 |
| `videos/01_start_flat_cmd0p6mps_front.mp4` … `10_slalom_brick...` | 十段逐课目回放，前视角，0.6 m/s 指令 |
| `model_12400_十类地形_0p6mps_同步前视.mp4` | 十段合成的整片（约 2 分钟） |
| `results.json` | 每段实测数据：用时、平均速度、路线进度、终止原因 |
| `视频实测说明.md` | 生成这批视频时的环境/参数记录 |

## 视频是怎么导出的

### 方式一：一键录制十段（与 `release/media` 完全一致）

```bash
git checkout policy-v1          # 必须：与策略匹配的代码
python -m pip install -e .
python -m pip install imageio-ffmpeg Pillow

python scripts/record_ten_terrain.py \
  --checkpoint release/policy/model_12400.pt \
  --perception release/policy/perception_3cca6497b8e9f392ad5e47e61ea1a6e03e324d80979d8cca1a4518ad79e46e23.pt \
  --output out/videos --max-seconds 20 --speed 0.6
```

脚本做的事：

1. 对 10 个课目逐个建环境（单环境、播放模式）；
2. 用真实策略推理推进，25 FPS 录制（50 Hz 控制器每两步采样一次，1 倍速）；
3. 画面叠加 HUD：课目名、档位、指令速度、实时平面速度、路线进度条；
4. 写出 `out/videos/*.mp4`、每段 `results.json`，并用 ffmpeg 合成整片。

### 方式二：官方 `play.py` 录制单段

```bash
python scripts/play.py LgDog1-Race-MGDP-WallTraining-D435i20 \
  --checkpoint-file release/policy/model_12400.pt \
  --perception-checkpoint release/policy/perception_3cca*.pt \
  --video --video-length 1200 --video-height 720 --video-width 1280
```

视频写到 `logs/.../videos/play/`（`VideoRecorder` 的默认目录）。

## 合成整片（ffmpeg）

十段录完后按顺序拼接（`concat.txt` 每行 `file 'xxx.mp4'`）：

```bash
cd out/videos
ls 0*_cmd0p6mps_front.mp4 | sort | sed "s/^/file '/;s/$/'/" > concat.txt
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy model_12400_十类地形_0p6mps_同步前视.mp4
```

## 说明

- 视频是**仿真**画面（MuJoCo 渲染），不是实机；
- 高墙那段是已知失败案例（墙前停滞），保留它是为了如实展示短板；
- 文件名里的 `cmd0p6mps` 表示指令速度 0.6 m/s。
