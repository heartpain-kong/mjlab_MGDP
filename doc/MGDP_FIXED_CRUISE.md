# MGDP固定巡航速度

> 2026-11-16 更新：比赛尺寸强化课程（含D435i20）的新训练默认从0.6 m/s渐进扩速，目标上限2.0 m/s；可在工作台切回固定巡航。下文历史说明中的0.25～0.35和D435i20默认固定1.0，适用于当时的旧版本，不是新默认值。旧模型回放仍按检查点恢复原设置。

原工作台的两个正负速度字段填1.0，含义是通用任务的−1～+1m/s速度包络。定向越障任务禁用这些字段，原CLI默认仍是0.25～0.35m/s起步、通过率达标后逐渐提高上限。因此旧命令没有1.0，`Curriculum/passage/speed_max`一直为0.35，符合旧实现，并未启动固定1m/s训练。

## 工作台

更新工程、重启Python工作台后端并刷新网页后，定向越障任务显示独立的“定向越障速度模式”和“正向巡航目标”。普通任务继续显示正负速度包络，标签明确说明−1～+1不代表固定正向1。

- “固定巡航速度”：以填写速度作为每回合沿路线的巡航目标，关闭课程自动扩速和降速；地形仍按通过情况及跟踪误差升级、失败时降级。
- “通过率渐进扩速（原方案）”：保持旧逻辑，初始0.25～0.35m/s，达标后提高上限。
- D435i下俯20°入口的新建训练表单默认固定1.0m/s；其他定向任务保留原模式默认。CLI不提供新参数时也保留原行为。

选择固定1.0后，命令预览必须包含：

```text
--passage-fixed-speed=1.0
```

巡航速度是沿路线的目标速度大小，不等同于机身X方向分量或实测速度。靠近路点会减速；绕桩转向准备时保留停车对齐。高墙直线段远离终点时，固定巡航1.0会产生大小为1.0m/s的平移指令。此修复不修改相机、奖励参数或路线规则，也不保证策略已经学会1m/s通过障碍。

## 用户Linux主机命令

更新代码后，在原环境新建实验：

```bash
cd /home/heartpain/mjlab/unitree_rl_mjlab_lgdog
conda activate unitree_rl_mjlab
python -m scripts.train LgDog1-Race-MGDP-WallTraining-D435i20 \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=16000 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --enable-nan-guard False \
  --terrain-num-rows=8 --terrain-num-cols=10 \
  --passage-fixed-speed=1.0 \
  --agent.run-name=V3_fixed1_seed42
```

启动日志应包含`Passage fixed cruise: 1 m/s`，该实验的`params/env.yaml`应记录巡航范围`(1.0, 1.0)`及课程参数`fixed_speed: 1.0`。

已运行的旧V2进程不会因更新文件而改变速度。若通过补丁覆盖，将`mgdp_fixed_cruise_20261115.zip`及同名`.sha256`放到`/home/heartpain/mjlab`，执行：

这是适用于已能启动D435i20任务工程的增量补丁；保留工作台已经保存的`src/wall_reward_settings.py`高墙参数。新部署使用完整工程包。

```bash
cd /home/heartpain/mjlab
sha256sum -c mgdp_fixed_cruise_20261115.zip.sha256 && unzip -o mgdp_fixed_cruise_20261115.zip -d /home/heartpain/mjlab
```

或者更新GitHub的`main`。覆盖后重启工作台Python后端并刷新网页。

## 记录、续训和回放

TensorBoard新增`Curriculum/passage/speed_min`和`speed_fixed`；固定1.0时，`speed_min`、`speed_max`、`speed_fixed`均为1.0。工作台运行状态读取这些实际日志，不根据表单选择伪装当前进程的速度。

检查点保存固定速度参数和范围。完整续训必须选择相同速度模式及固定值；旧渐进课程不能直接完整续训为固定1.0，错误会在分配物理环境前拒绝。若要迁移同一个D435i20任务的旧策略，在新命令中加入：

```text
--warm-start-checkpoint=/绝对路径/model_迭代数.pt
```

迁移保留配套感知和actor，重新创建critic、优化器、迭代计数及课程，不勾选完整续训。仅改变巡航速度不要求重新预训练感知；相机与感知仍须匹配。固定模式自身的完整续训需继续传入`--passage-fixed-speed=1.0`。

Play从检查点恢复固定巡航。ONNX记录训练巡航目标`mgdp_passage_fixed_speed`；Sim2Sim仍尊重用户显式给出的`--vx`/键盘指令，不会自行启动运动。

验证范围见`MGDP_FIXED_CRUISE_VALIDATION.json`。CPU短流程和配置测试用于确认1.0确实传入并保存，不代替GPU收敛或实物通过率验收。
