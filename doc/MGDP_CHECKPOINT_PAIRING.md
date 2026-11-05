# Play 停在障碍前：先核对策略与感知模型配对

2026-11-05 用户报告：Play 中旧策略在障碍前停止，且该策略是在重新预训练感知模型之前训练的。
这与代码中发现的加载问题相符，但尚未在用户的原始checkpoint和场景中复现行为。

## 明确发现的加载问题

MGDP PPO actor输入为本体历史加64维深度特征。这个64维向量不是固定物理量，
其语义取决于冻结的深度编码器和GRU参数。重新预训练后，维度可以不变，特征含义却已变化。
旧actor的权重和归一化统计仍基于原编码器，因此“文件能加载、维度匹配”不能证明组合正确。

原来Play从当前任务配置读取公共文件 `logs/mgdp/perception/mgdp_race_v3.pt`，
没有按选中的策略恢复原感知模型。公共文件被重新预训练覆盖后，旧策略会静默用上新特征。
此前ONNX/Sim2Sim已导出同run的 `policy.perception.pt`，但Play未使用它。

## 本次修复

- 保存新策略checkpoint时，在 `infos.mgdp_perception` 记录编码器参数/config指纹、
  快照文件名、文件SHA256、控制间隔和本体历史长度。
- 同run保存 `perception_<参数指纹>.pt`，同一编码器只需一份；之后的公共感知预训练不会替换它。
  复制策略时应保留该文件，建议复制整个run目录。
- Play在创建物理环境、生成第一帧观测前自动选取该快照，检查完整性，恢复历史长度，核对控制间隔。
  runner在加载actor前再次核对实际加载的感知模块参数。
- 完整续训也先恢复配套感知模型，再初始化环境，避免继续训练时混用公共新模型。
- 旧checkpoint优先使用同run的ONNX元数据及其感知快照，并检查已有SHA256；
  没有ONNX时可读取同run的 `policy.perception.pt`。
  旧checkpoint没有逐checkpoint的指纹记录，因此若同run曾更换编码器，仍须确认具体对应关系。
- 旧目录没有感知快照时，明确报错并要求指定原训练感知文件，不再静默回退到公共新文件。
  原编码器已丢失时，不能从actor权重逆推出它，需要找备份或重新训练策略。

修改集中在 `src/mgdp_checkpoint.py`、`scripts/play.py`、`scripts/train.py` 和
`src/tasks/velocity/rl/runner.py`。普通非MGDP策略不进入这套加载逻辑。

## 使用与确认

更新代码后，结束旧Play进程并重新启动；已经运行的窗口不会热更新。
工作台仍按原方式选策略、启动Play。查看日志中的：

```text
[INFO] MGDP policy encoder: .../<所选run>/policy.perception.pt
[INFO] MGDP encoder SHA256: ...
```

新checkpoint显示的是该run的 `perception_<参数指纹>.pt`。
若旧run没有快照，但保留了原感知文件，可以显式指定：

```bash
python -m scripts.play LgDog1-Race-MGDP \
  --checkpoint-file /path/to/run/model_1000.pt \
  --perception-checkpoint /path/to/original_encoder.pt
```

新checkpoint的显式覆盖也必须匹配已记录的编码器指纹，不能借此强行混用新模型。
旧checkpoint没有指纹时，显式文件的来源必须能证明它来自原训练。

## 箭头与“看见障碍”

若指的是速度可视化箭头：蓝色为目标线速度、青色为实际线速度，绿色系列为角速度。
它们不表示障碍识别；射线命中障碍则表示几何扫描有结果，也不等同于策略掌握了通过动作。

本次先修复确定存在的配对问题，保持奖励不变，以便按同一checkpoint、同一障碍和正确编码器复测。
如果正确配对后仍停住，再检查目标速度是否非零、实际速度/路线进度、训练轮数、接触代价和奖励分解。
旧代码中静止时部分跟踪/姿态奖励仍为正，这也是需要单独验证的训练因素；修复加载不等于策略已经学会越障。

## 验证

回归覆盖：覆盖公共文件后恢复原特征、错误编码器拒绝、快照损坏拒绝、旧ONNX文件校验、
旧文件缺失提示、Play在物理初始化前绑定、非MGDP保持原行为。
CPU PPO实际更新、保存含感知绑定的新checkpoint、恢复后再次更新通过。
全套286项测试，0失败、1跳过；实际旧版与新版PPO检查点均验证了感知文件的选择。
临时记录位于忽略目录 `.artifacts/mgdp-play-pairing/`。
未执行用户Linux上的原checkpoint视觉验收或GPU长训，越障效果仍待该组合实测。
