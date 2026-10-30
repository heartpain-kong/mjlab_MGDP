# HIM 三帧历史：训练与部署

2026-10-28。标准崎岖HIM、平地HIM、旧温和地形HIM的新训练统一由6帧改为3帧。统一常量位于 `src/him_contract.py`，环境观测、模型配置、历史编码器、部署缓存默认值与工作台维度使用同一来源。

| 项目 | 旧版 | 当前新训练 |
|---|---:|---:|
| 单帧观测 | 47 | 47 |
| 历史帧数 | 6 | 3 |
| actor历史张量 | [N,6,47] | [N,3,47] |
| ONNX输入 | [1,282] | [1,141] |
| 历史最早帧距当前 | 0.10秒 | 0.04秒 |
| 历史编码器首层 | 282→128 | 141→128 |
| 首层权重加偏置参数量 | 36,224 | 18,176 |

帧顺序仍为最旧到最新，reset重复首帧。估计速度3维、隐变量16维、actor的66维特征输入、critic、动作12维及控制周期20ms保持原配置。rollout仍为50，PPO和估计器epoch不随此项变更减少；奖励和地形参数也不因历史变更而重写。

## 怎么训练

必须新建3帧实验。6帧模型的编码器首层权重、优化器状态及观测形状不同，不能将旧checkpoint直接resume成3帧。模型加载会明确报告两边的历史协议。不要仅删除checkpoint中的协议标记或强行忽略shape错误。

```bash
# 工程根目录；仅预览，不启动
python lgdog.py train standard --terrain-curriculum lp_acrl \
  --velocity-min-abs 0.3 --velocity-max-abs 0.5 \
  --envs 1024 --iterations 16000 --dry-run
```

新run的 `params/env.yaml` 中actor history_length为3，`params/agent.yaml` 中actor history_length也为3。导出ONNX metadata应记录history_length=3、frame_dim=47、history_order=oldest_to_newest。

## Play、Sim2Sim与真机

- 新checkpoint在Play中自动使用3帧；旧6帧checkpoint在创建物理环境前读取自身的him_contract并按6帧建立观测和模型。该功能只用于回放，不会转换训练权重。
- Sim2Sim按ONNX metadata建立缓存，支持新141维、旧282维HIM及单帧PPO，不需要手动把旧ONNX改成141维。
- 真机程序应同步更换历史缓存长度，并按实际模型输入和metadata检查。若程序写死6帧/282维，不能直接替换为新ONNX。

## 时间和显存收益怎么判断

actor历史观测缓存/rollout中这部分存储减半，历史编码器首层的矩阵乘法量减半。以float32、1024环境、50步rollout为例，仅actor历史观测数组由约55.08MiB变为27.54MiB，节省约27.54MiB；2048环境节省约55.08MiB。这不是整个训练显存的估算，额外拷贝/分配也可能影响峰值。

物理仿真、碰撞、actor/critic主网络、target encoder和大部分辅助损失并未减半，所以不能据此保证训练速度翻倍。要在同GPU、同环境数、同rollout与epoch下，预热后比较iteration time、learning time和峰值显存。更短历史可能降低速度估计和地形适应能力，最终还需固定地形测试及训练曲线检验。

验证覆盖三帧PPO/HIM辅助更新、checkpoint恢复、六帧拒绝续训到三帧、旧模型Play协议恢复，以及真实ONNX/TorchScript导出与数值对齐；这些CPU检查不证明GPU提速或策略已收敛。

本次全量CPU测试171项：170项通过、1项跳过；JavaScript语法检查和轻量任务入口通过。没有启动GPU训练或覆盖已有策略文件。
