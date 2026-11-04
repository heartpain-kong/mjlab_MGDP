# MGDP 循环训练与 sim2sim 修复验证

日期：2026-11-04。设备：本机 CPU，PyTorch 2.14.1+cpu；未运行 GPU 或实机。

## 修改

- 预训练采集连续序列，episode 长度不连续时丢弃整段；默认长度 8。
- 保留原有损失，增加 GRU 输出到干净深度 token 的 MSE，默认权重 1。
- 导出 MGDP actor 时保存配套感知模型及 SHA-256；sim2sim 使用原针孔图案、相机 site、几何组和排除设置读取深度。
- 部署按每个本体观测项分别展开历史，然后拼接感知特征；复位清空所有历史。
- 新导出拒绝错误感知指纹；旧导出必须指定 `--perception`，由使用者确认来自该策略。

## 本轮实际执行

1. 感知/部署/HIM 协议/比赛地形相关 27 项测试全部通过，包含错误权重及旧格式入口反例。
2. GRU 序列反向传播：全部循环参数存在非零梯度，优化后 `weight_hh_l0` 变化。
3. 解析墙面深度与 CPU 射线结果一致，部署 GRU 特征与训练观测函数一致；历史顺序、复位及跨回合序列过滤检查通过。
4. CPU 预训练：2 环境，24 控制步，4 步序列，batch=2，2 次更新；产生 10 条有效序列、感知检查点，验证总损失约 0.973。
5. CPU PPO：seed=42，2 环境，每环境 4 步，1 次更新；真实导出 `policy.onnx`、`policy.perception.pt`，接口 `[1,252] -> [1,12]`。
6. 上述导出策略在 `LgDog1-Race-MGDP --map race` 中完成 0.2 秒 / 10 个控制步，无接口异常；这不证明策略收敛或越障能力。

预训练、PPO 和回放产物保存在项目内 `.artifacts/mgdp-fix/`，该目录已加入忽略规则。`ppo.log`、`sim2sim.log`、`tests.log`、`pipeline-tests.log` 可核对本轮结果。

用训练依赖环境执行专项测试：

```bash
python -B -m unittest tests.test_mgdp_pipeline tests.test_mgdp_perception tests.test_him_contract_light tests.test_race_course
```

既有预训练文件不会因源码修复自动获得训练后的 GRU。新感知模型需配套新的策略训练；已有策略回放应保留其原感知权重。使用说明见 [MGDP_MJLAB.md](MGDP_MJLAB.md)。
