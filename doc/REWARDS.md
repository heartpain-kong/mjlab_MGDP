# LgDog 奖励调参指南

从 `src/tasks/velocity/config/lgdog1/reward_cfg.py` 开始。LgDog 的奖励权重、公共参数、分任务参数和函数绑定都集中在这一个文件；`env_cfgs.py` 负责机器人、地形、指令、观测和终止。

## 先看最终生效的奖励

在工程根目录执行，使用已安装项目依赖的 Python 环境。这些命令只加载配置，不创建仿真环境，不需要 GPU 或 checkpoint：

```bash
# 查看全部权重和每个函数的源码位置
python scripts/list_rewards.py LgDog1-Rough-Blind-HIM

# 展开某个奖励的最终参数
python scripts/list_rewards.py LgDog1-Rough-Blind-HIM --term foot_clearance
python scripts/list_rewards.py LgDog1-Rough-Blind-HIM --term pose

# 查看回放配置，或者对照平地任务
python scripts/list_rewards.py LgDog1-Rough-Blind-HIM --play
python scripts/list_rewards.py LgDog1-Flat
```

输出中的 `source` 是函数所在的实际文件和行号，包括 mjlab 包中的基础函数。遇到“改了代码却没生效”，先确认这里指向的函数。`--term` 展示的是所有配置组合完成之后的参数；权重为 0 的项也会列出，并明确表示关闭。

## 配置位置与影响范围

| 要修改什么 | 修改位置 | 影响范围 |
|---|---|---|
| 公共权重 | `reward_cfg.py` 中的 `COMMON_WEIGHTS` | 所有 LgDog 任务，除非某个 profile 覆盖该权重 |
| 只改一个任务组的权重 | `REWARD_PROFILES[配置名]["weights"]` | 对应任务组 |
| 只改一个任务组的参数 | `REWARD_PROFILES[配置名]["params"][奖励名]` | 对应任务组 |
| 公共参数、函数绑定、作用的身体/关节 | `make_lgdog_rewards()` 中对应的 `term(...)` | 所有 LgDog 任务，除非 profile 覆盖该参数 |
| 奖励的计算公式 | `src/tasks/velocity/mdp/rewards.py`，或 `list_rewards.py` 指向的源码 | 所有使用该函数的任务，可能包含其他机器人 |
| 相对地面抬脚公式 | `config/lgdog1/blind_mdp.py::feet_clearance_terrain` | TerrainV4 和 HIM |

各 profile 都直接覆盖公共设置，相互独立：

| 配置名 | 任务 |
|---|---|
| `flat` | `LgDog1-Flat` |
| `rough` | `LgDog1-Rough`、`LgDog1-Rough-Blind`、`LgDog1-Rough-Blind-Gentle` |
| `terrain_v4` | `LgDog1-Rough-Blind-TerrainV4` |
| `him` | `LgDog1-Rough-Blind-HIM` |
| `him_standard` | `LgDog1-Rough-Blind-HIM-Standard`（标准 Rough，actor/critic 均无机身高度扫描） |

例如只调 HIM 的抬脚惩罚，在 `REWARD_PROFILES` 的 `him` 项中，将 `"weights": {}` 改为 `"weights": {"foot_clearance": -0.8}`。要同时调目标高度，在该项已有的 `"params" → "foot_clearance"` 中修改 `"target_height"`。更改 `him` 不会影响 TerrainV4 或 Gentle。

新标准崎岖地形 HIM 使用独立的 `him_standard`，同样按上述方法修改；它不继承 `him`。训练与输入说明见 [LGDOG_HIM_STANDARD.md](LGDOG_HIM_STANDARD.md)。

这次重构将 LgDog 的奖励配置独立出来。改通用 `velocity_env_cfg.py` 的奖励默认值不会再改变 LgDog 的这些配置；其他机器人的配置方式维持原有规则。算法函数仍可共享。

## 当前默认值

公共权重以本次重构前工程中的实际值为基准：

| 奖励名 | 公共权重 | 含义 |
|---|---:|---|
| `track_linear_velocity` | 2.0 | 跟踪线速度，包括竖直速度代价 |
| `track_angular_velocity` | 1.0 | 跟踪偏航角速度，并抑制 roll/pitch 角速度 |
| `body_orientation_l2` | -1.0 | 抑制机身倾斜 |
| `pose` | 1.5 | 按站立/行走/奔跑标准差约束姿态 |
| `foot_gait` | 0.5 | 按相位检查四足接触节律 |
| `body_ang_vel` | -0.05 | 抑制机身 roll/pitch 角速度 |
| `angular_momentum` | -0.025 | 抑制全身角动量 |
| `joint_acc_l2` | -2.5e-7 | 抑制关节加速度 |
| `joint_pos_limits` | -10.0 | 软限位代价 |
| `action_rate_l2` | -0.05 | 抑制相邻动作变化 |
| `foot_clearance` | -1.0 | 抬脚高度代价，见下方任务差异 |
| `foot_slip` | -0.25 | 支撑足打滑代价 |
| `soft_landing` | -0.001 | 落地冲击代价 |
| `feet_stumble` | -0.5 | 脚撞立面代价；Flat 覆盖为 0 |
| `feet_air_time` | 0 | 足端空中时间目标，当前关闭 |
| `leg_contact_force` | -1.0 | 腿部接触力代价；Flat 覆盖为 0 |
| `stand_still` | -1.0 | 零指令站立时的姿态代价 |
| `is_terminated` | -200.0 | 非超时终止代价 |

`foot_clearance.target_height`：Flat 为 0.10 m，Rough/Blind/Gentle 为 0.15 m，TerrainV4/HIM 为 0.10 m。前两类使用世界坐标 z；TerrainV4/HIM 使用足端相对正下方地面的高度，而且仅对摆动脚计入。相同数值不代表相同的奖励定义。TerrainV4/HIM 的腿部接触力奖励还会自动绑定到 `legs_ground_contact`。

每个控制步的贡献是 `raw_reward × weight × step_dt`，本工程默认 `step_dt=0.02 s`。例如终止项返回 1 时贡献为 `-200 × 0.02 = -4`。连续接触力、二值碰撞与其他框架的奖励返回值可能不同，不能仅按 dt 换算权重。

## 添加新奖励

1. 在 `src/tasks/velocity/mdp/rewards.py` 定义函数，返回 `[num_envs]` 张量。该模块通过 `mdp/__init__.py` 导出；若写在新的函数模块中，还需显式导出。
2. 在 `COMMON_WEIGHTS` 增加一个同名权重。希望默认关闭可设为 0。
3. 在 `make_lgdog_rewards()` 的字典里增加绑定。例如函数 `my_reward(env, command_name)`：

```python
"my_term": term("my_term", mdp.my_reward, command_name="twist"),
```

4. 只希望 HIM 使用时，将公共权重设为 0，再在 `REWARD_PROFILES["him"]["weights"]` 中设为所需值。新函数若依赖额外传感器，需要在环境配置里创建对应传感器。
5. 用 `list_rewards.py ... --term my_term` 确认绑定和参数，再进行短训练。

未知 profile 或 profile 中拼错的奖励名会直接报错；新函数的参数必须与其签名一致。

## 调参如何生效

源文件修改只在新的 Python 进程里加载。修改后重新启动训练；已运行的训练进程不会热更新。恢复旧 checkpoint 时会使用本次启动的奖励配置，改变奖励后建议用新的 run name 记录，避免将它与原实验混为一谈。

一次训练临时改数仍可使用原 CLI 覆盖：

```bash
python scripts/train.py LgDog1-Rough-Blind-HIM \
  --env.scene.num-envs=64 --agent.max-iterations=2 \
  --agent.logger=tensorboard \
  --env.rewards.foot_clearance.weight=-0.8
```

这种 CLI 覆盖发生在任务配置构建之后，只影响这次启动；`list_rewards.py` 展示的是源码默认配置。CLI 覆盖后的值应查看训练启动的 Reward 表和该 run 的 `params/env.yaml` 快照。

profile 的 `params` 按参数名覆盖，某个参数自身为字典时会整体替换。例如覆盖 `pose.std_standing` 时要提供所需的全部关节匹配规则。

## 本次重构的检查

使用本机 Python 3.12 验证环境完成 27 项 CPU 测试，其中新增 4 项检查 HIM 单独改权重、参数传递、任务间可变对象隔离以及配置错误提示。另从用户的 `unitree_rl_mjlab_lgdog.zip` 读取原工厂代码，逐项比较 6 个 LgDog 任务的训练/回放配置，共 12 组；18 个奖励项的顺序、函数绑定、权重和参数保持一致。

`list_rewards.py --help` 与 HIM 的 `--term foot_clearance` 实际运行通过，输出正确指向 `blind_mdp.py::feet_clearance_terrain`，并显示 0.10 m 目标高度、四个足端和四条地面射线。

之后改配置时，可运行这组隔离检查：

```bash
python -m unittest discover -s tests -p "test_lgdog_reward_profiles.py" -v
```

这些检查验证配置组织与隔离，不证明调参后的行走表现。新的权重仍需通过短训练和相同条件下的策略对比评估。
