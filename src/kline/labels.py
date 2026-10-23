"""指标词典：tag → 分组 / 中文名 / 方向 / 一句话解释。

页面上的「指标词典」抽屉、搜索、副图下拉都读这里；没有登记过的 tag 不会报错，
会退回按前缀分组 + 用 tag 末段当名字，所以新指标加进来立刻能看。
"""

GROUPS = (
  ("train", "训练", "Train/"),
  ("curriculum", "课程", "Curriculum/"),
  ("reward", "奖励项", "Episode_Reward/"),
  ("episode", "回合指标", "Episode_Metrics/"),
  ("metrics", "指标", "Metrics/"),
  ("loss", "损失", "Loss/"),
  ("policy", "策略", "Policy/"),
  ("termination", "终止", "Episode_Termination/"),
  ("perf", "性能", "Perf/"),
)

# tag: (分组, 中文名, 方向, 解释)
#   方向 up = 越大越好，down = 越小越好，"" = 中性/看情况
TAGS = {
  "Train/mean_reward": ("train", "平均回报", "up",
                        "每个迭代的平均回合回报，训练好不好主要看它。爬升变慢通常是策略已经接近这套奖励的上限。"),
  "Train/mean_episode_length": ("train", "平均回合长度", "up",
                                "回合平均能撑多少步。越接近上限（1000 步 = 20 秒）说明越少摔倒。"),
  "Train/mean_reward/time": ("train", "平均回报·累计耗时", "",
                             "同一个平均回报，横轴换成训练累计秒数 —— 想比「谁学得快」看这条。"),
  "Train/mean_episode_length/time": ("train", "回合长度·累计耗时", "",
                                     "回合长度对累计秒数，用来看同实际耗时下的收敛速度。"),

  "Curriculum/terrain_levels/level_mean": ("curriculum", "地形等级", "up",
                                           "课程地形的平均等级。塌到 0 说明升级判据失效，不是策略变差。"),
  "Curriculum/terrain_levels/promote_fraction": ("curriculum", "升级比例", "up",
                                                 "本批里成功升级的地形比例。早期为 0 是正常的，策略还没学会跟速度。"),
  "Curriculum/terrain_levels/demote_fraction": ("curriculum", "降级比例", "down",
                                                "本批里被降级的比例，持续偏高说明当前等级太难。"),
  "Curriculum/terrain_levels/tracking_ratio": ("curriculum", "跟踪比", "up",
                                               "实际前进距离 ÷ 指令距离，是升级判据里最硬的那一项。"),
  "Curriculum/terrain_levels/reset_batch_failure_fraction": ("curriculum", "失败率", "down",
                                                             "本批 reset 里判定失败的比例，摔倒、超时、非法接触都算。"),
  "Curriculum/terrain_levels/episode_commanded_m": ("curriculum", "指令距离", "",
                                                    "本批回合被指令要求的距离（米），会随指令范围课程变化。"),
  "Curriculum/terrain_levels/episode_progress_m": ("curriculum", "实际前进", "up",
                                                   "本批回合真实走出的距离（米），和上面那条比就是跟踪比。"),
  "Curriculum/terrain_levels/velocity_error_xy_mps": ("curriculum", "课程速度误差", "down",
                                                     "课程判据里用的速度误差（旧 run 才记录）。"),

  "Episode_Reward/track_linear_velocity": ("reward", "线速度跟踪奖励", "up",
                                           "跟住指令线速度给的奖励，一般是回报里最大的一项。"),
  "Episode_Reward/track_angular_velocity": ("reward", "角速度跟踪奖励", "up",
                                            "跟住指令偏航角速度给的奖励。"),
  "Episode_Reward/body_orientation_l2": ("reward", "机身姿态惩罚", "down",
                                         "机身倾斜的二次惩罚（负值），越接近 0 越好。"),
  "Episode_Reward/pose": ("reward", "姿态跟踪", "",
                          "关节姿态与默认姿态的偏差项，正负取决于配置。"),
  "Episode_Reward/foot_gait": ("reward", "步态节奏", "up", "四条腿按时序交替的奖励。"),
  "Episode_Reward/body_ang_vel": ("reward", "机身角速度惩罚", "down",
                                  "机身转动越快扣得越多（负值），用来压住抖动。"),
  "Episode_Reward/angular_momentum": ("reward", "角动量惩罚", "down",
                                      "角动量越大扣得越多，抑制甩腿。"),
  "Episode_Reward/joint_acc_l2": ("reward", "关节加速度惩罚", "down",
                                  "关节加速度的二次惩罚，直接影响动作是否平滑。"),
  "Episode_Reward/joint_pos_limits": ("reward", "关节限位惩罚", "down",
                                      "顶到关节限位就扣分，持续为负说明动作幅度过大。"),
  "Episode_Reward/action_rate_l2": ("reward", "动作变化率惩罚", "down",
                                    "相邻两步动作差的平方（负值）。抖动大时它先炸。"),
  "Episode_Reward/foot_clearance": ("reward", "抬脚高度", "up",
                                    "抬脚到目标高度的奖励，太低会绊脚、太高像在踏步。"),
  "Episode_Reward/foot_slip": ("reward", "足端打滑惩罚", "down", "支撑脚滑动就扣分。"),
  "Episode_Reward/soft_landing": ("reward", "软着陆", "up", "落地冲击小的奖励。"),
  "Episode_Reward/feet_stumble": ("reward", "绊脚惩罚", "down", "脚蹭到台阶立面扣分。"),
  "Episode_Reward/feet_air_time": ("reward", "腾空时间", "up", "鼓励步子迈开来。"),
  "Episode_Reward/leg_contact_force": ("reward", "接触力惩罚", "down", "支撑腿垂直力过大扣分。"),
  "Episode_Reward/stand_still": ("reward", "静止惩罚", "down", "指令为 0 时还在动才扣，属正常项。"),
  "Episode_Reward/is_terminated": ("reward", "终止惩罚", "down", "回合被提前终止的惩罚。"),

  "Episode_Metrics/mean_action_acc": ("episode", "平均动作加速度", "down",
                                      "动作二阶差分，越大越抖。V10 里这条一直是关注重点。"),
  "Episode_Metrics/traveled_distance": ("episode", "实际行走距离", "up", "回合里真实走出的米数。"),
  "Episode_Metrics/commanded_distance": ("episode", "指令距离", "", "本回合指令要求的米数。"),
  "Episode_Metrics/tracking_progress": ("episode", "跟踪进度", "up", "实际 / 指令，标量化的跟踪完成度。"),
  "Episode_Metrics/outside_spawn_platform": ("episode", "离开出生平台比例", "up",
                                             "回合里待在出生平台之外的时间占比，太低说明没走起来。"),
  "Episode_Metrics/velocity_error_xy_mps": ("episode", "水平速度误差", "down",
                                            "实际与指令速度的误差（m/s）。⚠️ mjlab 的 metrics 不做 dt 缩放，"
                                            "拿它判课程时要除以步数而不是秒数。"),

  "Metrics/twist/error_vel_xy": ("metrics", "线速度误差", "down", "指令线速度跟踪误差，越小越好。"),
  "Metrics/twist/error_vel_yaw": ("metrics", "偏航误差", "down", "指令偏航角速度误差。"),
  "Metrics/track_lin_vel": ("metrics", "线速度跟踪", "up", "旧任务里的跟踪得分（越高越好）。"),
  "Metrics/angular_momentum_mean": ("metrics", "平均角动量", "down", "机身角动量均值，越小越稳。"),
  "Metrics/foot_clearance_ray_valid": ("metrics", "抬脚射线有效率", "up",
                                       "脚底测距射线打中地形的比例，低说明踩空或被遮挡。"),
  "Metrics/slip_velocity_mean": ("metrics", "平均打滑速度", "down", "支撑脚滑动速度均值。"),
  "Metrics/landing_force_mean": ("metrics", "平均落地力", "down", "落地冲击力均值。"),
  "Metrics/feet_stumble_mean": ("metrics", "平均绊脚", "down", "绊脚判定均值。"),
  "Metrics/leg_contact_force_max": ("metrics", "最大接触力", "down", "支撑腿垂直力峰值。"),

  "Loss/value": ("loss", "价值函数损失", "down", "critic 的拟合误差，平稳就好；突然爆通常会触发 nan guard。"),
  "Loss/surrogate": ("loss", "PPO 策略损失", "", "策略梯度那一项，量级随实现而定，看趋势不看绝对值。"),
  "Loss/entropy": ("loss", "策略熵", "",
                   "策略的随机程度。掉到 0 附近说明过早收敛、探索没了；突然冲高说明策略在崩。"),
  "Loss/him_valid_fraction": ("loss", "HIM 有效率", "up", "HIM 估计器里有效样本的占比。"),
  "Loss/him_velocity_mse": ("loss", "HIM 速度 MSE", "down", "HIM 编码器重构速度的误差。"),
  "Loss/him_swap": ("loss", "HIM 交换误差", "down", "交换隐变量的重构误差，看编码器有没有真学到东西。"),
  "Loss/him_updates": ("loss", "HIM 更新次数", "", "估计器累计更新步数。"),
  "Loss/learning_rate": ("loss", "学习率", "", "当前学习率，会随自适应调度变化。"),

  "Policy/mean_std": ("policy", "策略标准差", "down",
                      "动作分布的方差项。涨到 3 以上通常意味着动作在发散。"),

  "Episode_Termination/time_out": ("termination", "超时终止", "",
                                   "回合撑满时间上限就结束，这项高其实是好事。"),
  "Episode_Termination/fell_over": ("termination", "摔倒比例", "down", "机身倾角超限被终止的比例。"),
  "Episode_Termination/illegal_contact": ("termination", "非法接触比例", "down", "机身、大腿等不该碰地的部位碰地。"),
  "Episode_Termination/sustained_leg_contact": ("termination", "持续腿接触比例", "down",
                                                "小腿长时间贴地之类的持续接触。"),

  "Perf/total_fps": ("perf", "训练帧率", "up", "每秒推进的环境步数，掉一半先看是不是换了并行环境数。"),
  "Perf/collection_time": ("perf", "采样耗时", "down", "一个迭代里采样花掉的秒数。"),
  "Perf/learning_time": ("perf", "学习耗时", "down", "一个迭代里更新网络花掉的秒数。"),
}

# 顶部指数条 + 轮次对比表默认盯的指标（按这个顺序显示）
KEY_TAGS = (
  "Train/mean_reward",
  "Curriculum/terrain_levels/level_mean",
  "Curriculum/terrain_levels/tracking_ratio",
  "Curriculum/terrain_levels/reset_batch_failure_fraction",
  "Episode_Metrics/velocity_error_xy_mps",
  "Policy/mean_std",
  "Loss/entropy",
  "Perf/total_fps",
)

PREFIX_ALIASES = (
  ("Train/", "train", "训练"),
  ("Curriculum/", "curriculum", "课程"),
  ("Episode_Reward/", "reward", "奖励项"),
  ("Episode_Metrics/", "episode", "回合指标"),
  ("Metrics/", "metrics", "指标"),
  ("Loss/", "loss", "损失"),
  ("Policy/", "policy", "策略"),
  ("Episode_Termination/", "termination", "终止"),
  ("Perf/", "perf", "性能"),
)


def describe(tag):
  """返回 {'tag','group','group_label','label','direction','desc','known'}。"""
  entry = TAGS.get(tag)
  if entry:
    group, label, direction, desc = entry
    return {
      "tag": tag,
      "group": group,
      "group_label": _group_label(group),
      "label": label,
      "direction": direction,
      "desc": desc,
      "known": True,
    }
  group, group_label = _group_of(tag)
  leaf = tag.rsplit("/", 1)[-1].replace("_", " ")
  return {
    "tag": tag,
    "group": group,
    "group_label": group_label,
    "label": leaf,
    "direction": "",
    "desc": "没有登记过的指标，按 tag 末段显示；数值含义以训练代码为准。",
    "known": False,
  }


def describe_all(tags):
  entries = [describe(tag) for tag in tags]
  order = {key: index for index, (key, _, _) in enumerate(GROUPS)}
  entries.sort(key=lambda item: (order.get(item["group"], 99), item["label"]))
  return entries


def _group_of(tag):
  for prefix, key, label in PREFIX_ALIASES:
    if tag.startswith(prefix):
      return key, label
  return "other", "其它"


def _group_label(key):
  for group_key, label, _prefix in GROUPS:
    if key == group_key:
      return label
  return "其它"
