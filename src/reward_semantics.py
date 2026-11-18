"""LgDog reward sign rules shared by the training factory and web editor.

Keep this module free of simulator/torch imports so a bad override can still
be opened and corrected in the standard-library dashboard.
"""

NONNEGATIVE_COSTS = frozenset({
  "body_orientation_l2", "body_ang_vel", "angular_momentum", "joint_acc_l2",
  "joint_pos_limits", "action_rate_l2", "smoothness",
  "foot_clearance", "foot_slip",
  "soft_landing", "feet_stumble", "leg_contact_force", "stand_still", "is_terminated",
  "cheat", "wall_waiting",
})


def reward_semantics(name, weight):
  cost = name in NONNEGATIVE_COSTS
  warning = f"{name} 返回非负代价，正权重会奖励误差/碰撞；请改为 0 或负数" if cost and weight > 0 else None
  return {
    "kind": "cost" if cost else "reward",
    "weight_hint": "代价：负数惩罚，0 关闭" if cost else "正向奖励：正数启用，0 关闭",
    "warning": warning,
  }


def validate_reward_weights(weights):
  for name, weight in weights.items():
    warning = reward_semantics(name, weight)["warning"]
    if warning:
      raise ValueError(warning + "；同时检查 DASHBOARD_REWARD_OVERRIDES 中的旧覆盖")
