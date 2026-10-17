"""Read and safely persist LgDog reward weights without importing robotics packages."""

from __future__ import annotations

import ast
from datetime import datetime
import math
from pathlib import Path
from pprint import pformat
import re
import shutil


SOURCE_RELATIVE = Path("src/tasks/velocity/config/lgdog1/reward_cfg.py")
BEGIN = "# BEGIN LGDOG DASHBOARD OVERRIDES"
END = "# END LGDOG DASHBOARD OVERRIDES"
NUMBER_LIMIT = 10000.0

IMPLEMENTATIONS = {
  "track_linear_velocity": ("src/tasks/velocity/mdp/rewards.py", "track_linear_velocity"),
  "track_angular_velocity": ("src/tasks/velocity/mdp/rewards.py", "track_angular_velocity"),
  "body_orientation_l2": ("src/tasks/velocity/mdp/rewards.py", "body_orientation_l2"),
  "pose": ("src/tasks/velocity/mdp/rewards.py", "variable_posture"),
  "foot_gait": ("src/tasks/velocity/mdp/rewards.py", "feet_gait"),
  "body_ang_vel": ("src/tasks/velocity/mdp/rewards.py", "body_angular_velocity_penalty"),
  "angular_momentum": ("src/tasks/velocity/mdp/rewards.py", "angular_momentum_penalty"),
  "foot_clearance": ("src/tasks/velocity/mdp/rewards.py", "feet_clearance"),
  "foot_slip": ("src/tasks/velocity/mdp/rewards.py", "feet_slip"),
  "soft_landing": ("src/tasks/velocity/mdp/rewards.py", "soft_landing"),
  "feet_stumble": ("src/tasks/velocity/mdp/rewards.py", "feet_stumble"),
  "feet_air_time": ("src/tasks/velocity/mdp/rewards.py", "feet_air_time"),
  "leg_contact_force": ("src/tasks/velocity/mdp/rewards.py", "leg_contact_force"),
  "stand_still": ("src/tasks/velocity/mdp/rewards.py", "stand_still"),
}
INHERITED = {"joint_acc_l2", "joint_pos_limits", "action_rate_l2", "is_terminated"}


def _literal_assignment(tree, name):
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
      return ast.literal_eval(node.value)
  raise ValueError(f"奖励配置缺少 {name}")


def _load(root):
  path = Path(root) / SOURCE_RELATIVE
  source = path.read_text(encoding="utf-8-sig")
  tree = ast.parse(source, filename=str(path))
  common = _literal_assignment(tree, "COMMON_WEIGHTS")
  profiles = _literal_assignment(tree, "REWARD_PROFILES")
  try:
    overrides = _literal_assignment(tree, "DASHBOARD_REWARD_OVERRIDES")
  except ValueError:
    overrides = {name: {"weights": {}, "params": {}} for name in profiles}
  if not isinstance(common, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in common.items()):
    raise ValueError("COMMON_WEIGHTS 格式无效")
  return path, source, common, profiles, overrides


def read_rewards(root, profile):
  path, _, common, profiles, overrides = _load(root)
  if profile not in profiles:
    raise ValueError(f"未知 reward profile: {profile}")
  override = overrides.get(profile, {}) if isinstance(overrides, dict) else {}
  weights = dict(common)
  weights.update(profiles[profile].get("weights", {}))
  weights.update(override.get("weights", {}))
  params = dict(profiles[profile].get("params", {}))
  params.update(override.get("params", {}))
  return {
    "profile": profile,
    "source": str(path),
    "rewards": [{"name": name, "weight": weights[name], "enabled": weights[name] != 0,
                 "params": params.get(name, {})} for name in common],
  }


def reward_detail(root, profile, name):
  path, source, common, profiles, _ = _load(root)
  if name not in common:
    raise ValueError(f"未知 reward: {name}")
  if profile not in profiles:
    raise ValueError(f"未知 reward profile: {profile}")
  if name == "foot_clearance" and profile in {"terrain_v4", "him", "him_standard"}:
    relative = Path("src/tasks/velocity/config/lgdog1/blind_mdp.py")
    function = "feet_clearance_terrain"
  elif name in IMPLEMENTATIONS:
    relative, function = IMPLEMENTATIONS[name]
  elif name in INHERITED:
    return {"name": name, "title": f"{name} 的实现", "source": "mjlab.envs.mdp（已安装依赖）", "line": None,
            "binding": f"reward_cfg.py: mdp.{name}", "text": "这个 reward 由 mjlab 的基础环境包提供，当前工程通过 mdp.%s 绑定它。\n\n修改权重和方案参数请使用上方编辑器；如果要修改函数公式，需要进入已安装的 mjlab 源码。" % name}
  else:
    raise ValueError(f"reward 尚未配置实现映射: {name}")
  implementation_path = Path(root) / relative
  implementation_source = implementation_path.read_text(encoding="utf-8-sig")
  tree = ast.parse(implementation_source, filename=str(implementation_path))
  node = next((node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == function), None)
  if node is None:
    raise ValueError(f"找不到函数实现: {function}")
  lines = implementation_source.splitlines()
  start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
  return {"name": name, "title": f"{name} · {function}", "source": str(implementation_path), "line": start,
          "binding": f"reward_cfg.py: mdp.{function}", "text": "\n".join(lines[start - 1:end])}


def _validate_changes(common, profiles, profile, changes):
  if not isinstance(changes, list) or not changes:
    raise ValueError("至少提交一个 reward")
  names = set(common)
  if profile not in profiles:
    raise ValueError(f"未知 reward profile: {profile}")
  result = {}
  for item in changes:
    if not isinstance(item, dict) or item.get("name") not in names:
      raise ValueError("reward 名称无效")
    name = item["name"]
    try:
      value = float(item.get("weight"))
    except (TypeError, ValueError):
      raise ValueError(f"{name} 的权重必须是数字") from None
    if not math.isfinite(value) or abs(value) > NUMBER_LIMIT:
      raise ValueError(f"{name} 的权重需在 -{NUMBER_LIMIT:g} 到 {NUMBER_LIMIT:g} 之间")
    params = item.get("params", {})
    if not isinstance(params, dict):
      raise ValueError(f"{name} 的参数必须是 JSON 对象")
    for key, param in params.items():
      if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError(f"{name} 的参数名无效")
      if isinstance(param, float) and not math.isfinite(param):
        raise ValueError(f"{name}.{key} 必须是有限数字")
      if not isinstance(param, (str, int, float, bool, type(None))):
        raise ValueError(f"{name}.{key} 只支持字符串、数字或布尔值")
    result[name] = {"weight": value, "params": params}
  return result


def save_rewards(root, profile, changes):
  path, source, common, profiles, overrides = _load(root)
  validated = _validate_changes(common, profiles, profile, changes)
  current = overrides.get(profile, {"weights": {}, "params": {}})
  weights = dict(current.get("weights", {}))
  params = dict(current.get("params", {}))
  for name, item in validated.items():
    baseline = profiles[profile].get("weights", {}).get(name, common[name])
    if item["weight"] == baseline:
      weights.pop(name, None)
    else:
      weights[name] = item["weight"]
    profile_params = profiles[profile].get("params", {}).get(name, {})
    if item["params"] == profile_params:
      params.pop(name, None)
    else:
      params[name] = item["params"]
  overrides = dict(overrides)
  overrides[profile] = {"weights": weights, "params": params}
  block = f"{BEGIN}\nDASHBOARD_REWARD_OVERRIDES = {pformat(overrides, width=100, sort_dicts=False)}\n{END}"
  pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
  if pattern.search(source):
    updated = pattern.sub(block, source, count=1)
  else:
    marker = "\n\ndef make_lgdog_rewards"
    if marker not in source:
      raise ValueError("找不到奖励配置插入位置")
    updated = source.replace(marker, f"\n\n{block}{marker}", 1)
  backup = path.with_suffix(path.suffix + ".bak")
  shutil.copy2(path, backup)
  temp = path.with_suffix(path.suffix + ".tmp")
  temp.write_text(updated, encoding="utf-8", newline="\n")
  temp.replace(path)
  return {"profile": profile, "source": str(path), "backup": str(backup), "saved_at": datetime.now().isoformat(timespec="seconds"), "rewards": read_rewards(root, profile)["rewards"]}
