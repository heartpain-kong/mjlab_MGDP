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
