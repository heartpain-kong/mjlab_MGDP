"""Read and safely edit the project-local LgDog terrain profile.

The dashboard process must work without torch/mjlab imports, so this module
edits the small declarative assignment block in terrain_cfg.py with regexes.
The training process still imports the real dataclass config at runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any


TERRAIN_FILE = Path("src/tasks/velocity/config/lgdog1/terrain_cfg.py")
TERRAIN_TYPES = {
  "flat": {"label": "平地", "description": "基础行走平面"},
  "pyramid_stairs": {"label": "上台阶", "description": "上台阶，难度随行号增加"},
  "pyramid_stairs_inv": {"label": "下台阶", "description": "下台阶和落差"},
  "hf_pyramid_slope": {"label": "上坡", "description": "金字塔上坡"},
  "hf_pyramid_slope_inv": {"label": "下坡", "description": "金字塔下坡"},
  "random_rough": {"label": "随机起伏", "description": "随机高度噪声"},
  "wave_terrain": {"label": "波浪", "description": "周期波浪，默认 4 个波"},
}
RANGE_FIELDS = {
  "pyramid_stairs": ("step_height_range", "m"),
  "pyramid_stairs_inv": ("step_height_range", "m"),
  "hf_pyramid_slope": ("slope_range", "gradient"),
  "hf_pyramid_slope_inv": ("slope_range", "gradient"),
  "random_rough": ("noise_range", "m"),
  "wave_terrain": ("amplitude_range", "m"),
}


def _path(root: Path) -> Path:
  return root / TERRAIN_FILE


def _number(text: str, label: str) -> float:
  try:
    value = float(text)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{label} 必须是数字") from error
  if not math.isfinite(value):
    raise ValueError(f"{label} 必须是有限数字")
  return value


def _read_number(source: str, pattern: str, label: str) -> float:
  match = re.search(pattern, source)
  if not match:
    raise ValueError(f"terrain_cfg.py 缺少 {label} 配置")
  return _number(match.group(1), label)


def _read_range(source: str, name: str, field: str) -> tuple[float, float]:
  pattern = rf'cfg\.sub_terrains\["{re.escape(name)}"\]\.{field}\s*=\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)'
  match = re.search(pattern, source)
  if not match:
    raise ValueError(f"terrain_cfg.py 缺少 {name}.{field} 配置")
  return _number(match.group(1), f"{name}.{field}.min"), _number(match.group(2), f"{name}.{field}.max")


def read_terrain(root: Path) -> dict[str, Any]:
  path = _path(root)
  source = path.read_text(encoding="utf-8")
  result = {
    "path": str(path),
    "rows": int(_read_number(source, r"cfg\.num_rows\s*=\s*([0-9]+)", "num_rows")),
    "cols": int(_read_number(source, r"cfg\.num_cols\s*=\s*([0-9]+)", "num_cols")),
    "terrains": [],
  }
  for name, info in TERRAIN_TYPES.items():
    item = {
      "name": name,
      "label": info["label"],
      "description": info["description"],
      "proportion": _read_number(
        source,
        rf'cfg\.sub_terrains\["{re.escape(name)}"\]\.proportion\s*=\s*([-+0-9.eE]+)',
        f"{name}.proportion",
      ),
    }
    if name in RANGE_FIELDS:
      field, unit = RANGE_FIELDS[name]
      low, high = _read_range(source, name, field)
      item.update({"field": field, "unit": unit, "min": low, "max": high})
    result["terrains"].append(item)
  return result


def _replace(source: str, pattern: str, replacement: str, label: str) -> str:
  source, count = re.subn(pattern, replacement, source, count=1)
  if count != 1:
    raise ValueError(f"terrain_cfg.py 未找到 {label}，没有保存修改")
  return source


def save_terrain(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
  path = _path(root)
  source = path.read_text(encoding="utf-8")
  rows = int(_number(payload.get("rows"), "地形行数"))
  cols = int(_number(payload.get("cols"), "地形列数"))
  if not 2 <= rows <= 100 or not 1 <= cols <= 100:
    raise ValueError("地形行数需为 2～100，列数需为 1～100")
  terrains = payload.get("terrains")
  if not isinstance(terrains, list) or {item.get("name") for item in terrains} != set(TERRAIN_TYPES):
    raise ValueError("地形类型列表不完整")
  values = {item["name"]: item for item in terrains}
  proportions = {}
  for name in TERRAIN_TYPES:
    proportion = _number(values[name].get("proportion"), f"{name}.proportion")
    if proportion < 0.0 or proportion > 1.0:
      raise ValueError(f"{name} 比例需在 0～1 之间")
    proportions[name] = proportion
  if abs(sum(proportions.values()) - 1.0) > 1e-6:
    raise ValueError(f"地形比例总和必须为 1.0，当前为 {sum(proportions.values()):.6f}")

  source = _replace(source, r"cfg\.num_rows\s*=\s*[0-9]+", f"cfg.num_rows = {rows}", "num_rows")
  source = _replace(source, r"cfg\.num_cols\s*=\s*[0-9]+", f"cfg.num_cols = {cols}", "num_cols")
  for name, proportion in proportions.items():
    source = _replace(
      source,
      rf'(cfg\.sub_terrains\["{re.escape(name)}"\]\.proportion\s*=\s*)[-+0-9.eE]+',
      rf'\g<1>{proportion:g}',
      f"{name}.proportion",
    )
    if name in RANGE_FIELDS:
      field, _ = RANGE_FIELDS[name]
      low = _number(values[name].get("min"), f"{name}.{field}.min")
      high = _number(values[name].get("max"), f"{name}.{field}.max")
      if low < 0.0 or high < low:
        raise ValueError(f"{name}.{field} 范围无效")
      if field == "slope_range" and high > 2.0:
        raise ValueError(f"{name}.{field} 最大梯度不能超过 2.0")
      source = _replace(
        source,
        rf'(cfg\.sub_terrains\["{re.escape(name)}"\]\.{field}\s*=\s*)\([-+0-9.eE]+,\s*[-+0-9.eE]+\)',
        rf'\g<1>({low:g}, {high:g})',
        f"{name}.{field}",
      )
  backup = path.with_suffix(path.suffix + ".bak")
  backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
  path.write_text(source, encoding="utf-8")
  result = read_terrain(root)
  result["backup"] = str(backup)
  return result
