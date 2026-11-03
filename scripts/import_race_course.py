"""把 27rc-dog-race 的比赛场地转换成工程内可直接 attach 的「地形 XML」。

    python scripts/import_race_course.py /path/to/27rc-dog-race-main
    python scripts/import_race_course.py /path/to/27rc-dog-race-main --check

为什么要转换，而不是直接 `--map-file maps/obstacle_course_2027.xml`：

1. 比赛场地用 `<compiler angle="degree">`，工程里的机器人/场景是弧度。`MjSpec.attach`
   是**按目标 spec 的单位**解释拷贝过来的 `euler`，直接挂过去 844 个 euler 会全部错位。
   这里先用 MuJoCo 自己加载再 `mj_saveLastXML`，让所有姿态都被烘成 `quat`、
   `compiler angle="radian"`，attach 就不会有歧义。
2. 场地自带 `<option>`/`<visual>`/`<statistic>`/`<default>`/灯光/相机。
   `<default>` 会改到机器人自己的 geom 默认值（摩擦 0.9 会传染给机器人脚底），
   所以这里把 default 里的物理参数**逐 geom 烘进去**，再把 default 整块删掉。
3. 任务赛的物资箱贴图是 `../assets/task_labels/*.png`，转换时拷到
   `src/assets/terrains/race_task_labels/` 并改写成相对路径。

输出：
  src/assets/terrains/race_obstacle_2027.xml   障碍赛（14×6 m，纯几何）
  src/assets/terrains/race_task_2027.xml       任务赛（6×4 m 内场 + 8 个动态物资箱）
  src/assets/terrains/race_task_labels/*.png   物资箱标识
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "src" / "assets" / "terrains"
LABEL_DIR = OUT_DIR / "race_task_labels"
MAPS = {"race_obstacle_2027": "obstacle_course_2027", "race_task_2027": "task_course_2027"}
KEEP_TOP_LEVEL = {"compiler", "asset", "worldbody"}


def flatten(source: Path, target: Path) -> dict:
  """把场地烘成「弧度 + quat + 无 default」的地形 XML。返回统计信息。"""
  model = mujoco.MjModel.from_xml_path(str(source))
  with tempfile.TemporaryDirectory() as temp:
    baked = Path(temp) / "baked.xml"
    mujoco.mj_saveLastXML(str(baked), model)
    tree = ET.parse(baked)
  root = tree.getroot()

  # 1) 只有 compiler / asset / worldbody 保留；option、visual、statistic、default、灯光、相机全删
  default_geom = {}
  for child in list(root):
    if child.tag == "default":
      geom = child.find("geom")
      if geom is not None:
        default_geom = dict(geom.attrib)
      root.remove(child)
    elif child.tag not in KEEP_TOP_LEVEL:
      root.remove(child)

  worldbody = root.find("worldbody")
  assert worldbody is not None, "场地没有 worldbody"
  for light in list(worldbody.findall("light")):
    worldbody.remove(light)
  for camera in list(worldbody.findall("camera")):
    worldbody.remove(camera)
  for body in list(worldbody.findall("body")):
    for light in list(body.findall("light")):
      body.remove(light)

  # 2) 把 default 里的物理参数烘到每个 geom 上；material 只影响外观，保留原样
  baked_attrs = 0
  for geom in root.iter("geom"):
    for key, value in default_geom.items():
      if key in {"material", "rgba", "name"}:
        continue
      if key not in geom.attrib:
        geom.set(key, value)
        baked_attrs += 1

  # 3) 贴图路径改到工程内的相对目录
  textures = []
  if LABEL_DIR.is_dir():
    for texture in root.iter("texture"):
      filename = texture.get("file")
      if not filename:
        continue
      name = Path(filename).name
      texture.set("file", f"race_task_labels/{name}")
      textures.append(name)

  target.parent.mkdir(parents=True, exist_ok=True)
  tree.write(target, encoding="utf-8", xml_declaration=True)
  return {
    "geoms": sum(1 for _ in root.iter("geom")),
    "bodies": sum(1 for _ in root.iter("body")),
    "meshes": sum(1 for _ in root.iter("mesh")),
    "textures": textures,
    "baked_attrs": baked_attrs,
    "bytes": target.stat().st_size,
  }


def main():
  parser = argparse.ArgumentParser(description="转换 27rc-dog-race 场地为工程地形 XML")
  parser.add_argument("source", help="27rc-dog-race-main 目录（含 maps/ 与 assets/）")
  parser.add_argument("--check", action="store_true", help="只检查源文件，不写工程")
  args = parser.parse_args()

  source = Path(args.source).expanduser().resolve()
  maps = source / "maps"
  if not maps.is_dir():
    parser.error(f"找不到 {maps}")

  labels = source / "assets" / "task_labels"
  if labels.is_dir() and not args.check and not LABEL_DIR.exists():
    shutil.copytree(labels, LABEL_DIR)
    print(f"贴图 → {LABEL_DIR}  ({len(list(LABEL_DIR.glob('*.png')))} 张)")

  for out_name, src_name in MAPS.items():
    src = maps / f"{src_name}.xml"
    if not src.is_file():
      print(f"[skip] 缺 {src}")
      continue
    if args.check:
      model = mujoco.MjModel.from_xml_path(str(src))
      print(f"[check] {src.name}: ngeom={model.ngeom} nbody={model.nbody} njnt={model.njnt}")
      continue
    target = OUT_DIR / f"{out_name}.xml"
    info = flatten(src, target)
    print(f"{src.name} → {target.relative_to(ROOT)}  "
          f"geoms={info['geoms']} bodies={info['bodies']} meshes={info['meshes']} "
          f"textures={info['textures']} 烘参数={info['baked_attrs']} {info['bytes'] / 1024:.0f} KB")

  if not args.check:
    print("\n下一步：python scripts/sim2sim.py <task> --map race --duration 3")
  return 0


if __name__ == "__main__":
  sys.exit(main())
