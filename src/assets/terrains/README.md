# 地形资产

| 文件 | 用途 | 谁在用 |
|---|---|---|
| `lgdog1_testmap.xml` | LgRobotDog 的部署测试图（191 个 geom） | `sim2sim.py --map lgdog1` |
| `race_obstacle_2027.xml` | 2027 比赛障碍赛场地（984 个 geom，14×6 m） | `sim2sim.py --map race` |
| `race_task_2027.xml` | 2027 比赛任务赛内场（105 个 geom，含 8 个动态物资箱） | `sim2sim.py --map race_task` |
| `race_task_labels/` | 物资箱标识纹理（食品/工具/仪器/药品） | `race_task_2027.xml` |

两个 `race_*.xml` 是**转换产物**，不要手改：由 `scripts/import_race_course.py` 从
[`27rc-dog-race`](https://github.com/uwvwko-zzz/27rc-dog-race) 的 `maps/*.xml` 生成，
原因（角度单位、default 污染、option 覆盖）写在脚本开头和 [doc/RACE_COURSE.md](../../../doc/RACE_COURSE.md)。

格式约定（同样适用于以后新增的地形 XML）：

- `compiler angle="radian"`，姿态用 `quat`，不要留 `euler`；
- 不带 `<option>` / `<default>` / `<visual>` / `<statistic>`：这些会改到机器人场景的全局设置；
- `sim2sim.py` 用 `MjSpec.attach` 挂载，所以文件里只应有 `<asset>` 和 `<worldbody>` 的几何/刚体。

训练地形不走这些 XML，而是 `src/tasks/velocity/config/lgdog1/race_terrain_cfg.py` 里的程序化拼块
（mjlab 的地形只能是 box / heightfield 拼块）。
