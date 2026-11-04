"""Optional terrain-only automatic curriculum; default task factories stay manual."""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from . import blind_mdp
from src.tasks.velocity.mdp.lp_terrain import LPTerrainCurriculum
from src.curriculum.lp import LearningProgressSampler


def configure_terrain_curriculum(cfg, mode="manual", *, temperature=2.0, uniform_mix=0.5,
                                 min_episodes=32, update_interval_s=100.0, status_path=None,
                                 progress_alpha=0.25, probability_alpha=0.2,
                                 max_task_probability=0.25):
  if mode not in {"manual", "uniform", "lp_acrl"}:
    raise ValueError("Unknown terrain curriculum mode")
  if "passage" in cfg.curriculum and mode != "manual":
    raise ValueError("定向越障已使用通过率门控，不能叠加 LP 或 uniform 课程")
  if mode == "manual":
    return
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("自动地形课程只适用于生成式崎岖地形；平地请选择手工模式")
  LearningProgressSampler(1, mode=mode, temperature=temperature,
                         uniform_mix=uniform_mix, min_episodes=min_episodes,
                         progress_alpha=progress_alpha, probability_alpha=probability_alpha)
  if not math.isfinite(max_task_probability) or not 0 < max_task_probability <= 1:
    raise ValueError("LP max_task_probability must be in (0, 1]")
  if not math.isfinite(update_interval_s) or update_interval_s <= 0:
    raise ValueError("LP update_interval_s must be positive")
  terrain.terrain_generator.curriculum = True
  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["terrain_sampling"] = CurriculumTermCfg(
    func=LPTerrainCurriculum,
    params={"mode": mode, "temperature": temperature, "uniform_mix": uniform_mix,
            "min_episodes": min_episodes, "update_interval_s": update_interval_s,
            "progress_alpha": progress_alpha, "probability_alpha": probability_alpha,
            "max_task_probability": max_task_probability,
            "status_path": str(status_path) if status_path else None},
  )
  for name, func in (("tracking_progress", blind_mdp.tracking_progress),
                     ("commanded_distance", blind_mdp.commanded_distance),
                     ("velocity_error_xy_mps", blind_mdp.velocity_error_xy)):
    cfg.metrics[name] = MetricsTermCfg(func=func)
