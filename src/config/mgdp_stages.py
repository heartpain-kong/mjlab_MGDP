"""Stage endpoints are part of the curriculum contract, never extra actor inputs."""


def staged_routes(command, generator):
  full=command.waypoints
  slalom=full['slalom_brick']
  routes={'slalom_brick':tuple(slalom[:n] for n in (6,8,10,12,14,len(slalom),len(slalom),len(slalom))),
          'high_wall':tuple(((0.,0.),(2.1,0.)) if row<2 else full['high_wall'] for row in range(8))}
  gap=generator.sub_terrains['gap_steps'];bank=[]
  for row in range(8):
    _,length,width,up,_=gap.gap_parameters((row+.5)/8)
    # Include 60cm of the 80cm connector so hind feet can finish the descent.
    first_end=2.+3*up+.8+width+.8+length
    goal=first_end+.6-1. if row<3 else full['gap_steps'][-1][0]
    bank.append(((0.,0.),(goal,0.)))
  routes['gap_steps']=tuple(bank)
  return routes


def configure_full_course_eval(cfg,level=6):
  """Explicit original-course acceptance, distinct from staged training replay."""
  command=cfg.commands.get('twist')
  if getattr(command,'skill_training_profile',{}).get('version')!=5:
    raise ValueError('完整赛道验收入口仅用于技能V5；旧模型请按原规则回放')
  if type(level)!=int or not 1<=level<=8:raise ValueError('验收难度必须为第1到8档')
  from src.config.mgdp_skill_training import SkillStonesTerrainCfg
  from src.tasks.velocity.config.lgdog1.course_v4_terrain_cfg import UnifiedWallTerrainCfg
  g=cfg.scene.terrain.terrain_generator;old=g.sub_terrains['high_wall']
  g.sub_terrains['high_wall']=UnifiedWallTerrainCfg(size=old.size,proportion=old.proportion,
    heights=old.heights,counts=(3,)*8)
  old=g.sub_terrains['stepping_stones']
  g.sub_terrains['stepping_stones']=SkillStonesTerrainCfg(size=old.size,proportion=old.proportion,
    counts=(6,)*8,entry_extensions=(0.,)*8)
  # ★ 2026-11-19：比赛验收改用官方 2×4 轮胎阵（2.2×1.1 m、高 165 mm），
  #   训练仍用铺满整格的版本，机器人无法从阵侧绕行。
  tire=g.sub_terrains.get('tire_array')
  if tire is not None and hasattr(tire,'competition_tires'):
    from dataclasses import replace
    g.sub_terrains['tire_array']=replace(tire,competition_tires=True)
  cfg.terminations['stone_ground_contact'].params['exit_by_level']=(5.9,)*8
  command.staged_waypoints={}
  cfg.scene.terrain.max_init_terrain_level=level-1
  cfg.events.pop('randomize_terrain',None)


def pin_full_course_level(env,level):
  import torch
  terrain=env.scene.terrain
  if type(level)!=int or not 1<=level<=terrain.terrain_origins.shape[0]:
    raise ValueError('验收难度超出地图行数')
  with torch.inference_mode():
    terrain.terrain_levels.fill_(level-1)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    env.reset()
