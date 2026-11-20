"""Directed passage objective and reset-time, outcome-gated curricula."""

from dataclasses import dataclass, asdict, field
import hashlib
import json
import math
import time
import warnings
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .auto_curriculum import (
  TrajectoryDifficultyScheduler,
  interpolate_bandwidth,
  trajectory_quality_score,
)
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


DEFAULT_SPEED_LIMIT = 1.0
# Legacy heading controller always clamped the yaw command to this magnitude.
LEGACY_YAW_LIMIT = .8
# ★ 2026-11-20：相对进展课程的升降级阈值（平均 route_progress_m / route_length_m）。
PROGRESS_UP_RATIO = .6
PROGRESS_DOWN_RATIO = .25


def terrain_column_kinds(generator, device=None):
  """Sub-terrain index owned by every terrain column."""
  proportions = torch.tensor([getattr(c, "proportion", 1.)
    for c in generator.sub_terrains.values()], device=device, dtype=torch.float32)
  cumulative = (proportions/proportions.sum()).cumsum(0)
  positions = (torch.arange(generator.num_cols, device=device, dtype=torch.float32)
               / generator.num_cols + .001)
  return torch.searchsorted(cumulative, positions, right=True)


# Terrain lessons the wall course can be trimmed to. ``LGDOG_WALL_LESSONS`` and
# the distillation single-lesson cfgs shrink ``sub_terrains`` to a subset of the
# ten lessons, so managers must recognise a name that WAS a lesson but is no
# longer part of the current terrain grid. The roster is imported lazily because
# the terrain config package imports the managers (a top-level import would
# close an import cycle).
_CANONICAL_TERRAIN_NAMES = None
_MISSING_TERRAIN_WARNED = set()
_CUT_SPEED_TERRAIN_WARNED = set()


def known_terrain_names():
  """All lesson names of the wall course, even when this terrain is trimmed."""
  global _CANONICAL_TERRAIN_NAMES
  if _CANONICAL_TERRAIN_NAMES is None:
    try:
      from ..config.lgdog1.passage_terrain_cfg import KINDS
      _CANONICAL_TERRAIN_NAMES = tuple(KINDS)
    except ImportError:  # pragma: no cover - the terrain config ships with the repo
      _CANONICAL_TERRAIN_NAMES = ()
  return _CANONICAL_TERRAIN_NAMES


def kind_columns_or_empty(env, kind):
  """Terrain columns owned by ``kind``; all False when the lesson is absent.

  A trimmed terrain (``LGDOG_WALL_LESSONS`` or a distillation single-lesson
  cfg) no longer contains every lesson, which used to crash managers through
  ``list.index``/``sub_terrains[kind]``. Such managers degrade instead: the
  returned mask has no True entry, so every per-environment lookup drawn from
  it is False and the related rule becomes a no-op.
  """
  generator = getattr(getattr(env.scene, 'terrain', None), 'cfg', None)
  generator = getattr(generator, 'terrain_generator', None)
  if generator is None:
    return torch.zeros(0, dtype=torch.bool, device=env.device)
  names = list(generator.sub_terrains)
  if kind not in names:
    if kind not in _MISSING_TERRAIN_WARNED:
      _MISSING_TERRAIN_WARNED.add(kind)
      warnings.warn(f'Terrain lesson {kind!r} is not part of the current terrain '
                    f'({names}); its rules and speed limits are disabled')
    return torch.zeros(generator.num_cols, dtype=torch.bool, device=env.device)
  proportions = torch.tensor([t.proportion for t in generator.sub_terrains.values()],
                             device=env.device)
  kinds = torch.searchsorted((proportions/proportions.sum()).cumsum(0),
    torch.arange(generator.num_cols, device=env.device)/generator.num_cols+.001, right=True)
  return kinds == names.index(kind)


def resolve_speed_groups(spec, column_names, kind_of_column, known_names=None):
  """Map each configured speed group to the terrain columns it owns.

  ``spec`` is the ``speed_groups`` mapping of group name to terrain-column
  names. ``None`` keeps the legacy single global gate. Terrain kinds left out
  of the mapping are collected into a ``default`` group so every column keeps
  a speed band and can still expand without waiting for the others.

  ``known_names`` is the full lesson roster (defaults to the wall course's
  ``KINDS``). A name that is listed in ``speed_groups`` but was trimmed from
  the current terrain is skipped with one warning instead of failing the whole
  task; a group whose members were all cut is dropped. A name that is not a
  lesson of the course at all is still a configuration error and raises
  ``ValueError``.
  """
  if spec is None:
    return None
  if not isinstance(spec, dict) or not spec:
    raise ValueError('speed_groups must be a non-empty mapping of group name to terrains')
  known = {name: index for index, name in enumerate(column_names)}
  roster = None
  owner = {}
  groups = {}
  skipped = []
  for group, members in spec.items():
    names = (members,) if isinstance(members, str) else tuple(members)
    if not names:
      raise ValueError(f'Speed group {group!r} has no terrains')
    kinds = []
    for name in names:
      if name not in known:
        if roster is None:
          roster = (tuple(known_names) if known_names is not None
                    else known_terrain_names())
        if name in roster:
          # The lesson exists in the course but was trimmed from this terrain.
          skipped.append(name)
          continue
        raise ValueError(f'Speed group {group!r} names unknown terrain {name!r}')
      if known[name] in owner:
        raise ValueError(f'Terrain {name!r} is in two speed groups')
      owner[known[name]] = group
      kinds.append(known[name])
    if not kinds:
      # Every member of this group was cut from the current terrain.
      continue
    groups[group] = torch.isin(kind_of_column,
      torch.tensor(kinds, dtype=torch.long, device=kind_of_column.device)).nonzero().flatten()
  if skipped:
    key = tuple(sorted(set(skipped)))
    if key not in _CUT_SPEED_TERRAIN_WARNED:
      _CUT_SPEED_TERRAIN_WARNED.add(key)
      warnings.warn(f'speed_groups name lessons cut from this terrain: {list(key)}; '
                    f'skipping them and keeping only the present lessons')
  missing = [index for index in range(len(column_names)) if index not in owner]
  if missing:
    unclaimed = torch.isin(kind_of_column,
      torch.tensor(missing, dtype=torch.long, device=kind_of_column.device)).nonzero().flatten()
    if len(unclaimed):
      name = 'default'
      while name in groups:
        name += '_'
      groups[name] = unclaimed
      warnings.warn(f'Terrains {[column_names[index] for index in missing]} are not in any '
                    f'speed group; expanding them as {name!r}')
  return groups


class PassageCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    generator = env.scene.terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    columns = terrain_column_kinds(generator, self.device)
    # Per-environment cruise band. The curriculum may widen or narrow a band
    # for one terrain group without touching the others.
    lower, upper = (float(value) for value in cfg.ranges.lin_vel_x)
    self.speed_floors = torch.full((self.num_envs,), lower, device=self.device)
    self.speed_limits = torch.full((self.num_envs,), upper, device=self.device)
    # ★ 2026-11-19 指令维课程：偏航与横向速度也各自拥有 per-env 带宽。默认
    # （cfg 字段为 None）完全保持旧行为：偏航仍按 ±0.8 裁剪，横向不额外裁剪。
    self.yaw_limits = torch.full((self.num_envs,), LEGACY_YAW_LIMIT, device=self.device)
    self.lateral_limits = None
    if getattr(cfg, 'yaw_max', None) is not None:
      self.yaw_limits.fill_(float(cfg.yaw_max))
    if getattr(cfg, 'vy_max', None) is not None:
      self.lateral_limits = torch.full((self.num_envs,), float(cfg.vy_max),
                                       device=self.device)
    self.column_names = names
    self.column_kinds = columns
    self.speed_group_columns = resolve_speed_groups(getattr(cfg, "speed_groups", None),
      names, columns)
    staged=getattr(cfg,'staged_waypoints',{})
    self.route_levels=generator.num_rows if staged else 1
    routes=[]
    for i in columns:
      name=names[int(i)];full=cfg.waypoints.get(name,((0.,0.),(cfg.goal_x,0.)))
      bank=staged.get(name,(full,)*self.route_levels)
      if len(bank)!=self.route_levels:raise ValueError('阶段路线必须覆盖所有地形行')
      routes.extend(bank)
    self.routes = torch.zeros((len(routes),max(map(len,routes)),2),device=self.device)
    self.last_waypoint = torch.tensor([len(r)-1 for r in routes],device=self.device)
    for i,route in enumerate(routes):
      if len(route)<2 or any(len(p)!=2 for p in route):
        raise ValueError("Passage routes need at least two xy points")
      self.routes[i,:len(route)] = torch.tensor(route,device=self.device)
      self.routes[i,len(route):] = self.routes[i,len(route)-1]
    lengths = torch.linalg.vector_norm(self.routes[:,1:]-self.routes[:,:-1],dim=-1)
    self.cumulative = torch.cat((torch.zeros((len(routes),1),device=self.device),lengths.cumsum(1)),1)
    self.waypoint_index = torch.ones(self.num_envs,device=self.device,dtype=torch.long)

  def route_keys(self):
    terrain=self._env.scene.terrain
    levels=getattr(self,'route_levels',1)
    return terrain.terrain_types*levels+terrain.terrain_levels if levels>1 else terrain.terrain_types

  def route_state(self):
    types = self.route_keys()
    index = self.waypoint_index
    start, target = self.routes[types,index-1], self.routes[types,index]
    delta = target-start
    length = torch.linalg.vector_norm(delta,dim=-1).clamp(min=1e-6)
    direction = delta/length[:,None]
    xy = self.local_xy()
    along = ((xy-start)*direction).sum(-1).clamp(min=0)
    projection = torch.minimum(along,length)
    lateral = torch.linalg.vector_norm(xy-start-projection[:,None]*direction,dim=-1)
    progress = self.cumulative[types,index-1]+projection
    return target, direction, lateral, progress

  def route_success(self):
    types = self.route_keys()
    final = self.routes[types,self.last_waypoint[types]]
    return ((self.waypoint_index == self.last_waypoint[types]) &
            (torch.linalg.vector_norm(self.local_xy()-final,dim=-1) <= self.cfg.waypoint_radius))

  def local_xy(self):
    return self.robot.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2]

  def column_mask(self, columns):
    """Environments whose terrain column is one of ``columns``."""
    columns = torch.as_tensor(columns, device=self.device, dtype=torch.long).flatten()
    if len(columns) == 0:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return torch.isin(self._env.scene.terrain.terrain_types, columns)

  def set_speed_limits(self, columns, low, high):
    """Set the cruise band of every environment on the given terrain columns."""
    low, high = float(low), float(high)
    if not low <= high:
      raise ValueError('Passage speed limits need low <= high')
    mask = self.column_mask(columns)
    self.speed_floors[mask] = low
    self.speed_limits[mask] = high
    self.speed[mask] = self.speed[mask].clamp(min=low, max=high)

  def set_global_speeds(self, low, high):
    """Set one cruise band for the whole batch (fixed-speed runs)."""
    low, high = float(low), float(high)
    if not low <= high:
      raise ValueError('Passage speed limits need low <= high')
    self.speed_floors.fill_(low)
    self.speed_limits.fill_(high)
    self.speed.clamp_(min=low, max=high)

  def set_group_speeds(self, group_name, low, high):
    """Set one configured ``speed_groups`` band by group name."""
    groups = getattr(self, 'speed_group_columns', None)
    if groups is None or group_name not in groups:
      raise KeyError(f'Unknown passage speed group {group_name!r}')
    self.set_speed_limits(groups[group_name], low, high)

  def set_command_limits(self, columns, yaw_max=None, vy_max=None):
    """Set per-environment yaw/lateral command bandwidth on terrain columns.

    ``yaw_max`` is a symmetric |yaw rate| cap and ``vy_max`` a symmetric body
    lateral speed cap.  ``None`` leaves that axis untouched.  The curriculum
    calls this whenever a terrain group finishes a successful speed window.
    """
    mask = self.column_mask(columns)
    if yaw_max is not None:
      yaw_max = float(yaw_max)
      if not yaw_max >= 0.:
        raise ValueError('Passage yaw bandwidth must be non-negative')
      self.yaw_limits[mask] = yaw_max
      self.vel_command_b[mask, 2] = self.vel_command_b[mask, 2].clamp(
        min=-yaw_max, max=yaw_max)
    if vy_max is not None:
      vy_max = float(vy_max)
      if not vy_max >= 0.:
        raise ValueError('Passage lateral bandwidth must be non-negative')
      if self.lateral_limits is None:
        self.lateral_limits = torch.full((self.num_envs,), torch.inf,
                                         device=self.device)
      self.lateral_limits[mask] = vy_max
      self.vel_command_b[mask, 1] = self.vel_command_b[mask, 1].clamp(
        min=-vy_max, max=vy_max)

  def set_global_command_limits(self, yaw_max=None, vy_max=None):
    """Set one yaw/lateral bandwidth for the whole batch."""
    self.set_command_limits(torch.arange(len(self.column_kinds), device=self.device),
                            yaw_max, vy_max)

  def _resample_command(self, env_ids):
    floor = self.speed_floors[env_ids]
    limit = self.speed_limits[env_ids].clamp(min=floor)
    fraction = torch.rand(len(env_ids), device=self.device)
    self.speed[env_ids] = floor + fraction*(limit-floor)
    self.is_heading_env[env_ids] = True
    self.is_standing_env[env_ids] = False
    self.heading_target[env_ids] = 0.
    self.waypoint_index[env_ids] = 1

  def _update_command(self):
    target, _, _, _ = self.route_state()
    types = self.route_keys()
    reached = torch.linalg.vector_norm(self.local_xy()-target,dim=-1) <= self.cfg.waypoint_radius
    self.waypoint_index += (reached & (self.waypoint_index < self.last_waypoint[types])).long()
    target, direction, _, _ = self.route_state()
    offset = target-self.local_xy()
    distance = torch.linalg.vector_norm(offset,dim=-1).clamp(min=1e-6)
    velocity = offset/distance[:,None] * torch.minimum(self.speed,(1.5*distance).clamp(min=.12))[:,None]
    heading = self.robot.data.heading_w
    self.vel_command_b[:,0] = velocity[:,0]*heading.cos()+velocity[:,1]*heading.sin()
    self.vel_command_b[:,1] = -velocity[:,0]*heading.sin()+velocity[:,1]*heading.cos()
    self.heading_target = torch.atan2(direction[:,1],direction[:,0])
    error = self.heading_target-heading
    self.heading_error = torch.atan2(error.sin(),error.cos())
    yaw = (.8*self.heading_error).clamp(-LEGACY_YAW_LIMIT, LEGACY_YAW_LIMIT)
    # Per-environment bandwidth caps only matter when the command curriculum
    # has widened/narrowed them; the defaults reproduce the legacy clamp.
    self.vel_command_b[:,2] = torch.maximum(torch.minimum(yaw, self.yaw_limits),
                                            -self.yaw_limits)
    if self.lateral_limits is not None:
      self.vel_command_b[:,1] = torch.maximum(
        torch.minimum(self.vel_command_b[:,1], self.lateral_limits),
        -self.lateral_limits)


def apply_level_quota(levels, start, fraction, generator=None):
  """★ 2026-11-20：满级地形只保留少量智能体，其余回落到低一级。

  ``levels`` 是每环境的地形等级（原地修改），从 ``start`` 起的等级视为"满级"。
  当满级环境数超过 ``fraction × 总数`` 时，随机把多出来的部分降到 ``start-1``，
  让样本回流到更平缓的等级，而不是把算力全压在最高难度上。返回被降级的数量。
  """
  if fraction >= 1. or start <= 0 or levels.numel() == 0:
    return 0
  elite = levels >= start
  n_elite = int(elite.sum())
  cap = int(float(fraction) * levels.numel())
  if n_elite <= cap:
    return 0
  idx = elite.nonzero(as_tuple=False).squeeze(-1)
  perm = torch.randperm(idx.numel(), device=levels.device)
  drop = idx[perm[:n_elite - cap]]
  levels[drop] = max(start - 1, 0)
  return int(drop.numel())


@dataclass(kw_only=True)
class PassageCommandCfg(UniformVelocityCommandCfg):
  goal_x: float = 5.3
  lane_half_width: float = .28
  # ★ 2026-11-19：按地形单独指定升级所需的连续成功次数（默认用全局 level_up_streak）。
  terrain_level_up_streak: dict[str, int] | None = None
  waypoint_radius: float = .15
  waypoints: dict[str, tuple[tuple[float,float],...]] = field(default_factory=dict)
  # ★ 2026-11-19: terrains of one group share a cruise band and expand together;
  # a stuck group (e.g. high_wall) no longer freezes every other terrain.
  # None keeps the old single global gate.
  speed_groups: dict[str, tuple[str, ...]] | None = None
  speed_group_floor: float = .35
  speed_group_ceiling: float = 2.
  # ★ 2026-11-19 指令维课程（TransCurriculum）：每一组在前向速度成功推进时，
  # 按推进比例把偏航/横向速度带宽从初始值提升到上限值。四个字段全部默认
  # None = 关闭：偏航沿用旧 ±0.8 裁剪，横向不做额外裁剪。
  yaw_max: float | None = None
  yaw_max_limit: float | None = None
  vy_max: float | None = None
  vy_max_limit: float | None = None

  def build(self, env):
    return PassageCommand(self, env)


def passage_success(env):
  term = env.command_manager.get_term("twist")
  if hasattr(term,"route_success"):
    return term.route_success()
  xy = term.local_xy()
  return (xy[:, 0] >= term.cfg.goal_x) & (xy[:, 1].abs() <= term.cfg.lane_half_width)


def left_lane(env):
  term = env.command_manager.get_term("twist")
  if hasattr(term,"route_state"):
    width=(term.lane_limits[env.scene.terrain.terrain_types]
           if hasattr(term,'lane_limits') else term.cfg.lane_half_width)
    return term.route_state()[2] > width
  xy = term.local_xy()
  return (xy[:, 1].abs() > term.cfg.lane_half_width) | (xy[:, 0] < -.5)


def failed(env):
  result = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  for name, cfg in env.cfg.terminations.items():
    if cfg is not None and not cfg.time_out and name != "passage_success":
      result |= env.termination_manager.get_term(name)
  return result


def failure_cost(env):
  return failed(env).float()


def success_bonus(env):
  # Fixed terminal bonus after reward manager's dt scaling, paid once by done.
  return (passage_success(env) & ~failed(env)).float() / env.step_dt


class ForwardProgress(ManagerTermBase):
  """Reward only new route distance; backtracking cannot farm reward."""
  def __init__(self, cfg, env):
    super().__init__(env)
    self.highwater = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.highwater[ids] = self._env.command_manager.get_term("twist").local_xy()[ids, 0]

  def __call__(self, env):
    command = env.command_manager.get_term("twist")
    xy = command.local_xy()
    x = command.route_state()[3] if hasattr(command,"route_state") else xy[:,0].clamp(max=command.cfg.goal_x)
    progress = (x-self.highwater).clamp(min=0) / env.step_dt
    self.highwater = torch.maximum(self.highwater, x)
    return progress.clamp(max=1.) * (~left_lane(env) & ~failed(env)).float()


class PassageCurriculum(ManagerTermBase):
  """Advance terrain only on successful crossings.

  With ``speed_groups`` configured, each terrain group owns its own cruise band
  and expands it on its own successful windows, so one unlearnable lesson (e.g.
  high_wall) only freezes its own group. Without it the old single global gate
  is kept unchanged.
  """
  def __init__(self, cfg, env):
    super().__init__(env)
    self.terrain = env.scene.terrain
    self.command = env.command_manager.get_term("twist")
    cols = self.terrain.cfg.terrain_generator.num_cols
    self.count = torch.zeros(cols, device=self.device)
    self.success = torch.zeros_like(self.count)
    self.error = torch.zeros_like(self.count)
    self.streak = 0
    self.level_success_streak = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
    self.level_failure_streak = torch.zeros_like(self.level_success_streak)
    self.params = dict(cfg.params)
    self.column_names = list(self.terrain.cfg.terrain_generator.sub_terrains)
    generator = self.terrain.cfg.terrain_generator
    self.grid = (generator.num_rows, generator.num_cols)
    command_contract = (asdict(self.command.cfg) if hasattr(self.command.cfg,"__dataclass_fields__") else
                        (self.command.cfg.goal_x,self.command.cfg.lane_half_width))
    if isinstance(command_contract,dict) and not command_contract.get('skill_training_profile'):
      command_contract.pop('skill_training_profile',None)
      command_contract.pop('slalom_use_lateral',None)
    if isinstance(command_contract,dict) and command_contract.get('obstacle_speed_limits') is None:
      command_contract.pop('obstacle_speed_limits',None)
    if isinstance(command_contract,dict) and not command_contract.get('staged_waypoints'):
      command_contract.pop('staged_waypoints',None)
    # New per-group defaults must not invalidate existing checkpoints.
    if isinstance(command_contract,dict):
      defaults = type(self.command.cfg)
      if command_contract.get('speed_groups') is None:
        command_contract.pop('speed_groups',None)
      for key in ('speed_group_floor','speed_group_ceiling'):
        if command_contract.get(key) == getattr(defaults,key,None):
          command_contract.pop(key,None)
      # ★ 2026-11-19 指令维课程字段同理：None = 关闭，旧 checkpoint 的指纹
      # 必须保持不变；只有真正配置了带宽才改变课程契约。
      for key in ('yaw_max','yaw_max_limit','vy_max','vy_max_limit'):
        if command_contract.get(key) is None:
          command_contract.pop(key,None)
    contract = (asdict(generator), env.step_dt, env.cfg.episode_length_s if hasattr(env.cfg,"episode_length_s") else None,
                command_contract)
    events = getattr(env.cfg, "events", {})
    if "himloco_pd_motor" in events:
      # Old mastery counts must not carry over to a different physics profile.
      randomization = tuple((name, events[name].params) for name in (
        "base_mass_inertia", "base_com", "foot_friction", "encoder_bias",
        "reset_robot_joints", "himloco_pd_motor", "himloco_push"))
      contract += (randomization, asdict(env.cfg.actions["joint_pos"]))
    self.fingerprint = hashlib.sha256(repr(contract).encode()).hexdigest()
    self.kind_of_column = terrain_column_kinds(generator, self.device)
    self.status_path = None
    self._status_last = 0.
    self.diagnostics=None
    if getattr(self.command.cfg,'skill_training_profile',{}).get('version')==5:
      from .passage_diagnostics import PassageDiagnostics
      self.diagnostics=PassageDiagnostics(env,self.column_names,self.kind_of_column)
    self.speed_groups = getattr(self.command.cfg, 'speed_groups', None)
    self.grouped = self.speed_groups is not None
    # ★ 2026-11-19 四项可开关的课程改进。解析只读 params（唯一例外：把不可跨
    # 进程比较的 safety_margin_fn 从 params 中移除），因此未启用时 state_dict、
    # params 与 fingerprint 都和旧 checkpoint 完全一致。
    self._configure_optional_curricula()
    self.groups = self._build_speed_groups()
    for group in self.groups.values():
      self._apply_speeds(group['columns'], group['low'], group['high'])
    if self.grouped:
      self._sync_command_range()
    self._configure_command_curriculum()
    self._build_trajectory_scheduler()
    self._apply_all_command_limits()

  # ------------------------------------------------------------------
  # Optional curriculum plumbing (all defaults reproduce the legacy task).
  # ------------------------------------------------------------------

  def _configure_optional_curricula(self):
    """Read the opt-in switches and validate their parameters."""
    params = self.params
    num_columns = len(self.count)
    # Feature 5: relative route-progress curriculum. When enabled it replaces the
    # per-episode success/quality gate for TERRAIN LEVELS only; speed groups,
    # quotas, resampling and command bandwidth keep running unchanged.
    self.progress_curriculum = bool(params.get('progress_curriculum', False))
    self.progress_count = torch.zeros(num_columns, device=self.device)
    self.progress_sum = torch.zeros(num_columns, device=self.device)
    self.progress_low_streak = torch.zeros(num_columns, dtype=torch.long,
                                           device=self.device)
    self.last_progress_ratio = {}
    self.last_progress_up = {}
    self.last_progress_down = {}
    # Feature 2: trajectory-level automatic curriculum.
    self.trajectory_curriculum = bool(params.get('trajectory_curriculum', False))
    self.trajectory_quality_kwargs = {
      'progress_weight': float(params.get('trajectory_progress_weight', .5)),
      'error_weight': float(params.get('trajectory_error_weight', .3)),
      'failure_weight': float(params.get('trajectory_failure_weight', .2)),
      'error_ref': float(params.get('trajectory_error_ref', .5)),
    }
    if self.trajectory_curriculum:
      # Probe once so an invalid weight/error_ref fails at construction time.
      trajectory_quality_score(0., 0., failed=0., **self.trajectory_quality_kwargs)
    self.trajectory_scheduler = None
    self.last_trajectory_decisions = {}
    # Feature 3: zero-success terrain column resampling.
    self.resample_windows = int(params.get('resample_zero_success_windows', 0))
    if self.resample_windows < 0:
      raise ValueError('resample_zero_success_windows must be non-negative')
    self.zero_success_windows = torch.zeros(num_columns, dtype=torch.long,
                                            device=self.device)
    self.resampled_counts = torch.zeros(num_columns, dtype=torch.long,
                                        device=self.device)
    self._resample_samples = torch.zeros(num_columns, device=self.device)
    self._resample_success = torch.zeros(num_columns, device=self.device)
    self._resample_last = {}
    # Environments demoted mid-episode keep their current tile until the next
    # reset; the origin is then refreshed before the reset event places them.
    self._resample_origin_pending = torch.zeros(self.num_envs, dtype=torch.bool,
                                                device=self.device)
    self._resetting_ids = None
    # Feature 4: predictive safety margin gate for every promotion.
    self.safety_margin_fn = params.get('safety_margin_fn')
    if self.safety_margin_fn is not None and not callable(self.safety_margin_fn):
      raise ValueError('safety_margin_fn must be callable')
    # Callables are not comparable across resume runs and cannot be meaningfully
    # checkpointed; the task config re-supplies the function on resume.
    params.pop('safety_margin_fn', None)
    self.safety_margin_threshold = float(params.get('safety_margin_threshold', .5))
    if not 0. <= self.safety_margin_threshold <= 1.:
      raise ValueError('safety_margin_threshold must be between 0 and 1')
    self.safety_sum = {}
    self.safety_count = {}

  def _configure_command_curriculum(self):
    """Validate the optional yaw/lateral bandwidth bands from the command cfg."""
    cfg = self.command.cfg
    yaw = getattr(cfg, 'yaw_max', None)
    yaw_limit = getattr(cfg, 'yaw_max_limit', None)
    vy = getattr(cfg, 'vy_max', None)
    vy_limit = getattr(cfg, 'vy_max_limit', None)
    self.yaw_band = None
    self.vy_band = None
    if yaw is not None:
      yaw = float(yaw)
      yaw_limit = float(yaw if yaw_limit is None else yaw_limit)
      if not (torch.isfinite(torch.tensor(yaw)) and torch.isfinite(torch.tensor(yaw_limit))
              and 0. <= yaw <= yaw_limit):
        raise ValueError('Passage yaw_max/yaw_max_limit need 0 <= yaw_max <= yaw_max_limit')
      self.yaw_band = (yaw, yaw_limit)
    elif yaw_limit is not None:
      raise ValueError('Passage yaw_max_limit requires yaw_max')
    if vy is not None:
      vy = float(vy)
      vy_limit = float(vy if vy_limit is None else vy_limit)
      if not (torch.isfinite(torch.tensor(vy)) and torch.isfinite(torch.tensor(vy_limit))
              and 0. <= vy <= vy_limit):
        raise ValueError('Passage vy_max/vy_max_limit need 0 <= vy_max <= vy_max_limit')
      self.vy_band = (vy, vy_limit)
    elif vy_limit is not None:
      raise ValueError('Passage vy_max_limit requires vy_max')
    self.command_curriculum = self.yaw_band is not None or self.vy_band is not None

  def _build_trajectory_scheduler(self):
    """Create the trajectory-quality scheduler when the feature is enabled."""
    if not self.trajectory_curriculum:
      return
    params = self.params
    min_samples = int(params.get('trajectory_min_samples',
                                 max(1, int(params.get('min_episodes', 1)))))
    self.trajectory_scheduler = TrajectoryDifficultyScheduler(
      list(self.groups),
      up_threshold=float(params.get('trajectory_up_threshold', .65)),
      down_threshold=float(params.get('trajectory_down_threshold', .35)),
      patience=int(params.get('trajectory_patience', 3)),
      down_patience=(int(params['trajectory_down_patience'])
                     if params.get('trajectory_down_patience') is not None else None),
      min_samples=min_samples)

  def _group_bounds(self):
    cfg = self.command.cfg
    floor = float(getattr(cfg, 'speed_group_floor', .35))
    ceiling = float(getattr(cfg, 'speed_group_ceiling', 2.))
    if floor > ceiling:
      raise ValueError('speed_group_floor must not exceed speed_group_ceiling')
    return floor, ceiling

  def _build_speed_groups(self):
    """Terrain columns and cruise band of every speed group."""
    cfg = self.command.cfg
    low, high = (float(value) for value in cfg.ranges.lin_vel_x)
    mapping = resolve_speed_groups(self.speed_groups, self.column_names, self.kind_of_column)
    if mapping is None:
      return {'all': {'columns': torch.arange(len(self.count), device=self.device),
                      'low': low, 'high': high, 'initial_high': high, 'streak': 0}}
    floor, ceiling = self._group_bounds()
    groups = {}
    for name, columns in mapping.items():
      group_low = max(low, floor)
      group_high = min(max(high, group_low), ceiling)
      groups[name] = {'columns': columns, 'low': group_low, 'high': group_high,
                      'initial_high': group_high, 'streak': 0}
    return groups

  def _apply_speeds(self, columns, low, high):
    """Push one cruise band to the command, per environment."""
    low, high = float(low), float(high)
    setter = getattr(self.command, 'set_speed_limits', None)
    if setter is None:
      self.command.cfg.ranges.lin_vel_x = (low, high)
    else:
      setter(columns, low, high)

  def _sync_command_range(self):
    """Keep the command's public range useful for logs, play and ONNX export."""
    groups = list(self.groups.values())
    if not groups:
      return
    self.command.cfg.ranges.lin_vel_x = (min(float(g['low']) for g in groups),
                                         max(float(g['high']) for g in groups))
    if not getattr(self, 'command_curriculum', False):
      return
    yaws, vys = [], []
    for group in groups:
      yaw, vy = self._group_command_bands(group)
      if yaw is not None:
        yaws.append(float(yaw))
      if vy is not None:
        vys.append(float(vy))
    if yaws:
      self.command.cfg.ranges.ang_vel_z = (-max(yaws), max(yaws))
    if vys:
      self.command.cfg.ranges.lin_vel_y = (-max(vys), max(vys))

  def _command_ceiling(self, group):
    """Forward-speed level at which a group's command bandwidth is fully open."""
    if self.grouped:
      return self._group_bounds()[1]
    return max(float(self.params.get('speed_limit', DEFAULT_SPEED_LIMIT)),
               float(group['high']))

  def _command_progress(self, group):
    """Fraction of the forward-speed expansion a group has finished."""
    ceiling = self._command_ceiling(group)
    span = ceiling - float(group.get('initial_high', group['high']))
    if span <= 0.:
      return 0.
    return min(max((float(group['high'])-float(group.get('initial_high', group['high'])))
                   / span, 0.), 1.)

  def _group_command_bands(self, group):
    """Current yaw/lateral bandwidth of one group (``None`` axis = disabled)."""
    if not getattr(self, 'command_curriculum', False):
      return None, None
    progress = self._command_progress(group)
    yaw = (interpolate_bandwidth(self.yaw_band[0], self.yaw_band[1], progress)
           if self.yaw_band is not None else None)
    vy = (interpolate_bandwidth(self.vy_band[0], self.vy_band[1], progress)
          if self.vy_band is not None else None)
    return yaw, vy

  def _apply_command_bands(self, columns, yaw, vy):
    """Push one group's yaw/lateral bandwidth to the command term."""
    if yaw is None and vy is None:
      return
    setter = getattr(self.command, 'set_command_limits', None)
    if setter is not None:
      setter(columns, yaw, vy)
    else:
      # Without per-environment support the public range still mirrors the
      # widest band, which is enough for logs and policy playback.
      if yaw is not None:
        self.command.cfg.ranges.ang_vel_z = (-float(yaw), float(yaw))
      if vy is not None:
        self.command.cfg.ranges.lin_vel_y = (-float(vy), float(vy))

  def _apply_group_command_bands(self, name):
    group = self.groups[name]
    yaw, vy = self._group_command_bands(group)
    self._apply_command_bands(group['columns'], yaw, vy)

  def _apply_all_command_limits(self):
    """Re-derive every group's yaw/lateral bandwidth from its speed progress."""
    if not getattr(self, 'command_curriculum', False):
      return
    for name in self.groups:
      self._apply_group_command_bands(name)
    self._sync_command_range()

  def _apply_fixed_speed(self, speed):
    for group in self.groups.values():
      group['low'] = group['high'] = float(speed)
      group['streak'] = 0
      self._apply_speeds(group['columns'], speed, speed)
    self.command.cfg.ranges.lin_vel_x = (speed, speed)
    setter = getattr(self.command, 'set_global_speeds', None)
    if setter is not None:
      setter(speed, speed)
    self._apply_all_command_limits()

  def write_distribution(self, force=False):
    """Write a throttled, atomic snapshot of all live environments."""
    if self.status_path is None or (not force and time.monotonic()-self._status_last<5.):
      return
    self._status_last=time.monotonic()
    rows=self.terrain.max_terrain_level
    kinds=self.kind_of_column[self.terrain.terrain_types]
    counts=torch.bincount(kinds*rows+self.terrain.terrain_levels,
      minlength=len(self.column_names)*rows).reshape(len(self.column_names),rows)
    counts=counts.detach().cpu().tolist()  # one compact device synchronization per snapshot
    data={'version':1,'updated':time.time(),'rows':rows,'total':sum(map(sum,counts)),
      'terrains':[{'name':name,'counts':values} for name,values in zip(self.column_names,counts)]}
    try:
      self.status_path.parent.mkdir(parents=True,exist_ok=True)
      temp=self.status_path.with_suffix('.tmp')
      temp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
      temp.replace(self.status_path)
      if getattr(self,'diagnostics',None) is not None:
        self.diagnostics.write(self.status_path.with_name('passage_diagnostics.json'))
    except OSError as error:
      if not getattr(self,'_status_warning',False):
        warnings.warn(f'Cannot write passage distribution: {error}')
        self._status_warning=True

  def __call__(self, env, env_ids, min_episodes=32, success_threshold=.7,
               max_error=.25, speed_limit=DEFAULT_SPEED_LIMIT, fixed_speed=None, speed_floor=.35,
               terrain_max_error='speed', level_up_streak=1, level_down_streak=1,
               level_quota_from=8, level_quota_fraction=1.0,
               progress_curriculum=False, trajectory_curriculum=False):
    if level_up_streak < 1 or level_down_streak < 1:
      raise ValueError('Terrain streak lengths must be positive')
    if fixed_speed is not None:
      from src.config.passage_speed import validate_fixed_speed
      fixed_speed=validate_fixed_speed(fixed_speed)
      self._apply_fixed_speed(fixed_speed)
    # Initial/randomized reset buffers are not completed episodes.
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=self.device)
    self._resetting_ids = ids.clone()
    if self.resample_windows > 0 and len(ids):
      pending = ids[self._resample_origin_pending[ids]]
      if len(pending):
        self.terrain.update_env_origins(
          pending, torch.zeros(len(pending), dtype=torch.bool, device=self.device),
          torch.zeros(len(pending), dtype=torch.bool, device=self.device))
        self._resample_origin_pending[pending] = False
    reset_mask = getattr(env, "reset_buf", None)
    ids = ids[:0] if reset_mask is None else ids[(env.episode_length_buf[ids] > 0) & reset_mask[ids]]
    # RSL-RL can randomize the first episode's length counter. Its artificial
    # timeout is not evidence that the robot failed a full passage attempt.
    actual_steps = env.metrics_manager._step_count
    ids = ids[actual_steps[ids] >= env.episode_length_buf[ids]]
    self._resample_last = {}
    self.last_trajectory_decisions = {}
    if len(ids):
      types = self.terrain.terrain_types[ids]
      passed = (passage_success(env) & ~failed(env))[ids]
      error_sum = env.metrics_manager._episode_sums["velocity_error_xy_mps"][ids]
      error = error_sum / env.episode_length_buf[ids].clamp(min=1)
      self.count.scatter_add_(0, types, torch.ones_like(error))
      self.success.scatter_add_(0, types, passed.float())
      self.error.scatter_add_(0, types, error)
      # Feature 4: accumulate the sensor-free safety proxy for this window.
      if self.safety_margin_fn is not None:
        self._observe_safety_margin(env, ids)
      # Reaching the top must stay at the top, not wrap to a random row.
      level = self.terrain.terrain_levels[ids]
      last = self.terrain.max_terrain_level-1
      # Obstacle mastery and cruise tracking are separate requirements. A
      # successful jump may involve deliberate braking; speed expansion below
      # still requires the original tracking gate across every column.
      threshold = max_error if terrain_max_error == 'speed' else terrain_max_error
      if self.diagnostics is not None:self.diagnostics.record(env,ids,passed)
      if self.progress_curriculum:
        # Feature 5: relative route progress decides terrain levels. The
        # trajectory scheduler, when also enabled, still observes its quality
        # windows (so its logs stay meaningful) but no longer moves levels.
        if self.trajectory_curriculum:
          self._observe_trajectory_windows(env, ids, passed)
        changed = self._apply_progress_levels(env, ids, min_episodes, level_down_streak)
        self.level_success_streak[changed] = 0
        self.level_failure_streak[changed] = 0
      elif self.trajectory_curriculum:
        # Feature 2: replace the per-env success streak with a trajectory-level
        # quality schedule. The scheduler applies its decision to this batch.
        changed = self._apply_trajectory_levels(env, ids, passed)
        self.level_success_streak[changed] = 0
        self.level_failure_streak[changed] = 0
      else:
        mastered = passed if threshold is None else passed & (error <= threshold)
        overrides = getattr(self.command.cfg, 'terrain_level_up_streak', None)
        if overrides:
          need = torch.tensor([int(overrides.get(self.column_names[int(t)], level_up_streak))
                               for t in types], device=self.device, dtype=torch.long)
        else:
          need = level_up_streak
        self.level_success_streak[ids] = torch.where(mastered, (self.level_success_streak[ids]+1).clamp(max=need), 0)
        self.level_failure_streak[ids] = torch.where(~passed, (self.level_failure_streak[ids]+1).clamp(max=level_down_streak), 0)
        up = mastered & (self.level_success_streak[ids] >= need) & (level < last)
        down = ~passed & (self.level_failure_streak[ids] >= level_down_streak) & (level > 0)
        # A small safety margin may veto promotions but never regressions:
        # pushing the lesson down is the recovery action of last resort.
        if self.safety_margin_fn is not None:
          up = up & self._safety_upgrade_mask(types)
        self.terrain.update_env_origins(ids, up, down)
        changed = ids[up | down]
        self.level_success_streak[changed] = 0
        self.level_failure_streak[changed] = 0
      # ★ 2026-11-20：满级（默认 ≥8，即 8 档课程的第 8 档）只保留少量智能体，
      # 其余回落到低一级，把样本回流到更平缓的等级。
      self._last_quota_demoted = apply_level_quota(
          self.terrain.terrain_levels, level_quota_from, level_quota_fraction)
      # Feature 3: a column with M zero-success windows is re-sampled downwards.
      if self.resample_windows > 0:
        self._observe_zero_success_columns(types, passed)
    rate = self.success / self.count.clamp(min=1)
    errors = self.error / self.count.clamp(min=1)
    represented = torch.bincount(self.kind_of_column, minlength=len(self.column_names)) > 0
    logs = {"speed_max": torch.tensor(self.command.cfg.ranges.lin_vel_x[1], device=self.device),
            "speed_min": torch.tensor(self.command.cfg.ranges.lin_vel_x[0], device=self.device),
            "speed_fixed": torch.tensor(float(fixed_speed is not None),device=self.device),
            "level_quota_demoted": torch.tensor(float(getattr(self, '_last_quota_demoted', 0)),
                                                 device=self.device),
            "level_mean": self.terrain.terrain_levels.float().mean()}
    limits=getattr(self.command.cfg,'obstacle_speed_limits',None)
    if limits is not None:
      logs['wall_speed_limit']=torch.tensor(limits[0],device=self.device)
      logs['slalom_speed_limit']=torch.tensor(limits[1],device=self.device)
    # Allocation is equal for the directed lessons; several columns may map to
    # one kind. Log reset-episode counts, rather than averaging per-step booleans.
    for i, name in enumerate(self.column_names):
      mask = self.kind_of_column == i
      n = self.count[mask].sum()
      logs[f"samples/{name}"] = n.clone()
      logs[f"success_rate/{name}"] = (self.success[mask].sum()/n.clamp(min=1)).clone()
      logs[f"tracking_error/{name}"] = (self.error[mask].sum()/n.clamp(min=1)).clone()
      env_mask = mask[self.terrain.terrain_types]
      total = env_mask.sum().clamp(min=1)
      levels = self.terrain.terrain_levels
      logs[f"level_mean/{name}"] = (levels.float()*env_mask).sum()/total
      logs[f"lowest_fraction/{name}"] = ((levels == 0) & env_mask).sum()/total
    if self.command_curriculum:
      for name, group in self.groups.items():
        yaw, vy = self._group_command_bands(group)
        if yaw is not None:
          logs[f'yaw_max/{name}'] = torch.tensor(float(yaw), device=self.device)
        if vy is not None:
          logs[f'vy_max/{name}'] = torch.tensor(float(vy), device=self.device)
    if self.trajectory_curriculum:
      for name in self.groups:
        quality = self.trajectory_scheduler.last_window_quality.get(name)
        if quality is not None:
          logs[f'quality/{name}'] = torch.tensor(float(quality), device=self.device)
        decision = self.last_trajectory_decisions.get(name, 0)
        logs[f'trajectory_up/{name}'] = torch.tensor(float(decision > 0), device=self.device)
        logs[f'trajectory_down/{name}'] = torch.tensor(float(decision < 0), device=self.device)
    if self.progress_curriculum:
      for name in self.column_names:
        logs[f'progress_ratio/{name}'] = torch.tensor(
          float(self.last_progress_ratio.get(name, 0.)), device=self.device)
        logs[f'progress_up/{name}'] = torch.tensor(
          float(self.last_progress_up.get(name, False)), device=self.device)
        logs[f'progress_down/{name}'] = torch.tensor(
          float(self.last_progress_down.get(name, False)), device=self.device)
    if self.resample_windows > 0:
      for name in self.column_names:
        logs[f'resampled/{name}'] = torch.tensor(
          float(self._resample_last.get(name, False)), device=self.device)
    # Missing obstacle kinds/samples never count as passed.
    if self.grouped:
      logs.update(self._expand_groups(rate, errors, min_episodes, success_threshold,
                                      max_error, fixed_speed))
    else:
      self._expand_global(represented, rate, errors, min_episodes, success_threshold,
                          max_error, speed_limit, speed_floor, fixed_speed)
    self.write_distribution()
    return logs

  def _expand_global(self, represented, rate, errors, min_episodes, success_threshold,
                     max_error, speed_limit, speed_floor, fixed_speed):
    """Legacy single gate: every column must pass before one +0.025 step."""
    if not (represented.all() and (self.count >= min_episodes).all()):
      return
    good = (rate >= success_threshold).all() & (errors <= max_error).all()
    bad = (rate < .35).any() | (errors > .4).any()
    self.streak = self.streak + 1 if good else 0
    low, high = self.command.cfg.ranges.lin_vel_x
    if fixed_speed is not None:
      self.streak=0
    elif self.streak >= 3:
      # Feature 4: only a safe window may consume the accumulated promotion.
      if self._safety_allows('all'):
        high = min(speed_limit, high+.025)
        self.streak = 0
    elif bad:
      high = max(speed_floor, low, high-.025)
    self.command.cfg.ranges.lin_vel_x = (low, high)
    self._apply_speeds(self.groups['all']['columns'], low, high)
    self.groups['all']['high'] = high
    self._apply_group_command_bands('all')
    self.count.zero_(); self.success.zero_(); self.error.zero_()
    self._reset_safety('all')

  def _expand_groups(self, rate, errors, min_episodes, success_threshold, max_error, fixed_speed):
    """Expand each terrain group on its own full window; never retract.

    A group only waits for its own columns, so a lesson that never succeeds
    (high_wall) freezes that group alone while the others keep speeding up.
    """
    ceiling = self._group_bounds()[1]
    for name, group in self.groups.items():
      columns = group['columns']
      if len(columns) == 0:
        continue
      counts = self.count[columns]
      if not bool((counts >= min_episodes).all()):
        continue  # this group's window is still filling
      good = bool(((rate[columns] >= success_threshold) & (errors[columns] <= max_error)).all())
      group['streak'] = group['streak'] + 1 if good else 0
      high = group['high']
      if fixed_speed is not None:
        group['streak'] = 0
      elif group['streak'] >= 3:
        # Feature 4: a safe window is required before the promotion is spent.
        # When blocked the streak is kept, so a later safe window promotes
        # immediately instead of waiting another three windows.
        if self._safety_allows(name):
          high = min(ceiling, high+.025)
          group['streak'] = 0
      if high != group['high']:
        group['high'] = high
        self._apply_speeds(columns, group['low'], high)
        # Feature 1: expand this group's yaw/lateral bandwidth in proportion
        # to the forward-speed progress it just unlocked.
        self._apply_group_command_bands(name)
      self.count[columns] = 0.
      self.success[columns] = 0.
      self.error[columns] = 0.
      self._reset_safety(name)
    self._sync_command_range()
    logs = {f'speed_max/{name}': torch.tensor(float(group['high']), device=self.device)
            for name, group in self.groups.items()}
    logs.update({f'speed_min/{name}': torch.tensor(float(group['low']), device=self.device)
                 for name, group in self.groups.items()})
    logs['speed_max'] = torch.tensor(float(self.command.cfg.ranges.lin_vel_x[1]),
                                     device=self.device)
    logs['speed_min'] = torch.tensor(float(self.command.cfg.ranges.lin_vel_x[0]),
                                     device=self.device)
    return logs

  # ------------------------------------------------------------------
  # Feature 4: predictive safety margin (optional gate on every promotion).
  # ------------------------------------------------------------------

  def _observe_safety_margin(self, env, ids):
    """Accumulate per-group safety margins for the completed batch."""
    values = self.safety_margin_fn(env, ids)
    values = torch.as_tensor(values, device=self.device, dtype=torch.float32).flatten()
    if (values.numel() != len(ids) or not torch.isfinite(values).all()
        or (values < 0.).any() or (values > 1.).any()):
      raise ValueError('safety_margin_fn must return one finite margin in [0, 1] per env')
    types = self.terrain.terrain_types[ids]
    for name, group in self.groups.items():
      mask = torch.isin(types, group['columns'])
      if bool(mask.any()):
        self.safety_sum[name] = self.safety_sum.get(name, 0.) + float(values[mask].sum())
        self.safety_count[name] = self.safety_count.get(name, 0) + int(mask.sum())

  def _safety_allows(self, name):
    """Whether the accumulated window margin clears the promotion threshold."""
    if self.safety_margin_fn is None:
      return True
    count = self.safety_count.get(name, 0)
    if count <= 0:
      return False  # no evidence of safety yet; never promote blindly
    return self.safety_sum.get(name, 0.)/count >= self.safety_margin_threshold

  def _safety_upgrade_mask(self, column_types):
    """Per-environment promotion mask derived from each group's margin."""
    mask = torch.zeros(len(column_types), dtype=torch.bool, device=self.device)
    for name, group in self.groups.items():
      if self._safety_allows(name):
        mask |= torch.isin(column_types, group['columns'])
    return mask

  def _reset_safety(self, name):
    self.safety_sum[name] = 0.
    self.safety_count[name] = 0

  # ------------------------------------------------------------------
  # Feature 2: trajectory-level automatic curriculum.
  # ------------------------------------------------------------------

  def _trajectory_progress_ratio(self, ids):
    """Fraction of the route a finished episode covered (0 start, 1 goal)."""
    command = self.command
    if hasattr(command, 'route_state') and hasattr(command, 'cumulative'):
      types = command.route_keys()[ids]
      final = command.last_waypoint[types]
      total = command.cumulative[types, final].clamp(min=1e-6)
      progress = command.route_state()[3][ids]
      return (progress/total).clamp(0., 1.)
    xy = command.local_xy()[ids]
    goal_x = float(getattr(command.cfg, 'goal_x', 5.3))
    if goal_x <= 0.:
      return torch.zeros(len(ids), device=self.device)
    return (xy[:, 0]/goal_x).clamp(0., 1.)

  def _trajectory_qualities(self, env, ids, passed):
    """Trajectory-quality score of every completed episode in ``ids``."""
    error = (env.metrics_manager._episode_sums['velocity_error_xy_mps'][ids]
             / env.episode_length_buf[ids].clamp(min=1))
    # A failed termination that still reports success is not a clean passage;
    # the penalty only applies to actual failures.
    failure = failed(env)[ids] & ~passed
    return trajectory_quality_score(
      self._trajectory_progress_ratio(ids), error,
      failed=failure.float(), **self.trajectory_quality_kwargs)

  def _observe_trajectory_windows(self, env, ids, passed):
    """Feature 2: feed one batch of completed-episode qualities to the scheduler."""
    quality = self._trajectory_qualities(env, ids, passed)
    types = self.terrain.terrain_types[ids]
    for name, group in self.groups.items():
      mask = torch.isin(types, group['columns'])
      if bool(mask.any()):
        self.trajectory_scheduler.observe(name, quality[mask])
    decisions = self.trajectory_scheduler.update()
    self.last_trajectory_decisions = dict(decisions)
    return decisions

  def _apply_trajectory_levels(self, env, ids, passed):
    """Feature 2: replace the per-env success streak with quality windows."""
    decisions = self._observe_trajectory_windows(env, ids, passed)
    types = self.terrain.terrain_types[ids]
    levels = self.terrain.terrain_levels[ids]
    last = self.terrain.max_terrain_level-1
    up = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
    down = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
    for name, decision in decisions.items():
      mask = torch.isin(types, self.groups[name]['columns'])
      if decision > 0:
        # Feature 4 also vetoes a quality-driven promotion.
        if self._safety_allows(name):
          up |= mask & (levels < last)
      elif decision < 0:
        down |= mask & (levels > 0)
    changed = up | down
    if bool(changed.any()):
      self.terrain.update_env_origins(ids[changed], up[changed], down[changed])
    return ids[changed]

  # ------------------------------------------------------------------
  # Feature 5: relative route-progress terrain curriculum.
  # ------------------------------------------------------------------

  def _apply_progress_levels(self, env, ids, min_episodes, level_down_streak):
    """Move every terrain column by its mean relative route progress.

    One window per column holds ``min_episodes`` completed episodes; its mean
    ``route_progress_m / route_length_m`` promotes the column above
    ``PROGRESS_UP_RATIO`` or demotes it after ``level_down_streak`` consecutive
    windows below ``PROGRESS_DOWN_RATIO``.  Only the environments resetting in
    this batch change tiles, exactly like the legacy success-rate gate, and a
    finished window always starts from zero.
    """
    min_episodes = max(int(min_episodes), 1)
    down_patience = max(int(level_down_streak), 1)
    types = self.terrain.terrain_types[ids]
    ratio = self._trajectory_progress_ratio(ids)
    self.progress_count.scatter_add_(0, types, torch.ones_like(ratio))
    self.progress_sum.scatter_add_(0, types, ratio)
    ready = self.progress_count >= min_episodes
    mean_ratio = self.progress_sum/self.progress_count.clamp(min=1)
    up_column = ready & (mean_ratio > PROGRESS_UP_RATIO)
    low_column = ready & (mean_ratio < PROGRESS_DOWN_RATIO)
    self.progress_low_streak = torch.where(
      low_column, self.progress_low_streak+1, self.progress_low_streak)
    self.progress_low_streak = torch.where(
      ready & ~low_column, torch.zeros_like(self.progress_low_streak),
      self.progress_low_streak)
    down_column = low_column & (self.progress_low_streak >= down_patience)
    # Log this call's windows before they are consumed.
    self.last_progress_ratio = {}
    self.last_progress_up = {}
    self.last_progress_down = {}
    for index, name in enumerate(self.column_names):
      mask = self.kind_of_column == index
      count = self.progress_count[mask].sum()
      if bool(count > 0.):
        self.last_progress_ratio[name] = float(
          self.progress_sum[mask].sum()/count)
      else:
        self.last_progress_ratio[name] = 0.
      self.last_progress_up[name] = bool(up_column[mask].any())
      self.last_progress_down[name] = bool(down_column[mask].any())
    levels = self.terrain.terrain_levels[ids]
    last = self.terrain.max_terrain_level-1
    up = up_column[types] & (levels < last)
    down = down_column[types] & (levels > 0)
    if self.safety_margin_fn is not None:
      # Feature 4 may veto a progress-driven promotion, never a regression.
      up = up & self._safety_upgrade_mask(types)
    changed = up | down
    if bool(changed.any()):
      self.terrain.update_env_origins(ids[changed], up[changed], down[changed])
    self.progress_count[ready] = 0.
    self.progress_sum[ready] = 0.
    self.progress_low_streak[up_column | down_column] = 0
    return ids[changed]

  # ------------------------------------------------------------------
  # Feature 3: zero-success terrain column resampling.
  # ------------------------------------------------------------------

  def _observe_zero_success_columns(self, types, passed):
    """Accumulate one reset batch and evaluate it as one resampling window."""
    samples = torch.bincount(types, minlength=len(self.count)).to(self.device)
    success = torch.bincount(types, weights=passed.float(), minlength=len(self.count))
    self._resample_samples += samples.float()
    self._resample_success += success.float()
    self.resample_zero_success_columns()

  def resample_zero_success_columns(self, window_samples=None, window_success=None):
    """Demote whole terrain columns after M consecutive zero-success windows.

    Called without arguments it consumes the batch accumulated by the last
    ``__call__``; passing explicit per-column sample/success vectors evaluates
    one window directly (monitoring code and unit tests use this).  A window
    with no samples on a column also counts as a zero-success window, as
    required for columns that receive no environments for a while.  Returns
    ``{terrain name: True}`` for every column demoted in this call.
    """
    if self.resample_windows <= 0:
      return {}
    if window_samples is None or window_success is None:
      samples, success = self._resample_samples, self._resample_success
      self._resample_samples = torch.zeros_like(self._resample_samples)
      self._resample_success = torch.zeros_like(self._resample_success)
    else:
      samples = torch.as_tensor(window_samples, device=self.device, dtype=torch.float32).flatten()
      success = torch.as_tensor(window_success, device=self.device, dtype=torch.float32).flatten()
      if (samples.shape != self.count.shape or success.shape != self.count.shape
          or not torch.isfinite(samples).all() or not torch.isfinite(success).all()
          or (samples < 0.).any() or (success < 0.).any()):
        raise ValueError('Invalid resampling window statistics')
    zero = (samples <= 0.) | (success <= 0.)
    self.zero_success_windows = torch.where(
      zero, self.zero_success_windows+1, torch.zeros_like(self.zero_success_windows))
    result = {}
    due = (self.zero_success_windows >= self.resample_windows).nonzero(as_tuple=False)
    for column in due.flatten().tolist():
      name = self.column_names[int(self.kind_of_column[column])]
      mask = (self.terrain.terrain_types == column) & (self.terrain.terrain_levels > 0)
      env_ids = mask.nonzero(as_tuple=False).flatten()
      if len(env_ids):
        # Book-keeping for the whole column, but only refresh the reset origin
        # of environments that are resetting this step; the rest keep the tile
        # they are standing on and get the new origin on their next reset.
        self.terrain.terrain_levels[env_ids] -= 1
        resetting = self._resetting_ids
        if resetting is not None and len(resetting):
          in_reset = torch.isin(env_ids, resetting)
        else:
          in_reset = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        refresh = env_ids[in_reset]
        deferred = env_ids[~in_reset]
        if len(refresh):
          self.terrain.update_env_origins(
            refresh, torch.zeros(len(refresh), dtype=torch.bool, device=self.device),
            torch.zeros(len(refresh), dtype=torch.bool, device=self.device))
        self._resample_origin_pending[deferred] = True
      self.resampled_counts[column] += 1
      self.zero_success_windows[column] = 0
      result[name] = True
    self._resample_last = result
    return result

  def _load_optional_curricula_state(self, state):
    """Restore the opt-in curriculum state; a missing key means \"start fresh\".

    Old checkpoints never stored these keys, so they load unchanged; the new
    features only add state when they are themselves enabled.
    """
    if self.trajectory_curriculum and self.trajectory_scheduler is not None:
      saved = state.get('trajectory_scheduler')
      if saved is not None:
        self.trajectory_scheduler.load_state_dict(saved)
    if self.resample_windows > 0:
      saved = state.get('resample')
      if saved is not None:
        restored = {}
        for key, target in (('windows', self.zero_success_windows),
                            ('counts', self.resampled_counts)):
          if saved.get(key) is None:
            raise ValueError('Invalid passage resampling state')
          value = torch.as_tensor(saved.get(key), device=self.device, dtype=torch.long).flatten()
          if value.shape != target.shape or (value < 0).any():
            raise ValueError('Invalid passage resampling state')
          restored[key] = value
        self.zero_success_windows.copy_(restored['windows'])
        self.resampled_counts.copy_(restored['counts'])
        pending = saved.get('pending')
        if pending is not None:
          pending = torch.as_tensor(pending, device=self.device).flatten().bool()
          if pending.shape != self._resample_origin_pending.shape:
            raise ValueError('Invalid passage resampling state')
          self._resample_origin_pending.copy_(pending)
    if self.progress_curriculum:
      saved = state.get('progress')
      if saved is not None:
        restored = {}
        for key, dtype in (('count', torch.float32), ('sum', torch.float32),
                           ('low_streak', torch.long)):
          if saved.get(key) is None:
            raise ValueError('Invalid passage progress state')
          restored[key] = torch.as_tensor(saved.get(key), device=self.device,
                                          dtype=dtype).flatten()
        count, total = restored['count'], restored['sum']
        if (count.shape != self.progress_count.shape
            or total.shape != count.shape
            or restored['low_streak'].shape != count.shape
            or not torch.isfinite(count).all() or not torch.isfinite(total).all()
            or (count < 0.).any() or (total < 0.).any()
            or (total > count+1e-6).any() or (restored['low_streak'] < 0).any()):
          raise ValueError('Invalid passage progress state')
        self.progress_count.copy_(count)
        self.progress_sum.copy_(total)
        self.progress_low_streak.copy_(restored['low_streak'])
    if self.safety_margin_fn is not None:
      saved = state.get('safety')
      if saved is not None:
        sums, counts = saved.get('sum', {}), saved.get('count', {})
        if set(sums) != set(self.groups) or set(counts) != set(self.groups):
          raise ValueError('Incompatible passage safety state')
        restored_sum, restored_count = {}, {}
        for name in self.groups:
          value, count = float(sums[name]), counts[name]
          if (not math.isfinite(value) or value < 0.
              or not isinstance(count, int) or count < 0):
            raise ValueError('Invalid passage safety state')
          restored_sum[name] = value
          restored_count[name] = count
        self.safety_sum = restored_sum
        self.safety_count = restored_count
    if self.command_curriculum:
      self._apply_all_command_limits()

  def state_dict(self):
    state = {"version": 1, "params": self.params, "names": self.column_names, "grid": self.grid,
      "ranges": tuple(self.command.cfg.ranges.lin_vel_x), "streak": self.streak,
      "fingerprint": self.fingerprint,
      **{k: getattr(self,k).detach().cpu().clone() for k in ("count","success","error")},
      **{k: getattr(self,k).detach().cpu().clone() for k in ("level_success_streak","level_failure_streak")},
      **({'diagnostics':self.diagnostics.state_dict()} if self.diagnostics is not None else {}),
      "levels": self.terrain.terrain_levels.detach().cpu().clone(),
      "types": self.terrain.terrain_types.detach().cpu().clone()}
    if self.grouped:
      state['speed_groups'] = {name: {'low': float(group['low']), 'high': float(group['high']),
                                      'streak': int(group['streak'])}
                               for name, group in self.groups.items()}
    # ★ 2026-11-19 新课程状态以可选键保存：旧 checkpoint 没有这些键也能加载，
    # 新 checkpoint 在对应功能关闭时也不会多写任何键。
    if self.command_curriculum:
      state['command_bands'] = {}
      for name, group in self.groups.items():
        yaw, vy = self._group_command_bands(group)
        state['command_bands'][name] = {'yaw': yaw, 'vy': vy}
    if self.trajectory_curriculum and self.trajectory_scheduler is not None:
      state['trajectory_scheduler'] = self.trajectory_scheduler.state_dict()
    if self.resample_windows > 0:
      state['resample'] = {
        'windows': self.zero_success_windows.detach().cpu().clone(),
        'counts': self.resampled_counts.detach().cpu().clone(),
        'pending': self._resample_origin_pending.detach().cpu().clone()}
    if self.progress_curriculum:
      state['progress'] = {
        'count': self.progress_count.detach().cpu().clone(),
        'sum': self.progress_sum.detach().cpu().clone(),
        'low_streak': self.progress_low_streak.detach().cpu().clone()}
    if self.safety_margin_fn is not None:
      state['safety'] = {'sum': dict(self.safety_sum), 'count': dict(self.safety_count)}
    return state

  def load_state_dict(self, state):
    saved_params = state.get("params")
    # Increasing only the ceiling preserves the meaning of existing samples,
    # streaks and difficulty levels. Keep the remaining contract checks strict.
    legacy_ceiling = (self.params.get("speed_limit") == DEFAULT_SPEED_LIMIT
                      and saved_params == {**self.params, "speed_limit": .6})
    if (state.get("version") != 1 or (saved_params != self.params and not legacy_ceiling)
        or state.get("names") != self.column_names or tuple(state.get("grid", ())) != self.grid
        or state.get("fingerprint") != self.fingerprint):
      raise ValueError("Incompatible passage curriculum state")
    values = {}
    for key in ("count", "success", "error"):
      tensor = torch.as_tensor(state[key], device=self.device)
      if tensor.shape != self.count.shape or not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("Invalid passage statistics")
      values[key] = tensor
    if (values["success"] > values["count"]).any():
      raise ValueError("Invalid passage successes")
    levels = torch.as_tensor(state["levels"], device=self.device, dtype=torch.long)
    types = torch.as_tensor(state["types"], device=self.device, dtype=torch.long)
    if (levels.shape != self.terrain.terrain_levels.shape or types.shape != levels.shape
        or (levels < 0).any() or (levels >= self.terrain.max_terrain_level).any()
        or (types < 0).any() or (types >= len(self.count)).any()):
      raise ValueError("Passage resume requires the same environment/terrain grid")
    low, high = state["ranges"]
    saved_ceiling = saved_params.get("speed_limit", DEFAULT_SPEED_LIMIT)
    fixed_speed=self.params.get('fixed_speed')
    if fixed_speed is not None:
      from src.config.passage_speed import validate_fixed_speed
      fixed_speed=validate_fixed_speed(fixed_speed)
    lower_bound=self.params.get('speed_floor',.25)
    if fixed_speed is not None:
      valid_range=(low,high)==(fixed_speed,fixed_speed)
    elif self.grouped:
      group_floor, group_ceiling = self._group_bounds()
      valid_range=(group_floor <= low <= high <= group_ceiling)
    else:
      valid_range=lower_bound <= low <= high <= saved_ceiling
    if not valid_range or not 0 <= int(state["streak"]) < 3:
      raise ValueError("Invalid passage speed range")
    saved_groups = state.get("speed_groups")
    if self.grouped:
      group_floor, group_ceiling = self._group_bounds()
      if fixed_speed is not None:
        group_floor = min(group_floor, fixed_speed)
        group_ceiling = max(group_ceiling, fixed_speed)
      if saved_groups is None:
        # Old checkpoints only know one global band; start every group there.
        groups = {name: (low, high, 0) for name in self.groups}
      else:
        if set(saved_groups) != set(self.groups):
          raise ValueError('Incompatible passage speed groups')
        groups = {}
        for name, entry in saved_groups.items():
          group_low, group_high = float(entry['low']), float(entry['high'])
          streak = int(entry['streak'])
          if not (group_floor <= group_low <= group_high <= group_ceiling) or not 0 <= streak < 3:
            raise ValueError('Invalid passage speed range')
          groups[name] = (group_low, group_high, streak)
    else:
      if saved_groups is not None:
        raise ValueError('Incompatible passage speed groups')
      groups = {}
    streaks = {}
    for key, limit in (("level_success_streak", self.params.get('level_up_streak', 1)),
                       ("level_failure_streak", self.params.get('level_down_streak', 1))):
      if limit > 1 and key not in state:
        raise ValueError('Missing terrain mastery streak')
      value = torch.as_tensor(state.get(key, torch.zeros_like(levels)), device=self.device)
      if (value.shape != levels.shape or not torch.isfinite(value).all()
          or (value < 0).any() or (value > limit).any() or (value != value.long()).any()):
        raise ValueError('Invalid terrain mastery streak')
      streaks[key] = value.long()
    if self.diagnostics is not None:
      if 'diagnostics' not in state:raise ValueError('Missing V5 passage diagnostics state')
      self.diagnostics.load_state_dict(state['diagnostics'])
    for key, value in values.items():
      getattr(self,key).copy_(value)
    for key, value in streaks.items():
      getattr(self,key).copy_(value)
    if self.grouped:
      for name, (group_low, group_high, streak) in groups.items():
        group = self.groups[name]
        group['low'], group['high'], group['streak'] = group_low, group_high, streak
        self._apply_speeds(group['columns'], group_low, group_high)
      self._sync_command_range()
    else:
      self.command.cfg.ranges.lin_vel_x = (low, high)
      self._apply_speeds(self.groups['all']['columns'], low, high)
      self.groups['all']['low'], self.groups['all']['high'] = low, high
    self.streak = int(state["streak"])
    self._load_optional_curricula_state(state)
    self.terrain.terrain_levels.copy_(levels); self.terrain.terrain_types.copy_(types)
    self.terrain.env_origins[:] = self.terrain.terrain_origins[levels, types]
