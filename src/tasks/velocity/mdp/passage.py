"""Directed passage objective and reset-time, outcome-gated curricula."""

from dataclasses import dataclass, asdict, field
import hashlib
import json
import time
import warnings
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


DEFAULT_SPEED_LIMIT = 1.0


def terrain_column_kinds(generator, device=None):
  """Sub-terrain index owned by every terrain column."""
  proportions = torch.tensor([getattr(c, "proportion", 1.)
    for c in generator.sub_terrains.values()], device=device, dtype=torch.float32)
  cumulative = (proportions/proportions.sum()).cumsum(0)
  positions = (torch.arange(generator.num_cols, device=device, dtype=torch.float32)
               / generator.num_cols + .001)
  return torch.searchsorted(cumulative, positions, right=True)


def resolve_speed_groups(spec, column_names, kind_of_column):
  """Map each configured speed group to the terrain columns it owns.

  ``spec`` is the ``speed_groups`` mapping of group name to terrain-column
  names. ``None`` keeps the legacy single global gate. Terrain kinds left out
  of the mapping are collected into a ``default`` group so every column keeps
  a speed band and can still expand without waiting for the others.
  """
  if spec is None:
    return None
  if not isinstance(spec, dict) or not spec:
    raise ValueError('speed_groups must be a non-empty mapping of group name to terrains')
  known = {name: index for index, name in enumerate(column_names)}
  owner = {}
  groups = {}
  for group, members in spec.items():
    names = (members,) if isinstance(members, str) else tuple(members)
    if not names:
      raise ValueError(f'Speed group {group!r} has no terrains')
    kinds = []
    for name in names:
      if name not in known:
        raise ValueError(f'Speed group {group!r} names unknown terrain {name!r}')
      if known[name] in owner:
        raise ValueError(f'Terrain {name!r} is in two speed groups')
      owner[known[name]] = group
      kinds.append(known[name])
    groups[group] = torch.isin(kind_of_column,
      torch.tensor(kinds, dtype=torch.long, device=kind_of_column.device)).nonzero().flatten()
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
    self.vel_command_b[:,2] = (.8*self.heading_error).clamp(-.8,.8)


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
    self.groups = self._build_speed_groups()
    for group in self.groups.values():
      self._apply_speeds(group['columns'], group['low'], group['high'])
    if self.grouped:
      self._sync_command_range()

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
                      'low': low, 'high': high, 'streak': 0}}
    floor, ceiling = self._group_bounds()
    groups = {}
    for name, columns in mapping.items():
      group_low = max(low, floor)
      group_high = min(max(high, group_low), ceiling)
      groups[name] = {'columns': columns, 'low': group_low, 'high': group_high, 'streak': 0}
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

  def _apply_fixed_speed(self, speed):
    for group in self.groups.values():
      group['low'] = group['high'] = float(speed)
      group['streak'] = 0
      self._apply_speeds(group['columns'], speed, speed)
    self.command.cfg.ranges.lin_vel_x = (speed, speed)
    setter = getattr(self.command, 'set_global_speeds', None)
    if setter is not None:
      setter(speed, speed)

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
               terrain_max_error='speed', level_up_streak=1, level_down_streak=1):
    if level_up_streak < 1 or level_down_streak < 1:
      raise ValueError('Terrain streak lengths must be positive')
    if fixed_speed is not None:
      from src.config.passage_speed import validate_fixed_speed
      fixed_speed=validate_fixed_speed(fixed_speed)
      self._apply_fixed_speed(fixed_speed)
    # Initial/randomized reset buffers are not completed episodes.
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=self.device)
    reset_mask = getattr(env, "reset_buf", None)
    ids = ids[:0] if reset_mask is None else ids[(env.episode_length_buf[ids] > 0) & reset_mask[ids]]
    # RSL-RL can randomize the first episode's length counter. Its artificial
    # timeout is not evidence that the robot failed a full passage attempt.
    actual_steps = env.metrics_manager._step_count
    ids = ids[actual_steps[ids] >= env.episode_length_buf[ids]]
    if len(ids):
      types = self.terrain.terrain_types[ids]
      passed = (passage_success(env) & ~failed(env))[ids]
      error_sum = env.metrics_manager._episode_sums["velocity_error_xy_mps"][ids]
      error = error_sum / env.episode_length_buf[ids].clamp(min=1)
      self.count.scatter_add_(0, types, torch.ones_like(error))
      self.success.scatter_add_(0, types, passed.float())
      self.error.scatter_add_(0, types, error)
      # Reaching the top must stay at the top, not wrap to a random row.
      level = self.terrain.terrain_levels[ids]
      last = self.terrain.max_terrain_level-1
      # Obstacle mastery and cruise tracking are separate requirements. A
      # successful jump may involve deliberate braking; speed expansion below
      # still requires the original tracking gate across every column.
      threshold = max_error if terrain_max_error == 'speed' else terrain_max_error
      if self.diagnostics is not None:self.diagnostics.record(env,ids,passed)
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
      self.terrain.update_env_origins(ids, up, down)
      changed = ids[up | down]
      self.level_success_streak[changed] = 0
      self.level_failure_streak[changed] = 0
    rate = self.success / self.count.clamp(min=1)
    errors = self.error / self.count.clamp(min=1)
    represented = torch.bincount(self.kind_of_column, minlength=len(self.column_names)) > 0
    logs = {"speed_max": torch.tensor(self.command.cfg.ranges.lin_vel_x[1], device=self.device),
            "speed_min": torch.tensor(self.command.cfg.ranges.lin_vel_x[0], device=self.device),
            "speed_fixed": torch.tensor(float(fixed_speed is not None),device=self.device),
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
      high = min(speed_limit, high+.025)
      self.streak = 0
    elif bad:
      high = max(speed_floor, low, high-.025)
    self.command.cfg.ranges.lin_vel_x = (low, high)
    self._apply_speeds(self.groups['all']['columns'], low, high)
    self.count.zero_(); self.success.zero_(); self.error.zero_()

  def _expand_groups(self, rate, errors, min_episodes, success_threshold, max_error, fixed_speed):
    """Expand each terrain group on its own full window; never retract.

    A group only waits for its own columns, so a lesson that never succeeds
    (high_wall) freezes that group alone while the others keep speeding up.
    """
    ceiling = self._group_bounds()[1]
    for group in self.groups.values():
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
        high = min(ceiling, high+.025)
        group['streak'] = 0
      if high != group['high']:
        group['high'] = high
        self._apply_speeds(columns, group['low'], high)
      self.count[columns] = 0.
      self.success[columns] = 0.
      self.error[columns] = 0.
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
    self.terrain.terrain_levels.copy_(levels); self.terrain.terrain_types.copy_(types)
    self.terrain.env_origins[:] = self.terrain.terrain_origins[levels, types]
