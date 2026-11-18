import os
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
from torch import nn

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.him_contract import HIM_HISTORY_ORDER
from src.tasks.velocity.mdp.curriculums import RewardBasedVelocityCurriculum
from src.tasks.velocity.mdp.lp_terrain import LPTerrainCurriculum
from src.tasks.velocity.mdp.passage import PassageCurriculum
from src.tasks.velocity.mdp.symmetry import (
  compute_symmetric_states,
  mirror_actor_obs,
)


# ---------------------------------------------------------------------------
# Optional reward-weight curriculum (WMP ``reward_curriculum`` port).
#
# WMP 用 ``reward_curriculum_schedule`` 在训练轮次之间线性改变某个奖励项的
# 系数；这里保留同样的语义，但作用在 mjlab 的 ``RewardTermCfg.weight`` 上。
# 默认关闭：``PPOHIMRunner.reward_weight_curriculum`` 为空元组。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeightSchedule:
  """One reward term's linear ramp: ``start`` → ``end`` over the given iterations."""

  term: str
  start: float
  end: float
  start_iter: int
  end_iter: int

  def __post_init__(self):
    if self.end_iter <= self.start_iter:
      raise ValueError(
        f"reward curriculum for {self.term!r} needs end_iter > start_iter, "
        f"got {self.start_iter} → {self.end_iter}"
      )


def linear_reward_weight(
  iteration: int, start: float, end: float, start_iter: int, end_iter: int
) -> float:
  """当前轮次应有的权重；区间外分别夹到 start / end。

  ``iteration <= start_iter`` 返回 ``start``，``>= end_iter`` 返回 ``end``，
  中间线性插值。WMP 的公式没有 clamp，这里夹住是为了防止负数轮次或续训
  越界把权重外推。
  """
  if end_iter <= start_iter:
    raise ValueError(
      f"end_iter must be greater than start_iter, got {start_iter} → {end_iter}"
    )
  progress = (float(iteration) - float(start_iter)) / float(end_iter - start_iter)
  progress = min(max(progress, 0.0), 1.0)
  return float(start) + (float(end) - float(start)) * progress


class RewardWeightCurriculum:
  """按训练轮次线性调整一组奖励项的权重。

  mjlab 的 ``RewardManager.compute`` 每个控制步都会重新读
  ``RewardTermCfg.weight``，而 ``get_term_cfg`` 返回的就是 manager 内部那份
  配置对象，所以直接赋值即可生效，不需要重建 manager。权重按【绝对】轮次
  计算，续训（iteration 从 checkpoint 接着走）也能落回正确的值。
  """

  def __init__(self, reward_manager, schedules):
    self._manager = reward_manager
    self._schedules = tuple(schedules)
    if not self._schedules:
      raise ValueError("reward curriculum needs at least one schedule")
    for schedule in self._schedules:
      if not isinstance(schedule, RewardWeightSchedule):
        raise TypeError(f"expected RewardWeightSchedule, got {type(schedule)!r}")
      reward_manager.get_term_cfg(schedule.term)  # 打错名字时在训练开始前失败
    self._last_logged: dict[str, float] = {}

  def weights(self, iteration: int) -> dict[str, float]:
    return {
      schedule.term: linear_reward_weight(
        iteration, schedule.start, schedule.end, schedule.start_iter, schedule.end_iter
      )
      for schedule in self._schedules
    }

  def update(self, iteration: int) -> dict[str, float]:
    """把各奖励项权重设为 ``iteration`` 对应的值并返回它们。"""
    weights = self.weights(iteration)
    for term, weight in weights.items():
      self._manager.get_term_cfg(term).weight = weight
      previous = self._last_logged.get(term)
      if previous is None or abs(previous - weight) >= 0.01:
        self._last_logged[term] = weight
        print(f"[reward-curriculum] iter={iteration} {term}={weight:.4f}", flush=True)
    return weights


def _symmetric_ppo_update(alg, augment):
  """Return a ``PPO.update`` replacement that trains on original + mirrored data.

  Only the policy (surrogate) loss sees the doubled batch.  The value loss is
  computed once on the *original* mini-batch, whose critic observations are
  never mirrored.  The rest of the loop mirrors rsl-rl 5.0.1's ``PPO.update``
  (adaptive KL, gradient clipping, loss logging).
  """

  def update() -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    generator = alg.storage.mini_batch_generator(
      alg.num_mini_batches, alg.num_learning_epochs
    )
    for batch in generator:
      original_obs = batch.observations
      original_batch_size = int(batch.actions.shape[0])

      if alg.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      # Keep the original tensors before repeating them for the doubled batch.
      original_log_prob = batch.old_actions_log_prob
      original_values = batch.values
      original_returns = batch.returns

      obs_aug, actions_aug = augment(obs=original_obs, actions=batch.actions)
      num_aug = int(actions_aug.shape[0]) // original_batch_size
      if num_aug != 2 or actions_aug.shape[0] != 2 * original_batch_size:
        raise RuntimeError(
          "symmetry augmentation must double the mini-batch, got "
          f"{actions_aug.shape[0]} rows for {original_batch_size} originals"
        )
      batch.observations = obs_aug
      batch.actions = actions_aug
      batch.old_actions_log_prob = original_log_prob.repeat(num_aug, 1)
      batch.advantages = batch.advantages.repeat(num_aug, 1)

      # Policy forward on [original, mirrored] actor observations.
      alg.actor(batch.observations, stochastic_output=True)
      actions_log_prob = alg.actor.get_output_log_prob(batch.actions)
      # KL and entropy use the original half only (same as rsl-rl symmetry).
      distribution_params = tuple(
        p[:original_batch_size] for p in alg.actor.output_distribution_params
      )
      entropy = alg.actor.output_entropy[:original_batch_size]

      if alg.desired_kl is not None and alg.schedule == "adaptive":
        with torch.inference_mode():
          kl = alg.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if alg.is_multi_gpu:
            torch.distributed.all_reduce(
              kl_mean, op=torch.distributed.ReduceOp.SUM
            )
            kl_mean /= alg.gpu_world_size
          if alg.gpu_global_rank == 0:
            if kl_mean > alg.desired_kl * 2.0:
              alg.learning_rate = max(1e-5, alg.learning_rate / 1.5)
            elif kl_mean < alg.desired_kl / 2.0 and kl_mean > 0.0:
              alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
          if alg.is_multi_gpu:
            lr_tensor = torch.tensor(alg.learning_rate, device=alg.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            alg.learning_rate = lr_tensor.item()
          for param_group in alg.optimizer.param_groups:
            param_group["lr"] = alg.learning_rate

      ratio = torch.exp(
        actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
      )
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value loss: original mini-batch only (critic data is never mirrored).
      values = alg.critic(original_obs)
      if alg.use_clipped_value_loss:
        value_clipped = original_values + (values - original_values).clamp(
          -alg.clip_param, alg.clip_param
        )
        value_losses = (values - original_returns).pow(2)
        value_losses_clipped = (value_clipped - original_returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (original_returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + alg.value_loss_coef * value_loss
        - alg.entropy_coef * entropy.mean()
      )

      alg.optimizer.zero_grad()
      loss.backward()
      if alg.is_multi_gpu:
        alg.reduce_parameters()
      nn.utils.clip_grad_norm_(alg.actor.parameters(), alg.max_grad_norm)
      nn.utils.clip_grad_norm_(alg.critic.parameters(), alg.max_grad_norm)
      alg.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

    num_updates = alg.num_learning_epochs * alg.num_mini_batches
    alg.storage.clear()
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
    }

  return update


def _enable_symmetry_augmentation(runner) -> None:
  """Activate the PPO mirror augmentation when ``cfg.symmetry_augment`` is set."""
  if not bool(runner.cfg.get("symmetry_augment", False)):
    return
  alg = runner.alg
  if getattr(alg, "rnd", None) is not None:
    raise ValueError("symmetry_augment cannot be combined with RND")
  if getattr(alg, "symmetry", None) is not None:
    raise ValueError(
      "symmetry_augment conflicts with rsl-rl's built-in symmetry configuration"
    )
  if getattr(alg.actor, "is_recurrent", False) or getattr(
    alg.critic, "is_recurrent", False
  ):
    raise ValueError("symmetry_augment does not support recurrent policies")
  if hasattr(alg.actor, "estimator"):
    raise ValueError(
      "symmetry_augment is not supported for HIM actors; run a plain PPO task"
    )

  actor_obs = runner.env.get_observations()["actor"]
  if actor_obs.dim() != 2:
    raise ValueError(
      "symmetry_augment expects flattened [batch, obs_dim] actor observations, "
      f"got shape {tuple(actor_obs.shape)}"
    )
  # Fail fast if the observation layout is not the LgDog1 47-wide frame.
  mirror_actor_obs(actor_obs[:1])

  alg.update = _symmetric_ppo_update(alg, compute_symmetric_states)
  print(
    "[INFO] symmetry augmentation enabled: PPO policy loss uses original + "
    f"left-right mirrored actor observations ({actor_obs.shape[-1]}-dim) and "
    f"{runner.env.num_actions}-dim actions (batch x2); value loss stays on the "
    "original mini-batch",
    flush=True,
  )


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    _enable_symmetry_augmentation(self)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class PPOHIMRunner(MjlabOnPolicyRunner):
  """One LgDog PPO entry point with optional HIM history identification.

  Plain PPO actors keep the normal single-frame contract.  HIM actors expose
  ``estimator`` and export their configured history length.  The runner is shared
  so checkpoint/export handling stays in one place while incompatible weights
  remain rejected by the model and algorithm contracts.
  """
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    _enable_symmetry_augmentation(self)

  # 可选奖励权重课程，默认关闭（空元组）。启用示例（feet_air_time 0.1 → 0.25，
  # 3000 → 8000 轮）：
  #   reward_weight_curriculum = (
  #     RewardWeightSchedule("feet_air_time", 0.1, 0.25, 3000, 8000),
  #   )
  # 注意：这会覆盖 profile 里的权重；第一条 schedule 的 start 就是训练开始时的值。
  reward_weight_curriculum: tuple[RewardWeightSchedule, ...] = ()

  def _reward_weight_curriculum(self):
    schedules = tuple(self.reward_weight_curriculum)
    if not schedules:
      return None
    manager = getattr(getattr(self.env, "unwrapped", None), "reward_manager", None)
    if manager is None:
      warnings.warn(
        "reward_weight_curriculum is set but the env has no reward_manager; skipping"
      )
      return None
    return RewardWeightCurriculum(manager, schedules)

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    curriculum = self._reward_weight_curriculum()
    if curriculum is None:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    # rsl-rl 的 learn() 每轮只通过 logger.log(it=...) 暴露迭代号；在它之前把
    # 权重设成下一轮该用的值，保证 rollout 用的就是本轮应有的权重。
    curriculum.update(self.current_learning_iteration)
    base_log = self.logger.log

    def log_with_reward_curriculum(*args, **kwargs):
      iteration = kwargs.get("it", self.current_learning_iteration)
      curriculum.update(iteration + 1)
      return base_log(*args, **kwargs)

    self.logger.log = log_with_reward_curriculum
    try:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.logger.log = base_log

  @property
  def is_him(self):
    return hasattr(self.alg.actor, "estimator")

  def _velocity_curriculum(self):
    manager = getattr(getattr(self.env, "unwrapped", None), "curriculum_manager", None)
    if manager is None or "command_vel" not in getattr(manager, "active_terms", ()):
      return None
    term = manager.get_term_cfg("command_vel").func
    return term if isinstance(term, RewardBasedVelocityCurriculum) else None

  def _terrain_curriculum(self):
    manager = getattr(getattr(self.env, "unwrapped", None), "curriculum_manager", None)
    if manager is None or "terrain_sampling" not in getattr(manager, "active_terms", ()):
      return None
    term = manager.get_term_cfg("terrain_sampling").func
    return term if isinstance(term, LPTerrainCurriculum) else None

  def _passage_curriculum(self):
    manager = getattr(getattr(self.env, "unwrapped", None), "curriculum_manager", None)
    if manager is None or "passage" not in manager.active_terms:
      return None
    term = manager.get_term_cfg("passage").func
    return term if isinstance(term, PassageCurriculum) else None

  def save(self, path: str, infos=None):
    infos = {**(infos or {})}
    if getattr(self, "_mgdp_warm_start", None) is not None:
      infos["mgdp_warm_start"] = self._mgdp_warm_start
    from src.mgdp_checkpoint import save_encoder_contract
    perception = save_encoder_contract(self.env.unwrapped, Path(path).parent)
    if perception is not None:
      infos["mgdp_perception"] = perception
    passage = self._passage_curriculum()
    if perception is not None or passage is not None:
      from src.config.mgdp_vision import vision_mode
      infos['mgdp_vision_mode']=vision_mode(self.env.unwrapped.cfg)
      from src.config.mgdp_course_contract import current_contract
      infos['mgdp_course_contract']=current_contract(self.env.unwrapped.cfg)
      from src.config.mgdp_course_contract import obstacle_speed_contract
      infos['mgdp_obstacle_speed']=obstacle_speed_contract(self.env.unwrapped.cfg)
      from src.config.mgdp_skill_training import skill_contract
      infos['mgdp_skill_training']=skill_contract(self.env.unwrapped.cfg)
    passage = self._passage_curriculum()
    if passage is not None:
      infos["mgdp_passage_curriculum"] = passage.state_dict()
      from src.config.mgdp_course_contract import rotation_stall_contract
      infos['mgdp_rotation_stall']=rotation_stall_contract(self.env.unwrapped.cfg)
    from src.config.tire_guidance import guidance_contract
    guidance=guidance_contract(self.env.unwrapped.cfg)
    if guidance is not None:
      infos['mgdp_tire_hole_guidance']=guidance
    curriculum = self._velocity_curriculum()
    if curriculum is not None:
      infos["him_velocity_curriculum"] = curriculum.state_dict()
    terrain_curriculum = self._terrain_curriculum()
    if terrain_curriculum is not None:
      infos["terrain_task_curriculum"] = terrain_curriculum.state_dict()
      terrain_curriculum.write_status()
    super().save(path, infos)
    policy_dir = Path(path).parent
    self.export_policy_to_onnx(str(policy_dir), "policy.onnx")
    try:
      self.export_policy_to_jit(str(policy_dir), "policy_jit.pt")
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
      print(f"[WARN] TorchScript export skipped: {type(error).__name__}: {error}")
    if self.logger.logger_type in ["wandb"]:
      wandb.save(str(policy_dir / "policy.onnx"), base_path=str(policy_dir.parent))
      jit_path = policy_dir / "policy_jit.pt"
      if jit_path.is_file():
        wandb.save(str(jit_path), base_path=str(policy_dir.parent))

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    passage = self._passage_curriculum()
    has_perception = "perception" in getattr(getattr(self.env.unwrapped, "cfg", None), "observations", {})
    if passage is not None or has_perception:
      checkpoint = torch.load(path, map_location="cpu", weights_only=False)
      from src.config.tire_guidance import validate_tire_guidance_resume
      validate_tire_guidance_resume(self.env.unwrapped.cfg,checkpoint)
      if has_perception or passage is not None:
        from src.config.mgdp_vision import validate_vision_resume
        validate_vision_resume(self.env.unwrapped.cfg,checkpoint)
        from src.config.mgdp_skill_training import validate_skill_resume
        validate_skill_resume(self.env.unwrapped.cfg,checkpoint)
        from src.config.mgdp_course_contract import validate_resume
        validate_resume(self.env.unwrapped.cfg,checkpoint)
        from src.config.mgdp_course_contract import validate_obstacle_speed_resume
        validate_obstacle_speed_resume(self.env.unwrapped.cfg,checkpoint)
      if has_perception:
        from src.mgdp_checkpoint import validate_runtime_encoder
        validate_runtime_encoder(self.env.unwrapped, checkpoint)
    if passage is not None:
      from src.config.mgdp_course_contract import validate_rotation_stall_resume
      validate_rotation_stall_resume(self.env.unwrapped.cfg,checkpoint)
      state = checkpoint.get("infos", {}).get("mgdp_passage_curriculum")
      if state is None:
        raise ValueError("定向越障 v2 不能完整续训旧 MGDP：缺少通过率课程状态，请创建新训练")
      passage.load_state_dict(state)
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "mgdp_warm_start" in infos:
      self._mgdp_warm_start = infos["mgdp_warm_start"]
    if passage is not None:
      with torch.inference_mode():
        self.env.reset()
    curriculum = self._velocity_curriculum()
    state = (infos or {}).get("him_velocity_curriculum") if isinstance(infos, dict) else None
    if curriculum is not None:
      if state is not None:
        curriculum.load_state_dict(state)
      else:
        self._restore_legacy_velocity_ranges(curriculum, path)
      # Refresh command samples after restoring mutable curriculum ranges.
      with torch.inference_mode():
        self.env.reset()
    terrain_curriculum = self._terrain_curriculum()
    if terrain_curriculum is not None:
      state = (infos or {}).get("terrain_task_curriculum") if isinstance(infos, dict) else None
      if state is not None:
        terrain_curriculum.load_state_dict(state)
      else:
        warnings.warn("Checkpoint has no terrain sampler state; starting fresh LP statistics")
      with torch.inference_mode():
        self.env.reset()
    return infos

  @staticmethod
  def _restore_legacy_velocity_ranges(curriculum, checkpoint_path):
    """Recover ranges from old env.yaml checkpoints without curriculum state."""
    try:
      import yaml
      from src.utils.training_logs import ConfigLoader
      config_path = Path(checkpoint_path).resolve().parent / "params" / "env.yaml"
      if config_path.is_file():
        saved_cfg = yaml.load(config_path.read_text(), Loader=ConfigLoader)
        if isinstance(saved_cfg, dict) and curriculum.restore_legacy_ranges(saved_cfg):
          return
    except Exception:
      pass
    warnings.warn(
      "Checkpoint has no velocity curriculum state or saved command ranges; "
      "using the configured initial ranges."
    )

  def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
    super().export_policy_to_onnx(path, filename, verbose)
    if self.is_him:
      metadata = get_base_metadata(self.env.unwrapped, "lgdog_him")
      metadata.update({
        "policy_architecture": "lgdog_him_v1",
        "history_length": str(self.alg.actor.history_length),
        "frame_dim": str(self.alg.actor.frame_dim),
        "history_order": HIM_HISTORY_ORDER,
        "history_reset": "repeat_first_frame",
        "control_dt": str(self.env.unwrapped.step_dt),
      })
    elif "perception" in self.env.unwrapped.cfg.observations:
      import json
      from src.mgdp_checkpoint import sensing_contract
      from src.mgdp_deployment import file_digest
      state = getattr(self.env.unwrapped, "_mgdp_perception_state", None)
      if state is None:
        raise RuntimeError("MGDP perception has not been initialized before export")
      sidecar = Path(path) / (Path(filename).stem + ".perception.pt")
      state["module"].save(sidecar)
      metadata = get_base_metadata(self.env.unwrapped, "lgdog_mgdp")
      metadata.update({
        "policy_architecture": "lgdog_mgdp_v1",
        "history_length": str(self.env.unwrapped.cfg.observations["actor"].history_length),
        "frame_dim": "47", "perception_dim": str(state["module"].cfg.gru_hidden),
        "history_order": "term_major_oldest_to_newest",
        "perception_checkpoint": sidecar.name,
        "perception_sha256": file_digest(sidecar),
        "mgdp_sensing": json.dumps(sensing_contract(self.env.unwrapped.cfg), sort_keys=True),
        "mgdp_inference_precision": self.env.unwrapped.cfg.observations["perception"].terms["depth_feature"].params.get("inference_precision", "fp32"),
        "control_dt": str(self.env.unwrapped.step_dt),
      })
      if state['module'].camera is not None:
        metadata['mgdp_camera']=json.dumps(state['module'].camera,sort_keys=True)
      passage=self._passage_curriculum()
      if passage is not None and passage.params.get('fixed_speed') is not None:
        metadata['mgdp_passage_fixed_speed']=str(passage.params['fixed_speed'])
      from src.config.tire_guidance import guidance_contract
      guidance=guidance_contract(self.env.unwrapped.cfg)
      if guidance is not None:metadata['mgdp_tire_hole_guidance']=json.dumps(guidance,sort_keys=True)
    elif 'terrain_oracle' in self.env.unwrapped.cfg.observations:
      metadata=get_base_metadata(self.env.unwrapped,'lgdog_mgdp_height_oracle')
      metadata.update({'policy_architecture':'lgdog_mgdp_height_oracle_v1',
        'mgdp_vision_mode':'height_oracle','simulation_only':'true',
        'history_length':str(self.env.unwrapped.cfg.observations['actor'].history_length),
        'control_dt':str(self.env.unwrapped.step_dt)})
    else:
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      metadata.update({"policy_architecture": "lgdog_ppo_v1", "history_length": "1"})
    from src.config.mgdp_skill_training import skill_contract
    import json
    if skill_contract(self.env.unwrapped.cfg):
      metadata['mgdp_skill_training']=json.dumps(skill_contract(self.env.unwrapped.cfg),sort_keys=True)
    attach_metadata_to_onnx(str(Path(path) / filename), metadata)
