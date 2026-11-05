import os
from pathlib import Path
import warnings

import torch

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


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

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
    from src.mgdp_checkpoint import save_encoder_contract
    perception = save_encoder_contract(self.env.unwrapped, Path(path).parent)
    if perception is not None:
      infos["mgdp_perception"] = perception
    passage = self._passage_curriculum()
    if passage is not None:
      infos["mgdp_passage_curriculum"] = passage.state_dict()
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
      if has_perception:
        from src.mgdp_checkpoint import validate_runtime_encoder
        validate_runtime_encoder(self.env.unwrapped, checkpoint)
    if passage is not None:
      state = checkpoint.get("infos", {}).get("mgdp_passage_curriculum")
      if state is None:
        raise ValueError("定向越障 v2 不能完整续训旧 MGDP：缺少通过率课程状态，请创建新训练")
      passage.load_state_dict(state)
    infos = super().load(path, load_cfg, strict, map_location)
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
        "control_dt": str(self.env.unwrapped.step_dt),
      })
    else:
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      metadata.update({"policy_architecture": "lgdog_ppo_v1", "history_length": "1"})
    attach_metadata_to_onnx(str(Path(path) / filename), metadata)
