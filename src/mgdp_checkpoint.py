"""Bind MGDP policies to the frozen encoder used during their training."""

import hashlib
import json
from pathlib import Path
import warnings

import torch


def file_digest(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def encoder_fingerprint(module):
  """Hash tensor values/config, independent of torch.save's zip filename."""
  digest = hashlib.sha256(json.dumps(module.cfg.to_dict(), sort_keys=True).encode())
  for name, value in sorted(module.state_dict().items()):
    value = value.detach().cpu().contiguous()
    digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
  return digest.hexdigest()


def save_encoder_contract(env, directory):
  if "perception" not in getattr(getattr(env, "cfg", None), "observations", {}):
    return None
  state = getattr(env, "_mgdp_perception_state", None)
  if state is None:
    raise RuntimeError("MGDP encoder must be initialized before saving the policy")
  module = state["module"]
  fingerprint = encoder_fingerprint(module)
  path = Path(directory) / f"perception_{fingerprint}.pt"
  if not path.is_file():
    temporary = path.with_suffix(".tmp")
    module.save(temporary)
    temporary.replace(path)
  return {"version": 1, "file": path.name, "sha256": file_digest(path),
          "fingerprint": fingerprint, "control_dt": env.step_dt,
          "history_length": env.cfg.observations["actor"].history_length}


def _local_artifact(directory, name):
  if not isinstance(name, str) or not name or Path(name).name != name:
    raise ValueError("Invalid MGDP encoder artifact name in checkpoint")
  return Path(directory) / name


def bind_checkpoint_encoder(env_cfg, checkpoint, override=None):
  """Set the encoder path BEFORE environment allocation/first observations.

  Legacy policies can use their run-local exported ONNX encoder. Without
  one, the user must specify the original encoder; a newly pretrained global
  checkpoint is not a safe automatic fallback for an old policy.
  """
  if "perception" not in env_cfg.observations:
    return None
  checkpoint = Path(checkpoint).resolve()
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  contract = (saved.get("infos") or {}).get("mgdp_perception")
  path = Path(override).expanduser().resolve() if override else None
  if contract is not None:
    if contract.get("version") != 1:
      raise ValueError("Unsupported MGDP encoder contract version")
    if path is None:
      path = _local_artifact(checkpoint.parent, contract["file"])
      if not path.is_file() or file_digest(path) != contract["sha256"]:
        raise ValueError(f"MGDP 配套感知文件丢失或校验失败：{path}")
    else:
      from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception
      module = MGPDPerception.from_checkpoint(path, map_location="cpu")
      if encoder_fingerprint(module) != contract["fingerprint"]:
        raise ValueError("指定的感知模型与该策略训练时不一致，不能混用重新预训练的编码器")
    if abs(env_cfg.decimation * env_cfg.sim.mujoco.timestep - contract["control_dt"]) > 1e-8:
      raise ValueError("MGDP policy and current environment control intervals differ")
    env_cfg.observations["actor"].history_length = contract["history_length"]
  elif path is None:
    # Earlier versions exported a per-run encoder for Sim2Sim, but Play ignored it.
    onnx_path = checkpoint.parent / "policy.onnx"
    if onnx_path.is_file():
      import onnx
      model = onnx.load(str(onnx_path), load_external_data=False)
      metadata = {item.key: item.value for item in model.metadata_props}
      if metadata.get("policy_architecture") == "lgdog_mgdp_v1":
        path = _local_artifact(checkpoint.parent, metadata.get("perception_checkpoint"))
        if not path.is_file() or file_digest(path) != metadata.get("perception_sha256"):
          raise ValueError(f"旧策略目录的感知文件丢失或与ONNX记录不匹配：{path}")
        if "history_length" in metadata:
          env_cfg.observations["actor"].history_length = int(metadata["history_length"])
        if "control_dt" in metadata and abs(
            env_cfg.decimation * env_cfg.sim.mujoco.timestep - float(metadata["control_dt"])) > 1e-8:
          raise ValueError("Legacy MGDP policy and current environment control intervals differ")
    if path is None:
      candidate = checkpoint.parent / "policy.perception.pt"
      if candidate.is_file():
        path = candidate
    if path is None:
      raise ValueError("旧MGDP策略没有保存配套感知文件，不能自动使用公共目录的新模型。"
                       "请用 --perception-checkpoint 指定训练时的旧感知权重；"
                       "若旧权重已丢失，需要用新感知模型重新训练策略。")
    warnings.warn("旧checkpoint未记录感知指纹，使用同run导出的感知模型；"
                  "若该run曾更换编码器，需另行确认旧checkpoint对应关系。")
  elif contract is None:
    warnings.warn("旧checkpoint未记录感知指纹，使用显式指定的感知文件；请确认它来自原训练。")
  if not path.is_file():
    raise FileNotFoundError(f"感知权重不存在：{path}")
  env_cfg.observations["perception"].terms["depth_feature"].params["checkpoint"] = str(path)
  print(f"[INFO] MGDP policy encoder: {path}", flush=True)
  print(f"[INFO] MGDP encoder SHA256: {file_digest(path)}", flush=True)
  return path


def validate_runtime_encoder(env, checkpoint):
  contract = (checkpoint.get("infos") or {}).get("mgdp_perception")
  if contract is None or "perception" not in getattr(getattr(env, "cfg", None), "observations", {}):
    return
  state = getattr(env, "_mgdp_perception_state", None)
  if state is None or encoder_fingerprint(state["module"]) != contract["fingerprint"]:
    raise ValueError("MGDP策略与当前感知模型不匹配；请使用该checkpoint配套的感知快照，"
                     "不要将旧策略与重新预训练的模型混用。")
  if abs(env.step_dt - contract["control_dt"]) > 1e-8:
    raise ValueError("MGDP policy and current environment control intervals differ")
  if env.cfg.observations["actor"].history_length != contract["history_length"]:
    raise ValueError("MGDP policy and current observation history lengths differ")
