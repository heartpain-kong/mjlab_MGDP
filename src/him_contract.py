"""Lightweight HIM defaults shared by training, deployment and the dashboard."""

HIM_HISTORY_LENGTH = 3
HIM_FRAME_DIM = 47
HIM_INPUT_DIM = HIM_HISTORY_LENGTH * HIM_FRAME_DIM


def apply_him_play_contract(env_cfg, agent_cfg, contract):
  """Use checkpoint history for playback; never convert training weights."""
  if (not isinstance(contract, dict) or contract.get("frame_dim") != HIM_FRAME_DIM
      or contract.get("order") != "oldest_to_newest"
      or type(contract.get("history_length")) is not int or contract["history_length"] < 1):
    raise ValueError(f"Missing or invalid LgDog HIM checkpoint history contract: {contract}")
  history_length = contract["history_length"]
  env_cfg.observations["actor"].history_length = history_length
  agent_cfg.actor.history_length = history_length
  return history_length
