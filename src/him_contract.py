"""LgDog HIM observation and checkpoint contract, without framework imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HimLayout:
  """Ordered per-frame actor terms; history is frame-major, oldest first."""

  terms: tuple[tuple[str, int], ...]
  history_length: int

  @property
  def frame_dim(self):
    return sum(width for _, width in self.terms)

  @property
  def input_dim(self):
    return self.frame_dim * self.history_length

  def validate_terms(self, terms):
    actual = tuple(terms)
    expected = tuple(name for name, _ in self.terms)
    if actual != expected:
      raise ValueError(f"Incompatible LgDog HIM actor terms: expected {expected}, got {actual}")


LGDOG_HIM_LAYOUT = HimLayout(
  terms=(("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3),
         ("phase", 2), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12)),
  history_length=3,
)

HIM_HISTORY_LENGTH = LGDOG_HIM_LAYOUT.history_length
HIM_FRAME_DIM = LGDOG_HIM_LAYOUT.frame_dim
HIM_INPUT_DIM = LGDOG_HIM_LAYOUT.input_dim
HIM_HISTORY_ORDER = "oldest_to_newest"


def make_him_checkpoint_contract(frame_dim, history_length):
  """Preserve the existing checkpoint wire format."""
  return {"frame_dim": frame_dim, "history_length": history_length,
          "order": HIM_HISTORY_ORDER}


def apply_him_play_contract(env_cfg, agent_cfg, contract):
  """Use checkpoint history for playback; never convert training weights."""
  if (not isinstance(contract, dict) or contract.get("frame_dim") != HIM_FRAME_DIM
      or contract.get("order") != HIM_HISTORY_ORDER
      or type(contract.get("history_length")) is not int or contract["history_length"] < 1):
    raise ValueError(f"Missing or invalid LgDog HIM checkpoint history contract: {contract}")
  history_length = contract["history_length"]
  env_cfg.observations["actor"].history_length = history_length
  agent_cfg.actor.history_length = history_length
  return history_length
