"""HIM tasks: legacy gentle terrain and standard rough terrain without grid scans."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from .blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg
from .rl_cfg import lgdog1_ppo_runner_cfg


def body_linear_velocity(env):
  """Training label only: physical body-frame m/s, no observation noise."""
  return env.scene["robot"].data.root_link_lin_vel_b


def lgdog1_him_env_cfg(play=False):
  """Original V7 task: gentle terrain, privileged critic grid scan."""
  cfg = lgdog1_blind_terrain_env_cfg(play=play, reward_profile="him")
  return _add_him_observations(cfg)


def lgdog1_him_standard_env_cfg(play=False):
  """Standard rough terrain; no actor/critic body height scan."""
  cfg = lgdog1_blind_terrain_env_cfg(
    play=play, reward_profile="him_standard", gentle_terrain=False,
    critic_height_scan=False,
  )
  return _add_him_observations(cfg)


def _add_him_observations(cfg):
  # Separate clean target groups prevent fragile hard-coded critic slicing.
  cfg.observations["him_target"] = deepcopy(cfg.observations["actor"])
  cfg.observations["him_target"].enable_corruption = False
  cfg.observations["him_velocity"] = ObservationGroupCfg(
    terms={"velocity": ObservationTermCfg(func=body_linear_velocity)},
    concatenate_terms=True, enable_corruption=False, history_length=1,
  )
  cfg.observations["actor"].history_length = 6
  # Do not flatten each individual term: that produces term-major history.
  # Concatenate [N,6,term_dim] on the last axis -> [N,6,47].
  cfg.observations["actor"].flatten_history_dim = False
  cfg.observations["actor"].concatenate_dim = -1
  return cfg


@dataclass
class HIMModelCfg(RslRlModelCfg):
  history_length: int = 6
  frame_dim: int = 47
  latent_dim: int = 16


@dataclass
class HIMPpoCfg(RslRlPpoAlgorithmCfg):
  estimator_learning_rate: float = 1e-3
  # CUDA fused Adam reduces optimizer launch overhead for the PPO/HIM minibatches.
  # HIMPPO falls back to regular Adam on CPU or older PyTorch builds.
  fused_optimizer: bool = True
  # Match the five PPO epochs: HIMPPO performs one auxiliary pass per
  # mini-batch in each estimator epoch, so this yields 5 * 4 updates instead
  # of only 4 updates per rollout.
  estimator_epochs: int = 5


def lgdog1_him_runner_cfg(experiment_name="lgdog1_velocity_blind_him_v1"):
  cfg = lgdog1_ppo_runner_cfg(experiment_name)
  cfg.actor = HIMModelCfg(**vars(cfg.actor))
  cfg.actor.class_name = "src.tasks.velocity.rl.him:HIMActor"
  cfg.algorithm = HIMPpoCfg(**vars(cfg.algorithm))
  cfg.algorithm.class_name = "src.tasks.velocity.rl.him:HIMPPO"
  return cfg
