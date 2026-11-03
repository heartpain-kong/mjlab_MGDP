"""HIM tasks: legacy gentle terrain and standard rough terrain without grid scans."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from src.him_contract import HIM_HISTORY_LENGTH, HIM_FRAME_DIM, LGDOG_HIM_LAYOUT

from .blind_terrain_env_cfg import lgdog1_blind_terrain_env_cfg
from .env_cfgs import BASE_BODY, FOOT_SITES
from .reward_cfg import make_lgdog_rewards
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


def lgdog1_race_env_cfg(play=False):
  """2027 比赛场地地形；观测/奖励与标准 HIM 完全一致。

  地形按规则尺寸复刻（梅花桩 0.2×0.2×0.1、单木桥 0.4 宽、台阶 0.12/0.32、
  砖木坑碎砖、绕桩细柱、0.3 m 高墙），里程与 sim2sim 的 `--map race` 是同一套
  障碍定义 —— 训练里练"能不能过"，真场地上用 `--map race` 验"过不过得去"。
  """
  cfg = lgdog1_blind_terrain_env_cfg(
    play=play, reward_profile="him_standard", gentle_terrain=False,
    critic_height_scan=False, terrain="race",
  )
  return _add_him_observations(cfg)


def lgdog1_him_flat_env_cfg(play=False):
  """Flat pretraining with the exact standard HIM observation contract."""
  cfg = lgdog1_blind_terrain_env_cfg(
    play=play, reward_profile="him_standard", gentle_terrain=False,
    critic_height_scan=False,
  )
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.curriculum.pop("terrain_levels", None)
  cfg.rewards = make_lgdog_rewards(
    "flat", base_body=BASE_BODY, foot_sites=FOOT_SITES,
  )
  cfg.sim.nconmax = None if play else 35
  cfg.sim.njmax = 300
  return _add_him_observations(cfg)


def _add_him_observations(cfg):
  # A 47-wide vector with reordered terms would silently invalidate policies.
  LGDOG_HIM_LAYOUT.validate_terms(cfg.observations["actor"].terms)
  # Separate clean target groups prevent fragile hard-coded critic slicing.
  cfg.observations["him_target"] = deepcopy(cfg.observations["actor"])
  cfg.observations["him_target"].enable_corruption = False
  cfg.observations["him_velocity"] = ObservationGroupCfg(
    terms={"velocity": ObservationTermCfg(func=body_linear_velocity)},
    concatenate_terms=True, enable_corruption=False, history_length=1,
  )
  cfg.observations["actor"].history_length = HIM_HISTORY_LENGTH
  # Do not flatten each individual term: that produces term-major history.
  # Concatenate [N,H,term_dim] on the last axis -> [N,H,47].
  cfg.observations["actor"].flatten_history_dim = False
  cfg.observations["actor"].concatenate_dim = -1
  return cfg


@dataclass
class HIMModelCfg(RslRlModelCfg):
  history_length: int = HIM_HISTORY_LENGTH
  frame_dim: int = HIM_FRAME_DIM
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
