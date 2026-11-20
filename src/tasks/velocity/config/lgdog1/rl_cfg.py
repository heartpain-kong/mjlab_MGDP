"""RL configuration for LgDog1 velocity task.

网络结构与 PPO 超参沿用 mjlab 给四足调的默认（与 go2 相同），因为这套是 mjlab 的
算法实现（rsl-rl 的 PPO + 非 HIM 结构）。用户 legged_gym 那边用的是
`HIMActorCritic` + 45 维观测历史 + 估计器，与 mjlab 的 actor/critic 分组观测
不是同一套东西，逐项搬超参没有意义。

用户侧的可比项：
    learning_rate 3e-4 / entropy_coef 0.0 / num_mini_batches 16 /
    schedule 'adaptive' / desired_kl 0.01 / hidden [512,256,128] elu
→ 下面这些与 mjlab 默认一致或已按用户值设置。
"""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class LgDogPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  """LgDog1 PPO runner options on top of mjlab's defaults."""

  symmetry_augment: bool = False
  """Left-right mirror data augmentation for the PPO policy loss.

  When enabled, every mini-batch is augmented with the mirrored actor
  observation/action pair (batch x2).  The critic/value loss stays on the
  original batch.  Off by default; implemented by wrapping the PPO update at
  runtime (see ``src/tasks/velocity/rl/runner.py``).
  """


def lgdog1_ppo_runner_cfg(
  experiment_name: str = "lgdog1_velocity", *, symmetry_augment: bool = False
) -> LgDogPpoRunnerCfg:
  """Create RL runner configuration for LgDog1 velocity task.

  ★ 带高度扫描的 `LgDog1-Rough`（234 维观测）和不带的 `LgDog1-Rough-Blind`（47 维）
  用【不同的 experiment_name】—— 两边的 checkpoint 互不兼容，分开存才不会误加载。

  ``symmetry_augment=True`` 打开左右镜像数据增广（默认关闭）；训练日志会打印
  「symmetry augmentation enabled」。
  """
  return LgDogPpoRunnerCfg(
    symmetry_augment=symmetry_augment,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Keep exploration inside the normalized action range while the
        # environment wrapper clips executed actions to [-1, 1].
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # An unbounded Gaussian plus a positive entropy bonus made V10/V11
      # increase std until almost every action was clipped before execution.
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    # ★ 2026-11-19：50 → 32，单轮采样步数降 36%，配合地形列数减半把 8 s/轮压到约 3 s。
    # 32 步 = 0.64 s rollout，仍覆盖约 1.5 个 trot 周期，HIM 历史（4 帧）足够。
    num_steps_per_env=24,
    max_iterations=10001,
    # The previous runs had unbounded policy outputs.  Clip normalized actions
    # before they are converted to joint targets; the value is still exposed in
    # agent.yaml so an A/B run can set it back to null explicitly.
    clip_actions=1.0,
  )
