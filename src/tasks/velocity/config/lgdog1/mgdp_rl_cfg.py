"""RL configuration for the LgDog1 + MGDP perception task.

Plain PPO (no HIM estimator): the actor consumes the 4-frame proprioception
history *and* the 64-dim perception feature, which is the mjlab equivalent of
MGDP's ``actor(obs45 + proprio_token + visual_token)``.
"""

from __future__ import annotations

from src.tasks.velocity.config.lgdog1.rl_cfg import lgdog1_ppo_runner_cfg


def lgdog1_mgdp_runner_cfg(experiment_name: str = "lgdog1_velocity_mgdp_v1"):
    cfg = lgdog1_ppo_runner_cfg(experiment_name)
    # Two actor groups: proprio history (4 x 47) and perception (64).
    cfg.obs_groups = {
        "actor": ("actor", "perception"),
        "critic": ("critic",),
    }
    # actor input = 188 + 64 = 252; the default [512, 256, 128] is fine.
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.actor.obs_normalization = True
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.critic.obs_normalization = True
    cfg.max_iterations = 10001
    cfg.save_interval = 100
    return cfg
