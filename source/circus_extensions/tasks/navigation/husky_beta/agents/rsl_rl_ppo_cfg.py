# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from circus_extensions.simulation.rsl_rl.rl_cfg import RslRlPpoActorCriticEncoderCfg

@configclass
class HuskyBetaHLPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "husky_beta_rl"
    logger = "wandb"
    wandb_project = "husky_beta_hlp"
    empirical_normalization = False
    store_code_state = False
    policy = RslRlPpoActorCriticEncoderCfg(
        class_name="ActorCriticEncoder",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[128, 64, 32],
        num_encoder_obs=272,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )