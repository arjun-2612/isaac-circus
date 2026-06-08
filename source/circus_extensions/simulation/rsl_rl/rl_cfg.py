# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg

@configclass
class RslRlPpoActorCriticEncoderCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticEncoder"
    """The policy class name. Default is ActorCritic."""

    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""

    num_encoder_obs: int = MISSING