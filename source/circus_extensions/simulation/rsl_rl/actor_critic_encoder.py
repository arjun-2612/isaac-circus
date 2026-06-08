# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic

class ActorCriticEncoder(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[128, 64, 32],
        num_encoder_obs=0,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """
        Initializes the ActorCriticEncoder model.

        This modified version supports separate MLP encoders for high-dimensional
        observations like height encoders and privileged information.

        Args:
            network_config (NetworkCfg, optional): Configuration to dynamically build the networks.
                Example structure with encoders:
                ```
                network_config = NetworkCfg(
                    'num_actor_obs': int,
                    'num_critic_obs': int
                    'num_encoder_obs': int,             # Can be 0
                    'num_actions': int,
                    'encoder_hidden_dims': list[int],   # Can be None
                    'actor_hidden_dims': list[int],
                    'critic_hidden_dims': list[int],
                    'activation': str
                )
                ```
        """
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        self.encoder_obs = num_encoder_obs
        self.proprioceptive_obs = num_actor_obs - num_encoder_obs

        actor_input_dim = self.proprioceptive_obs + encoder_hidden_dims[-1]
        critic_input_dim = (
            num_critic_obs - num_encoder_obs
        ) + encoder_hidden_dims[-1]

        super().__init__(
            actor_input_dim,
            critic_input_dim,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            **kwargs
        )

        activation_fn = get_activation(activation)

        self.encoder = self.build_mlp(
            input_dim=num_encoder_obs,
            hidden_dims=encoder_hidden_dims[:-1],
            output_dim=encoder_hidden_dims[-1],
            activation=activation_fn,
        )

        # Print networks and parameter counts
        print(f"Encoder network: {self.encoder}")
        print(f"Number of learnable parameters: {self._count_parameters(self.encoder):,}")

        Normal.set_default_validate_args = False

    def build_mlp(self, input_dim, hidden_dims, output_dim, activation):
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    @staticmethod
    def _count_parameters(model: nn.Module) -> int:
        """Counts the number of learnable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def act(self, observations: torch.Tensor, **kwargs):
        prop_obs = observations[:, 0 : self.proprioceptive_obs]
        encoder_obs = observations[:, self.proprioceptive_obs :]
        encoder_latent = self.encoder(encoder_obs)

        actor_input = torch.cat([prop_obs, encoder_latent], dim=-1)
        return super().act(actor_input)

    def act_inference(self, observations: torch.Tensor):
        prop_obs = observations[:, 0 : self.proprioceptive_obs]
        encoder_obs = observations[:, self.proprioceptive_obs :]
        encoder_latent = self.encoder(encoder_obs)

        actor_input = torch.cat([prop_obs, encoder_latent], dim=-1)
        return super().act_inference(actor_input)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        prop_obs = critic_observations[:, 0 : self.proprioceptive_obs]
        encoder_obs = critic_observations[:, self.proprioceptive_obs : self.proprioceptive_obs + self.encoder_obs]
        priv_obs = critic_observations[:, self.proprioceptive_obs + self.encoder_obs :]
        encoder_latent = self.encoder(encoder_obs)

        critic_input = torch.cat([prop_obs, encoder_latent, priv_obs], dim=-1)
        return super().evaluate(critic_input)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
