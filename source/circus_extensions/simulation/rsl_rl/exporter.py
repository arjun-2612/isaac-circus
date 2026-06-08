# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_onnx(
    actor_critic: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""

class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)
        self.proprioceptive_obs = None
        self.encoder = None
        try:
            self.encoder = copy.deepcopy(actor_critic.encoder)
            self.proprioceptive_obs = actor_critic.proprioceptive_obs
        except AttributeError:
            self.encoder = None
            self.proprioceptive_obs = None
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x): # this is only getting what the actor gets, not the full input!!!
        if self.encoder:
            encoder_latent = self.encoder(x[:, self.proprioceptive_obs:])
            return self.actor(torch.cat((x[:, :self.proprioceptive_obs], encoder_latent), dim=-1))
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            if self.encoder:
                encoder_obs = torch.zeros(1, self.encoder[0].in_features)
                actor_obs = torch.zeros(1, self.actor[0].in_features)
                torch.onnx.export(
                    self.encoder,  # Direct - no wrapper needed
                    encoder_obs,
                    os.path.join(path, "encoder.onnx"),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["encoder_obs"],
                    output_names=["encoder_latent"],
                    dynamic_axes={},
                )
                torch.onnx.export(
                    self.actor,
                    actor_obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
            else:
                obs = torch.zeros(1, self.actor[0].in_features)
                torch.onnx.export(
                    self.actor,
                    obs,
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
