# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##

def air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
) -> torch.Tensor:
    """Reward longer feet air time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    reward = torch.where(current_air_time < mode_time, current_air_time, 0)
    return torch.sum(reward, dim=1)

def contact_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
) -> torch.Tensor:
    """Reward longer feet contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    # compute the reward
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    reward = torch.where(current_contact_time < mode_time, current_contact_time, 0)
    # reward = torch.exp(-(current_contact_time - mode_time) / std)
    return torch.sum(reward, dim=1)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.air_time: float = cfg.params["air_time"]
        self.contact_time: float = cfg.params["contact_time"]
        self.terrain_type: str = cfg.params["terrain_type"]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        air_time: float, 
        contact_time: float,
        terrain_type: str,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1

        return sync_reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)

        se_air_1 = torch.clip(torch.square(air_time[:, foot_0] - self.air_time), max=self.max_err**2)
        se_contact_1 = torch.clip(torch.square(contact_time[:, foot_0] - self.contact_time), max=self.max_err**2)

        se_air_2 = torch.clip(torch.square(air_time[:, foot_1] - self.air_time), max=self.max_err**2)
        se_contact_2 = torch.clip(torch.square(contact_time[:, foot_1] - self.contact_time), max=self.max_err**2)

        if self.terrain_type == "flat":
            return torch.exp(-(se_air_1 + se_air_2 + se_contact_1 + se_contact_2) / self.std)
        
        return torch.exp(-(se_air + se_contact) / self.std)


##
# Regularization Penalties
##

def walk_morph_joint_position_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    stand_still_scale: float, 
    velocity_threshold: float, 
    command_name: str = "base_vel_and_height",
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(torch.cat([cmd[:, :2], cmd[:, 3].unsqueeze(-1)], dim=1), dim=1)
    walk_default_pos = asset.data.default_joint_pos.clone()
    morph_default_pos = torch.tensor([
        1.57, -1.57, 1.57, -1.57,
        0.698132, 0.698132, 0.698132, 0.698132,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ], device=env.device).unsqueeze(0).expand(env.num_envs, -1)

    switch_default_pos = cmd[:, 2] < 0.2
    default_pos = torch.where(
        switch_default_pos.unsqueeze(-1), 
        morph_default_pos, 
        walk_default_pos, 
    )
    reward = torch.linalg.norm((asset.data.joint_pos - default_pos), dim=1)
    return torch.where(cmd_norm > velocity_threshold, reward, stand_still_scale * reward)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return penalty


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)