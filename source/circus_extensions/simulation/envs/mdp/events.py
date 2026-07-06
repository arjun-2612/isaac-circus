# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that can be used to enable Spot randomizations.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from circus_extensions.simulation.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reference_name: str,
):
    if env_ids is None:
        env_ids = slice(None)

    cmd = env.command_manager.get_term(reference_name)
    cmd.reset_reference_trajectory(env_ids)

def reset_joint_state_to_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reference_name: str,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(reference_name)
    q_on_reset = cmd.joint_pos[env_ids]
    dq_on_reset = cmd.joint_vel[env_ids]

    # Smart randomization based on how often environment is able to complete the task
    randomization_level = (cmd.task_completion_ratio[env_ids] * (cmd.task_phase[env_ids] < 0.01)).unsqueeze(-1)

    q_min = q_on_reset + randomization_level * position_range[0]
    q_max = q_on_reset + randomization_level * position_range[1]
    dq_min = dq_on_reset + randomization_level * velocity_range[0]
    dq_max = dq_on_reset + randomization_level * velocity_range[1]

    # Clip to the limits
    q_limits = asset.data.soft_joint_pos_limits[env_ids, :len(asset_cfg.joint_ids), :]
    q_min_limited = torch.clip(q_min, min=q_limits[..., 0], max=q_limits[..., 1])
    q_max_limited = torch.clip(q_max, min=q_limits[..., 0], max=q_limits[..., 1])

    dq_limits = asset.data.soft_joint_vel_limits[env_ids, :len(asset_cfg.joint_ids)]
    dq_min_limited = torch.clip(dq_min, min=-dq_limits, max=dq_limits)
    dq_max_limited = torch.clip(dq_max, min=-dq_limits, max=dq_limits)

    # Sample from clipped values uniformly and write to sim
    q_rand = math_utils.sample_uniform(q_min_limited, q_max_limited, q_min_limited.shape, q_max_limited.device)
    dq_rand = math_utils.sample_uniform(dq_min_limited, dq_max_limited, dq_min_limited.shape, dq_max_limited.device)

    asset.write_joint_state_to_sim(q_rand, dq_rand, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)


def reset_root_state_to_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reference_name: str,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(reference_name)
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] = cmd.root_pos_w[env_ids, :]
    root_states[:, 3:7] = cmd.root_quat_w[env_ids]
    root_states[:, 7:10] = cmd.root_lin_vel_w[env_ids]
    root_states[:, 10:13] = cmd.root_ang_vel_w[env_ids]

    # Smart randomization based on how often environment is able to complete the task
    randomization_level = (cmd.task_completion_ratio[env_ids] * (cmd.task_phase[env_ids] < 0.01)).unsqueeze(-1)

    # Position/orientation randomization
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + rand_samples[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    # Velocity randomization
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    rand_samples *= randomization_level

    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def push_robot_under_wrench_assist(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)

    command = env.command_manager.get_term(command_name)
    alpha = (command.command_level[env_ids]).unsqueeze(-1)

    vel_w[:] += alpha * math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)

    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def randomize_command_restitution(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    restitution_range: tuple[float, float],
):
    """Randomize the racket-shuttle coefficient of restitution per environment.

    The impact restitution is uncertain and speed-dependent in reality, so it is resampled per
    episode to make the policy robust to the true value (to be pinned down later by system-ID
    against real hits recorded with the mocap rig).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term(command_name)
    lo, hi = restitution_range
    cmd.restitution[env_ids] = sample_uniform(lo, hi, (len(env_ids),), env.device)


def randomize_command_aero_length(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    aero_length_range: tuple[float, float],
):
    """Randomize the shuttlecock aerodynamic length L per episode (drag domain randomization).

    L = 2m/(rho*S*C_D) sets how hard the shuttle decelerates (terminal velocity = sqrt(g*L)). The
    real value is uncertain -- mocap-marker mass, feather wear, air density (temp/humidity), and the
    single-L quadratic model being an approximation all shift it -- so sampling it per episode makes
    the policy robust to the true drag rather than overfitting the nominal.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term(command_name)
    lo, hi = aero_length_range
    cmd.L[env_ids] = sample_uniform(lo, hi, (len(env_ids),), env.device)


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_flight_spawn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    action_name: str,
    flight_spawn_prob: float = 0.25,
    flight_spawn_height: float = 0.6,
):
    """Randomly initialize some environments in flight mode with elevated spawn height.
    
    For flight-mode envs: sets mode=1 in the action term and lifts the robot by
    flight_spawn_height above its current spawn position.
    For walk-mode envs: no change (mode stays 0 from action term reset).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)

    # Randomly select which envs spawn in flight
    rand = torch.rand(len(env_ids), device=env.device)
    flight_mask = rand < flight_spawn_prob  # [len(env_ids)]
    flight_env_ids = env_ids[flight_mask]
    
    if len(flight_env_ids) == 0:
        return

    # Set mode and prev_mode for flight envs
    # This overwrites the reset() call which always sets mode=0
    action_term._mode[flight_env_ids] = 1
    action_term._prev_mode[flight_env_ids] = 1

    # Reset the flight controller for these envs specifically
    action_term._flight_action.reset(flight_env_ids)

    # Lift the robot z position by flight_spawn_height
    root_pose = asset.data.root_pos_w[flight_env_ids].clone()
    root_pose[:, 2] += flight_spawn_height
    root_quat = asset.data.root_quat_w[flight_env_ids].clone()
    
    asset.write_root_pose_to_sim(
        torch.cat([root_pose, root_quat], dim=-1),
        env_ids=flight_env_ids,
    )
