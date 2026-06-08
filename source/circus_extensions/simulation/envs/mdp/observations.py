# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from circus_extensions.simulation.envs.mdp import MotionDatasetSampler, WrenchAssistCommand, RMPCCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_6do(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, noise: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """Body 6D orientation vector in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.root_quat_w

    # generate random value within the bounds of noise tuple
    if noise[0] != 0.0 or noise[1] != 0.0:
        r, p, y = math_utils.euler_xyz_from_quat(body_quat_w)
        body_euler_w = torch.stack([r, p, y], dim=-1) # [N, 3]
        rand_noise = torch.empty_like(body_euler_w).normal_(noise[0], noise[1])
        noisy_euler_w = math_utils.wrap_to_pi(body_euler_w + rand_noise)
        body_quat_w = math_utils.quat_from_euler_xyz(noisy_euler_w[:, 0], noisy_euler_w[:, 1], noisy_euler_w[:, 2])

    body_r_matrix_w = math_utils.matrix_from_quat(body_quat_w)
    body_6do_w = body_r_matrix_w[:, :, :2].flatten(start_dim=1)
    return body_6do_w


def pos_rel_env_origin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position relative to environment origin."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def goal_rel_to_origin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Goal position relative to environment origin."""
    cmd = env.command_manager.get_command(command_name)[:, :3]
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w
    goal_w = math_utils.quat_rotate(base_quat, cmd) + base_pos
    return goal_w - env.scene.env_origins


def height_difference_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Signed height difference to target (positive = target above)."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :3]
    base_quat = asset.data.root_quat_w
    base_pos = asset.data.root_pos_w
    cmd_w = math_utils.quat_rotate(base_quat, cmd) + base_pos
    robot_z = base_pos[:, 2]
    target_z = cmd_w[:, 2]
    return (target_z - robot_z).unsqueeze(-1)


def distance_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """3D distance to target."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :3]
    base_quat = asset.data.root_quat_w
    base_pos = asset.data.root_pos_w
    cmd_w = math_utils.quat_rotate(base_quat, cmd) + base_pos
    robot_pos = base_pos[:, :3]
    target_pos = cmd_w[:, :3]
    return torch.norm(target_pos - robot_pos, dim=-1, keepdim=True)


def joint_pos_reference_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, reference_name: str = "motion_reference") -> torch.Tensor:
    """Joint position relative to the reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd: MotionDatasetSampler | RMPCCommand = env.command_manager.get_term(reference_name) 
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - cmd.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_reference(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, reference_name: str = "motion_references") -> torch.Tensor:
    """Joint position reference."""
    # extract the used quantities (to enable type-hinting)
    cmd: MotionDatasetSampler | RMPCCommand = env.command_manager.get_term(reference_name)
    return cmd.joint_pos[:, asset_cfg.joint_ids]


def base_pos_z_reference(env: ManagerBasedRLEnv, reference_name: str) -> torch.Tensor:
    """Root height in the reference trajectory."""
    # extract the used quantities (to enable type-hinting)
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    return cmd.root_pos_w[:, 2].unsqueeze(-1)


def projected_gravity_reference(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, reference_name: str = "motion_reference") -> torch.Tensor:
    """Gravity projection reference."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    return math_utils.quat_rotate_inverse(cmd.root_quat_w, asset.data.GRAVITY_VEC_W)


def body_relative_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w
    body_pos_wrt_b_in_w = body_pos_w - base_pos_w.unsqueeze(1)
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, body_pos_wrt_b_in_w.size(1), -1)
    body_pos_b = math_utils.quat_rotate_inverse(base_quat_w_expanded, body_pos_wrt_b_in_w)
    return body_pos_b.view(env.num_envs, -1)


def body_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    base_quat_w = asset.data.root_quat_w
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, body_lin_vel_w.size(1), -1)
    body_lin_vel_b = math_utils.quat_rotate_inverse(base_quat_w_expanded, body_lin_vel_w)
    return body_lin_vel_b.view(env.num_envs, -1)


def base_vel_reference(env: ManagerBasedRLEnv, reference_name: str = "motion_references") -> torch.Tensor:
    """Root linear velocity reference."""
    # extract the used quantities (to enable type-hinting)
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    velocity_refs = torch.cat([cmd.root_lin_vel_b, cmd.root_ang_vel_b], dim=-1)
    return velocity_refs


def wrench_force(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd: WrenchAssistCommand = env.command_manager.get_term(command_name)
    return (cmd.wrench_scale * torch.linalg.norm(cmd.forces_b, dim=-1)).unsqueeze(-1)


def wrench_torque(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd: WrenchAssistCommand = env.command_manager.get_term(command_name)
    return (cmd.wrench_scale * torch.linalg.norm(cmd.torques_b, dim=-1)).unsqueeze(-1)


def wrench_scale(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd: WrenchAssistCommand = env.command_manager.get_term(command_name)
    return cmd.wrench_scale.unsqueeze(-1)


def similarity_metric(env: ManagerBasedRLEnv, reference_name: str) -> torch.Tensor:
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    return cmd.metrics["similarity_metric"].unsqueeze(-1)


def joint_efforts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns contact forces for the given set of bodies.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration specifying the contact force bodies.

    Returns:
        A tensor of shape (num_envs, num_bodies, 3) representing the contact forces in the world frame.
    """
    # Get the rigid object (robot) using the scene entity config
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)

def foot_targets(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Returns the foot targets generated by the planner.
    
    Args:
        env: The environment instance.
        command_name: The name of the command
        
    Returns:
        A tensor of shape (num_envs, 4, 2) representing the XY positions of each foot in base frame.
    """
    cmd = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    ft = cmd.foot_targets.reshape(env.num_envs, 4, 2)
    z = torch.zeros((env.num_envs, 4, 1), device=env.device)
    ft = torch.cat([ft, z], dim=-1)  # (E,4,3)
    base_pos = asset.data.root_pos_w.unsqueeze(1)  # (E,1,3)
    base_quat = asset.data.root_quat_w.unsqueeze(1)  # (E,1,4)
    ft_b = math_utils.quat_rotate_inverse(base_quat, ft - base_pos)
    return ft_b[..., :2].reshape(env.num_envs, -1)

def swing_targets(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Returns the swing foot targets generated by the planner.
    
    Args:
        env: The environment instance.
        command_name: The name of the command
        
    Returns:
        A tensor of shape (num_envs, 4, 3) representing the XYZ positions of each foot in base frame.
    """
    cmd = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    st = cmd.swing_targets.reshape(env.num_envs, 4, 3)
    base_pos = asset.data.root_pos_w.unsqueeze(1)  # (E,1,3)
    base_quat = asset.data.root_quat_w.unsqueeze(1)  # (E,1,4)
    st_b = math_utils.quat_rotate_inverse(base_quat, st - base_pos)
    return st_b.reshape(env.num_envs, -1)

def foot_forces(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Returns the foot forces generated by the MPC.
    
    Args:
        env: The environment instance.
        command_name: The name of the command
        
    Returns:
        A tensor of shape (num_envs, 4, 3) representing the XYZ forces of each foot in base frame.
    """
    cmd = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    ff = cmd.foot_forces.reshape(env.num_envs, 4, 3)
    base_quat = asset.data.root_quat_w.unsqueeze(1)  # (E,1,4)
    ff_b = math_utils.quat_rotate_inverse(base_quat, ff)
    return ff_b.reshape(env.num_envs, -1)

def foot_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the foot positions in base frame
    
    Args:
        env: The environment instance.
        asset_cfg: The asset 
        
    Returns:
        The foot positions in base frame
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.root_pos_w.unsqueeze(1)  # (E,1,3)
    base_quat = asset.data.root_quat_w.unsqueeze(1)  # (E,1,4)
    fp_w = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()
    fp_b = math_utils.quat_rotate_inverse(base_quat, fp_w - base_pos)
    return fp_b.flatten(start_dim=1)

def contact_schedule(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the contact schedule
    
    Args:
        env: The environment instance.
        command_name: The name of the command
        
    Returns:
        The contact schedule 
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.swing_mask.float()

def desired_base_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the desired base state (pos and quat)
    
    Args:
        env: The environment instance.
        command_name: The name of the command
        
    Returns:
        The desired base state
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.desired_body_state

def applied_foot_forces(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Returns the applied foot forces
    
    Args:
        env: The environment instance.
        action_name: The name of the action
        
    Returns:
        The applied foot forces
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.applied_forces