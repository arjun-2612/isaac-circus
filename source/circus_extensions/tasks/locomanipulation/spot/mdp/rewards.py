# Edited from IsaacLab, used by Silicon Synapse Lab RL Research Team
# Author: Arjun Viswanathan

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from circus_extensions.simulation.envs.mdp import ShuttleLauncherCommand
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Task Rewards
##


def ee_approach_reward(
    env: ManagerBasedRLEnv, 
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.5, 
) -> torch.Tensor:
    ee_frame = env.scene[ee_frame_cfg.name]
    cmd: ShuttleLauncherCommand = env.command_manager.get_term("shuttle_launcher")

    pos_error = torch.norm(cmd.interception_pos - ee_frame.data.target_pos_w.squeeze(1), dim=-1)
    reward = torch.exp(-pos_error / std)

    return reward * cmd.at_intercept_time.float()


def ee_velocity_tracking_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    perp_std: float = 3.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd: ShuttleLauncherCommand = env.command_manager.get_term("shuttle_launcher")

    # True racket velocity = racket velocity - base velocity
    # Remove base velocity from hacking the reward
    ee_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids, :3].squeeze(1).clone()
    ee_vel_w[:, 2] = ee_vel_w[:, 2] - torch.abs(asset.data.root_lin_vel_w[:, 2])

    swing_dir = -cmd.interception_vel.clone()
    swing_dir[:, 2] = swing_dir[:, 2] + 1.0
    swing_dir[:, 1] = 0.0
    swing_dir = swing_dir / (torch.norm(swing_dir, dim=-1, keepdim=True) + 1e-6)

    speed_along = torch.sum(ee_vel_w * swing_dir, dim=-1)
    forward = torch.clamp(speed_along / cmd.swing_speed_target, 0.0, 1.0)

    # Penalize ONLY sideways (world-Y) motion. The racket is free to curve through
    # the forward-up (XZ) plane -- that arc IS the wrist flick.
    lateral_speed = torch.abs(ee_vel_w[:, 1])
    align = torch.exp(-(lateral_speed ** 2) / (perp_std ** 2))

    reward = forward * align
    return reward * cmd.at_intercept_time.float()


def ee_orientation_tracking_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.3, 
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd: ShuttleLauncherCommand = env.command_manager.get_term("shuttle_launcher")

    # Racket face normal is the LOCAL -Z axis -> rotate [0,0,-1] into world.
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    local_normal = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    ee_normal_w = math_utils.quat_rotate(ee_quat_w, local_normal)

    # Face should point where the racket pushes the shuttle: forward (+X) and up --
    # i.e. the same world direction as the swing.
    target_normal = -cmd.interception_vel.clone()
    target_normal[:, 2] = torch.abs(target_normal[:, 2]) + 0.5
    target_normal[:, 1] = 0.0
    target_normal = target_normal / (torch.norm(target_normal, dim=-1, keepdim=True) + 1e-6)

    cos_dist = 1.0 - torch.sum(ee_normal_w * target_normal, dim=-1)
    reward = 1.0 / (1.0 + cos_dist ** 2 / std)
    return reward * cmd.at_intercept_time.float()


def placement_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "shuttle_launcher",
    target: tuple[float, float] = (5.0, 0.0),
    pos_std: float = 2.0,
    clearance_scale: float = 0.2,
) -> torch.Tensor:
    """Reward returning the shuttle over the net into the opponent court.

    Fires only at the interception instant (like the other task rewards). At the bounce the
    post-hit trajectory is rolled forward to estimate the landing point (relative to the court
    origin) and the height at which it crosses the net plane. Reward = (net cleared) x (landing
    near ``target``), a broad kernel so it rewards a REGION, not a pinpoint. A missed hit leaves
    ``net_clearance`` very negative -> ~0 reward.
    """
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name)

    target_t = torch.tensor(target, device=env.device)
    dist_sq = torch.sum((cmd.predicted_landing_local - target_t) ** 2, dim=-1)
    land_reward = torch.exp(-dist_sq / (pos_std ** 2))
    cleared = torch.sigmoid(cmd.net_clearance / clearance_scale)  # ~0 into-net/short, ~1 well over

    return cleared * land_reward * cmd.at_intercept_time.float()


def stand_still_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    """
    Reward for standing still when no target is assigned or all targets exhausted.
    r_stand = exp(-1/(sigma*N_joints) * |q - q*|)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd: ShuttleLauncherCommand = env.command_manager.get_term("shuttle_launcher")
    
    # Active when: all targets used up OR time_to_intercept is large (shuttle far away)
    # Option 1: All targets exhausted
    all_targets_done = (cmd.targets_remaining <= 0).float()
    
    # Option 2: Waiting phase (shuttle hasn't been launched yet or is far)
    # Could also use: time_to_intercept > some_threshold
    
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    joint_error = torch.sum(torch.abs(joint_pos - default_pos), dim=-1)
    n_joints = joint_pos.shape[-1]
    
    reward = torch.exp(-joint_error / (std * n_joints))
    return reward * all_targets_done


def face_net_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward for facing the net. Encourages the robot to orient itself toward the opponent and shuttle.
    r_face = cos(theta) where theta is angle between robot forward and net normal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get robot forward direction (assuming +X is forward in world frame)
    base_quat = asset.data.root_quat_w.clone()
    local_forward = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    forward_dir = math_utils.quat_rotate(base_quat, local_forward)
    
    # Net normal points from robot to opponent (assume opponent is along +X in world)
    net_normal = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    
    cos_theta = torch.sum(forward_dir * net_normal, dim=-1)
    
    return cos_theta


def impact_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize Z-axis linear accelerations on the articulation using L2 squared kernel.

    NOTE: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their linear accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2]), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)