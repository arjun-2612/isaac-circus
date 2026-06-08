# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for planar box manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands.planar_command import PlanarManipulationCommand
    from .commands.box_manipulation_command import BoxManipulationCommand

# ── helpers ──────────────────────────────────────────────────────────────

def _keypoints_from_pose(pos: torch.Tensor, quat: torch.Tensor, half_side: float) -> torch.Tensor:
    """Three axis-aligned keypoints from a 6-DoF pose. Shape (N, 3, 3): [batch, keypoint, xyz]."""
    offsets = torch.tensor(
        [[half_side, 0.0, 0.0], [0.0, half_side, 0.0], [0.0, 0.0, half_side]],
        device=pos.device, dtype=pos.dtype,
    )
    N = pos.shape[0]
    rotated = math_utils.quat_apply(
        quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4),
        offsets.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3),
    ).reshape(N, 3, 3)
    return pos.unsqueeze(1) + rotated


def track_pos_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Gaussian reward on xy distance between box and goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    d2 = torch.sum((obj.data.root_pos_w[:, :2] - cmd.pos_command_w[:, :2]) ** 2, dim=-1)
    return torch.exp(-d2 / (sigma**2))


def track_yaw_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Gaussian reward on yaw error between box and goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    _, _, box_yaw = math_utils.euler_xyz_from_quat(obj.data.root_quat_w)
    err = math_utils.wrap_to_pi(box_yaw - cmd.yaw_command)
    return torch.exp(-(err**2) / (sigma**2))


# def manipulation_success_bonus(
#     env: ManagerBasedRLEnv,
#     command_name: str,
# ) -> torch.Tensor:
#     """Binary 1.0 when the box is at the goal (position *and* yaw)."""
#     cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
#     pos_ok = cmd.metrics["position_error"] < cmd.cfg.pos_success_threshold
#     yaw_ok = cmd.metrics["heading_error"] < cmd.cfg.yaw_success_threshold
#     return (pos_ok & yaw_ok).float()

def manipulation_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Binary 1.0 when the box is settled at the goal."""
    cmd: PlanarManipulationCommand | BoxManipulationCommand = env.command_manager.get_term(command_name)
    return (cmd.metrics["settled_steps"] > 0).float()

def approach_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    robot_body_positions = robot.data.body_pos_w
    target_position = obj.data.root_pos_w

    distances = torch.norm(robot_body_positions - target_position[:, None, :], dim=-1)
    
    closest_distance, _ = torch.min(distances, dim=-1)

    return 1.0 - torch.tanh(closest_distance / std)

def box_stationary_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Returns ~1.0 when box is still, decays to ~0.0 when moving."""
    obj: RigidObject = env.scene[object_cfg.name]
    speed = torch.norm(obj.data.root_lin_vel_w[:, :2], dim=-1)
    return torch.exp(-speed / threshold)

def box_vel_toward_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Positive reward when the box moves toward the goal, zero otherwise."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    # unit vector from box to goal
    delta = cmd.pos_command_w[:, :2] - obj.data.root_pos_w[:, :2]
    dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-4)
    direction = delta / dist
    # project box velocity onto that direction
    vel_toward = torch.sum(obj.data.root_lin_vel_w[:, :2] * direction, dim=-1)
    return vel_toward.clamp(min=0.0)

def joint_activity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean absolute joint velocity — rewards joints that are moving."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.joint_vel), dim=-1)

def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step — weighted negatively to create urgency."""
    return torch.ones(env.num_envs, device=env.device)

def track_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Negative xy distance from box to goal — constant gradient everywhere."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    return torch.norm(obj.data.root_pos_w[:, :2] - cmd.pos_command_w[:, :2], dim=-1)

def track_yaw_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Absolute yaw error between box and goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    _, _, box_yaw = math_utils.euler_xyz_from_quat(obj.data.root_quat_w)
    return math_utils.wrap_to_pi(box_yaw - cmd.yaw_command).abs()

def goal_completion_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-time 1.0 on the step the box completes settling at the goal."""
    cmd: PlanarManipulationCommand | BoxManipulationCommand = env.command_manager.get_term(command_name)
    return (cmd.metrics["settled_steps"] >= cmd._settled_threshold).float()

def _keypoints_from_pose(pos: torch.Tensor, quat: torch.Tensor, half_side: float) -> torch.Tensor:
    """Three axis-aligned keypoints from a 6-DoF pose. Shape (N, 3, 3): [batch, keypoint, xyz]."""
    offsets = torch.tensor(
        [[half_side, 0.0, 0.0], [0.0, half_side, 0.0], [0.0, 0.0, half_side]],
        device=pos.device, dtype=pos.dtype,
    )
    N = pos.shape[0]
    rotated = math_utils.quat_apply(
        quat.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4),
        offsets.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3),
    ).reshape(N, 3, 3)
    return pos.unsqueeze(1) + rotated


def box_keypoint_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    half_side: float = 0.1,
) -> torch.Tensor:
    """Mean L2 keypoint distance — constant gradient everywhere."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand | BoxManipulationCommand = env.command_manager.get_term(command_name)

    box_kps = _keypoints_from_pose(obj.data.root_pos_w, obj.data.root_quat_w, half_side)
    goal_kps = _keypoints_from_pose(cmd.pos_command_w, cmd.quat_command_w, half_side)

    per_kp_dist = torch.norm(box_kps - goal_kps, dim=-1)
    return per_kp_dist.mean(dim=-1)


def box_keypoint_sqrt(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    half_side: float = 0.1,
) -> torch.Tensor:
    """Mean sqrt keypoint distance — gradient intensifies near the goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)

    box_kps = _keypoints_from_pose(obj.data.root_pos_w, obj.data.root_quat_w, half_side)
    goal_kps = _keypoints_from_pose(cmd.pos_command_w, cmd.quat_command_w, half_side)

    per_kp_dist = torch.norm(box_kps - goal_kps, dim=-1)
    return per_kp_dist.sqrt().mean(dim=-1)

# ── success / completion ─────────────────────────────────────────────────

def box_upright(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Returns 1.0 when box +Z points down (flipped), 0.0 when upright."""
    obj: RigidObject = env.scene[object_cfg.name]
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    z_world = math_utils.quat_apply(obj.data.root_quat_w, z_local)
    # return (z_world[:, 2] < 0.2).float()
    return 1.0 - z_world[:, 2].clamp(0.0, 1.0)