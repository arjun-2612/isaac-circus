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
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from circus_extensions.simulation.envs.mdp import ShuttleLauncherCommand
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def swing_target_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the interception position of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    asset: Articulation = env.scene[asset_cfg.name]

    base_pos = asset.data.root_pos_w.clone()
    base_quat = asset.data.root_quat_w.clone()
    shuttle_intercept = cmd.interception_pos
    shuttle_intercept_b = math_utils.quat_rotate_inverse(base_quat, shuttle_intercept - base_pos)

    assert torch.isfinite(shuttle_intercept_b).all(), "NaN or Inf detected in tensor shuttle_interception_pos_b!"
    return shuttle_intercept_b


def swing_target_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the interception velocity of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    asset: Articulation = env.scene[asset_cfg.name]

    base_quat = asset.data.root_quat_w.clone()
    shuttle_vel = cmd.interception_vel
    shuttle_vel_b = math_utils.quat_rotate_inverse(base_quat, shuttle_vel)

    assert torch.isfinite(shuttle_vel_b).all(), "NaN or Inf detected in tensor shuttle_interception_vel_b!"
    return shuttle_vel_b


def time_to_swing(env: ManagerBasedEnv, command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the interception time of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    time_to_intercept = cmd.time_to_intercept
    assert torch.isfinite(time_to_intercept).all(), "NaN or Inf detected in tensor time_to_intercept!"
    return time_to_intercept.unsqueeze(-1)


def shuttlecock_position_gt(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the position of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    asset: Articulation = env.scene[asset_cfg.name]

    base_pos = asset.data.root_pos_w.clone()
    base_quat = asset.data.root_quat_w.clone()
    shuttle_pos = cmd.shuttle_pos
    shuttle_pos_b = math_utils.quat_rotate_inverse(base_quat, shuttle_pos - base_pos)

    assert torch.isfinite(shuttle_pos_b).all(), "NaN or Inf detected in tensor shuttle_pos_b!"
    return shuttle_pos_b


def shuttlecock_velocity_gt(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the velocity of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    asset: Articulation = env.scene[asset_cfg.name]

    shuttle_vel = cmd.shuttle_vel
    base_quat = asset.data.root_quat_w.clone()
    shuttle_vel_b = math_utils.quat_rotate_inverse(base_quat, shuttle_vel)

    assert torch.isfinite(shuttle_vel_b).all(), "NaN or Inf detected in tensor shuttle_vel_b!"
    return shuttle_vel_b

def targets_remaining(env: ManagerBasedEnv, command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the interception position of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    targets_remaining = cmd.targets_remaining
    assert torch.isfinite(targets_remaining).all(), "NaN or Inf detected in tensor targets_remaining!"
    return targets_remaining.unsqueeze(-1)


def next_target_dist(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "shuttle_launcher") -> torch.Tensor:
    """Return the interception position of the shuttle relative to the base."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    asset: Articulation = env.scene[asset_cfg.name]

    base_pos = asset.data.root_pos_w.clone()
    base_quat = asset.data.root_quat_w.clone()
    next_pos = cmd.next_target_pos
    next_pos_b = math_utils.quat_rotate_inverse(base_quat, next_pos - base_pos)

    assert torch.isfinite(next_pos_b).all(), "NaN or Inf detected in tensor next_target_pos_from_b!"
    return next_pos_b

def end_effector_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Return the end effector position relative to the base."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    base_pos_w = asset.data.root_pos_w.clone()
    base_quat_w = asset.data.root_quat_w.clone()
    end_effector_pos_w = ee_frame.data.target_pos_w.clone().squeeze(1)

    end_effector_pos_b = math_utils.quat_rotate_inverse(base_quat_w, end_effector_pos_w - base_pos_w)
    assert torch.isfinite(end_effector_pos_b).all(), "NaN or Inf detected in tensor end_effector_pos_b!"
    return end_effector_pos_b


def end_effector_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the end effector velocity relative to the base."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat_w = asset.data.root_quat_w.clone()
    end_effector_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].clone().squeeze(1)

    end_effector_vel_b = math_utils.quat_rotate_inverse(base_quat_w, end_effector_vel_w)
    assert torch.isfinite(end_effector_vel_b).all(), "NaN or Inf detected in tensor end_effector_vel_b!"
    return end_effector_vel_b


def end_effector_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the end effector orientation in world."""
    asset: Articulation = env.scene[asset_cfg.name]
    end_effector_orient_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :].clone().squeeze(1)
    assert torch.isfinite(end_effector_orient_w).all(), "NaN or Inf detected in tensor end_effector_orient_w!"
    return end_effector_orient_w