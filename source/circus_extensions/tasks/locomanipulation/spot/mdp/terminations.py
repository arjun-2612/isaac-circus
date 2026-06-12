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

from circus_extensions.simulation.envs.mdp import ShuttleLauncherCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def missed_shuttle(
    env: ManagerBasedEnv, 
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), 
    command_name: str = "shuttle_launcher",
    success_threshold: float = 0.3,
) -> torch.Tensor:
    """Terminate if the shuttle is missed."""
    cmd: ShuttleLauncherCommand = env.command_manager.get_term(command_name) 
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w.squeeze(1)
    intercept_pos_w = cmd.current_intercept_pos_w

    missed = cmd.at_intercept_time & (torch.norm(intercept_pos_w - ee_pos_w, dim=-1) > success_threshold)
    return missed


def fallen_over(
    env: ManagerBasedEnv,
    limit_angle: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the robot is fallen over, defined as having a tilt angle greater than `limit_angle`."""
    asset = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0))
    too_tilted = tilt > limit_angle
    return too_tilted 
