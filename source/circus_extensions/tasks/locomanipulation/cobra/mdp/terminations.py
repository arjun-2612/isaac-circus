# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the cobra box-push environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands.planar_command import PlanarManipulationCommand
    from .commands.box_manipulation_command import BoxManipulationCommand


# def goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     """Terminate when the box satisfies both position and yaw thresholds."""
#     cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
#     pos_ok = cmd.metrics["position_error"] < cmd.cfg.pos_success_threshold
#     yaw_ok = cmd.metrics["heading_error"] < cmd.cfg.yaw_success_threshold
#     return pos_ok  & yaw_ok

def goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the box has been settled at the goal for the required duration."""
    cmd: PlanarManipulationCommand | BoxManipulationCommand = env.command_manager.get_term(command_name)
    return cmd.metrics["settled_steps"] >= cmd._settled_threshold

def box_flipped(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Terminate when the box's +Z axis is no longer pointing up."""
    obj: RigidObject = env.scene[object_cfg.name]
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    z_world = math_utils.quat_apply(obj.data.root_quat_w, z_local)
    return z_world[:, 2] < 0.5