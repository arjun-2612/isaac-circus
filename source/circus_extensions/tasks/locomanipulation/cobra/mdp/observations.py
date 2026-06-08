# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms: box and goal as (x, y, yaw) in the robot head frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.managers import ObservationTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands.planar_command import PlanarManipulationCommand


def _planar_pose_in_head_frame(
    head_pos_w: torch.Tensor,
    head_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return (x, y, yaw) of *target* expressed in the head frame. Shape (N, 3)."""
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(
        head_pos_w, head_quat_w, target_pos_w, target_quat_w
    )
    _, _, yaw = math_utils.euler_xyz_from_quat(rel_quat)
    return torch.cat([rel_pos[:, :2], yaw.unsqueeze(-1)], dim=-1)


def box_pose_in_head_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Box (x, y, yaw) relative to the cobra head link. Shape (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return _planar_pose_in_head_frame(
        robot.data.root_pos_w, robot.data.root_quat_w,
        obj.data.root_pos_w, obj.data.root_quat_w,
    )


def goal_pose_in_head_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Goal (x, y, yaw) relative to the cobra head link. Shape (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    return _planar_pose_in_head_frame(
        robot.data.root_pos_w, robot.data.root_quat_w,
        cmd.pos_command_w, cmd.quat_command_w,
    )

def box_pose_env(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Box (x, y, yaw) in the environment frame. Shape (N, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos_e = obj.data.root_pos_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(obj.data.root_quat_w)
    return torch.cat([pos_e[:, :2], yaw.unsqueeze(-1)], dim=-1)


def goal_pose_env(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Goal (x, y, yaw) in the environment frame. Shape (N, 3)."""
    cmd: PlanarManipulationCommand = env.command_manager.get_term(command_name)
    pos_e = cmd.pos_command_w - env.scene.env_origins
    _, _, yaw = math_utils.euler_xyz_from_quat(cmd.quat_command_w)
    return torch.cat([pos_e[:, :2], yaw.unsqueeze(-1)], dim=-1)

class VirtualChassisObservations(ManagerTermBase):
    """Box and goal poses expressed in the virtual chassis frame.

    The virtual chassis is defined as:
    - Position: centroid of all body link xy positions
    - Heading: circular mean of all body link yaw angles

    Returns (N, 6): [box_x, box_y, box_yaw, goal_x, goal_y, goal_yaw] in VC frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.params["robot_cfg"].name]
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._command_name: str = cfg.params["command_name"]
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/VirtualChassis")
        marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._vc_marker = VisualizationMarkers(marker_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        command_name: str,
    ) -> torch.Tensor:
        # --- virtual chassis frame ---
        body_pos_w = self._robot.data.body_pos_w        # (N, B, 3)
        body_quat_w = self._robot.data.body_quat_w      # (N, B, 4)
        N, B, _ = body_pos_w.shape

        # position: centroid of all body xy
        vc_xy = body_pos_w[:, :, :2].mean(dim=1)        # (N, 2)

        # heading: circular mean of body yaws
        _, _, yaws = math_utils.euler_xyz_from_quat(body_quat_w.reshape(-1, 4))
        yaws = yaws.reshape(N, B)
        vc_yaw = torch.atan2(yaws.sin().mean(dim=1), yaws.cos().mean(dim=1))

        # assemble VC world-frame pose for subtract_frame_transforms
        vc_pos_w = torch.zeros(N, 3, device=self.device)
        vc_pos_w[:, :2] = vc_xy
        vc_pos_w[:, 2] = env.scene.env_origins[:, 2]
        zeros = torch.zeros_like(vc_yaw)
        vc_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, vc_yaw)

        # --- box in VC frame ---
        box_pos, box_quat = math_utils.subtract_frame_transforms(
            vc_pos_w, vc_quat_w,
            self._object.data.root_pos_w, self._object.data.root_quat_w,
        )
        _, _, box_yaw_vc = math_utils.euler_xyz_from_quat(box_quat)

        # --- goal in VC frame ---
        cmd = env.command_manager.get_term(self._command_name)
        goal_pos, goal_quat = math_utils.subtract_frame_transforms(
            vc_pos_w, vc_quat_w,
            cmd.pos_command_w, cmd.quat_command_w,
        )
        _, _, goal_yaw_vc = math_utils.euler_xyz_from_quat(goal_quat)

        self._vc_marker.visualize(translations=vc_pos_w, orientations=vc_quat_w)

        return torch.cat([
            box_pos[:, :2], box_yaw_vc.unsqueeze(-1),
            goal_pos[:, :2], goal_yaw_vc.unsqueeze(-1),
        ], dim=-1)
