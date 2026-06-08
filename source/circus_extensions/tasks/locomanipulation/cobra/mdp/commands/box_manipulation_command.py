# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Full 3D goal command — samples both box and goal poses at reset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from ..sampling import sample_face_yaw_quat, sample_valid_xy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import BoxManipulationCommandCfg


class BoxManipulationCommand(CommandTerm):
    """Samples box + goal with random face orientations; tracks 3D settling."""

    cfg: BoxManipulationCommandCfg

    def __init__(self, cfg: BoxManipulationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]

        # goal buffers
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0

        # settling
        self._settled_threshold = round(cfg.required_settled_time / env.step_dt)

        # metrics
        self.metrics["position_error"] = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.metrics["orientation_error"] = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.metrics["settled_steps"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.pos_command_w, self.quat_command_w], dim=-1)

    # ── metrics ──────────────────────────────────────────────────────

    def _update_metrics(self):
        self.metrics["position_error"] = torch.norm(
            self.object.data.root_pos_w[:, :2] - self.pos_command_w[:, :2], dim=-1
        )
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )

        box_speed = torch.norm(self.object.data.root_lin_vel_w[:, :2], dim=-1)
        settled = (
            (self.metrics["position_error"] < self.cfg.pos_success_threshold)
            & (self.metrics["orientation_error"] < self.cfg.orientation_success_threshold)
            & (box_speed < self.cfg.max_goal_speed)
        )
        self.metrics["settled_steps"] = torch.where(
            settled, self.metrics["settled_steps"] + 1, torch.zeros_like(self.metrics["settled_steps"])
        )

    # ── sampling ─────────────────────────────────────────────────────

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return

        cfg = self.cfg
        origins = self._env.scene.env_origins[env_ids]

        # --- sample and place box ---
        box_xy = sample_valid_xy(n, cfg.xy_range, cfg.robot_exclusion, self.device)
        box_quat = sample_face_yaw_quat(n, cfg.yaw_range, self.device)

        box_pos = torch.zeros(n, 3, device=self.device)
        box_pos[:, 0] = box_xy[:, 0] + origins[:, 0]
        box_pos[:, 1] = box_xy[:, 1] + origins[:, 1]
        box_pos[:, 2] = cfg.object_height + origins[:, 2]

        box_vel = torch.zeros(n, 6, device=self.device)
        self.object.write_root_pose_to_sim(torch.cat([box_pos, box_quat], dim=-1), env_ids)
        self.object.write_root_velocity_to_sim(box_vel, env_ids)

        # --- sample goal (avoiding robot and box) ---
        goal_xy = sample_valid_xy(
            n, cfg.xy_range, cfg.robot_exclusion, self.device,
            avoid_xy=box_xy, min_dist=cfg.min_box_goal_dist,
        )
        goal_quat = sample_face_yaw_quat(n, cfg.yaw_range, self.device)

        self.pos_command_w[env_ids, 0] = goal_xy[:, 0] + origins[:, 0]
        self.pos_command_w[env_ids, 1] = goal_xy[:, 1] + origins[:, 1]
        self.pos_command_w[env_ids, 2] = cfg.object_height + origins[:, 2]
        self.quat_command_w[env_ids] = goal_quat

        # invalidate metrics
        self.metrics["position_error"][env_ids] = float("inf")
        self.metrics["orientation_error"][env_ids] = float("inf")
        self.metrics["settled_steps"][env_ids] = 0.0

    def _update_command(self):
        pass

    # ── visualisation ────────────────────────────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_marker",
                    markers={"goal": self.cfg.goal_visualizer_cfg},
                )
                self.goal_marker = VisualizationMarkers(marker_cfg)
            self.goal_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_marker"):
                self.goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_marker.visualize(
            translations=self.pos_command_w,
            orientations=self.quat_command_w,
        )