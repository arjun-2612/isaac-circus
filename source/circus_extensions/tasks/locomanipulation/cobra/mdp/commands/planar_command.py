# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Planar (x, y, yaw) goal command — sampled once per episode at reset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PlanarManipulationCommandCfg


class PlanarManipulationCommand(CommandTerm):
    """Fixed planar goal: sampled at reset, held for the episode."""

    cfg: PlanarManipulationCommandCfg

    def __init__(self, cfg: PlanarManipulationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]

        # world-frame goal buffers
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0
        self.yaw_command = torch.zeros(self.num_envs, device=self.device)

        # metrics (updated every step for reward / termination queries)
        self.metrics["position_error"] = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.metrics["heading_error"] = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.metrics["settled_steps"] = torch.zeros(self.num_envs, device=self.device)
        self._settled_threshold = round(cfg.required_settled_time / (env.step_dt))

    # ── properties ───────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        """Goal as (x, y, z, qw, qx, qy, qz).  Shape (N, 7)."""
        return torch.cat([self.pos_command_w, self.quat_command_w], dim=-1)

    # ── internal hooks ───────────────────────────────────────────────

    def _update_metrics(self):
        """Compute position and yaw errors (used by rewards and terminations)."""
        box_xy = self.object.data.root_pos_w[:, :2]
        self.metrics["position_error"] = torch.norm(box_xy - self.pos_command_w[:, :2], dim=-1)

        _, _, box_yaw = math_utils.euler_xyz_from_quat(self.object.data.root_quat_w)
        self.metrics["heading_error"] = math_utils.wrap_to_pi(box_yaw - self.yaw_command).abs()

        box_speed = torch.norm(self.object.data.root_lin_vel_w[:, :2], dim=-1)

        settled = (
            (self.metrics["position_error"] < self.cfg.pos_success_threshold)
            & (self.metrics["heading_error"] < self.cfg.yaw_success_threshold)
            & (box_speed < self.cfg.max_goal_speed)
        )
        self.metrics["settled_steps"] = torch.where(settled, self.metrics["settled_steps"] + 1, torch.zeros_like(self.metrics["settled_steps"]))

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new (x, y, yaw) goal relative to env origins."""
        n = len(env_ids)
        if n == 0:
            return

        cfg = self.cfg
        x = torch.empty(n, device=self.device).uniform_(*cfg.goal_x_range)
        y = torch.empty(n, device=self.device).uniform_(*cfg.goal_y_range)
        yaw = torch.empty(n, device=self.device).uniform_(
            math.radians(cfg.goal_yaw_range[0]),
            math.radians(cfg.goal_yaw_range[1]),
        )

        origins = self._env.scene.env_origins[env_ids]
        self.pos_command_w[env_ids, 0] = x + origins[:, 0]
        self.pos_command_w[env_ids, 1] = y + origins[:, 1]
        self.pos_command_w[env_ids, 2] = cfg.object_height + origins[:, 2]

        self.yaw_command[env_ids] = yaw
        zeros = torch.zeros_like(yaw)
        self.quat_command_w[env_ids] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        self.metrics["position_error"][env_ids] = float("inf")
        self.metrics["heading_error"][env_ids] = float("inf")

    def _update_command(self):
        """No-op: goal is fixed for the entire episode."""
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
