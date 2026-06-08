from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from circus_extensions.simulation.envs.mdp.actions import HuskyBetaMultimodalJointPosAction
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .multimodal_command_cfg import HuskyBetaMultimodalJointPosCommandCfg

class HuskyBetaMultimodalJointPosCommand(CommandTerm):
    cfg: HuskyBetaMultimodalJointPosCommandCfg

    def __init__(self, cfg: HuskyBetaMultimodalJointPosCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._env = env 
        
        self._foot_targets_w = torch.zeros((self.num_envs, 8), device=self.device)
        self._swing_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._velocity_command = torch.zeros((self.num_envs, 4), device=self.device)

        self.time_left_s = torch.zeros((self.num_envs, 1), device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        pass
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mode
    
    @property
    def prev_mode(self) -> torch.Tensor:
        return self._prev_mode

    @property
    def foot_targets(self) -> torch.Tensor:
        return self._foot_targets_w
    
    @property
    def swing_mask(self) -> torch.Tensor:
        return self._swing_mask
    
    @property
    def desired_velocity(self) -> torch.Tensor:
        return self._velocity_command
    
    @property
    def time_left_in_episode_s(self) -> torch.Tensor:
        return self.time_left_s

    """
    Command functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        pass 

    def _update_command(self):  
        act: HuskyBetaMultimodalJointPosAction = self._env.action_manager.get_term(self.cfg.action_name)
        self._mode = act.mode
        self._prev_mode = act.prev_mode
        self._foot_targets_w = act.foot_targets
        self._swing_mask = act.swing_mask
        self._velocity_command = act.processed_actions[:, :4]
        self.time_left_s = act.time_left_s

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xyz_velocity_to_arrow(self._velocity_command[:, :3])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xyz_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :3])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xyz_velocity_to_arrow(self, xyz_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xyz_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xyz_velocity, dim=1) * 3.0
        # arrow-direction
        yaw = torch.atan2(xyz_velocity[:, 1], xyz_velocity[:, 0])
        pitch = torch.atan2(xyz_velocity[:, 2], torch.linalg.norm(xyz_velocity[:, :2], dim=1))

        zeros = torch.zeros_like(yaw)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch, yaw)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
        