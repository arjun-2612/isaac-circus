from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from circus_extensions.simulation.envs.mdp.actions import RMPCAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .rmpc_command_cfg import RMPCCommandCfg

class RMPCCommand(CommandTerm):
    cfg: RMPCCommandCfg

    def __init__(self, cfg: RMPCCommandCfg, env: ManagerBasedEnv):
        self.foot_target_visualizer = VisualizationMarkers(cfg.foot_targets_visualizer_cfg)
        self.swing_target_visualizer = VisualizationMarkers(cfg.swing_targets_visualizer_cfg)
        self.foot_forces_visualizer = VisualizationMarkers(cfg.foot_forces_visualizer_cfg)
        self.applied_foot_forces_visualizer = VisualizationMarkers(cfg.applied_foot_forces_visualizer_cfg)

        super().__init__(cfg, env)

        self.asset: Articulation = env.scene[cfg.asset_name]
        self._env = env 
        
        self.foot_body_ids = self.asset.find_bodies(cfg.foot_bodies)[0]

        self.foot_targets_w = torch.zeros((self.num_envs, 8), device=self.device)
        self.swing_targets_w = torch.zeros((self.num_envs, 12), device=self.device)
        self.foot_pos_w = torch.zeros((self.num_envs, 12), device=self.device)
        self.f_w = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.applied_forces_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.q_desired = torch.zeros((self.num_envs, 12), device=self.device)
        self.q_touchdown = torch.zeros((self.num_envs, 12), device=self.device)
        self._swing_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.desired_body_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.desired_body_quat_w = torch.zeros((self.num_envs, 4), device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        pass

    @property
    def desired_body_state(self) -> torch.Tensor:   
        return torch.cat([self.desired_body_pos, self.desired_body_quat], dim=-1)
    
    @property
    def desired_body_pos(self) -> torch.Tensor:
        return self.desired_body_pos_w
    
    @property
    def desired_body_quat(self) -> torch.Tensor:
        return self.desired_body_quat_w

    @property
    def foot_targets(self) -> torch.Tensor:
        return self.foot_targets_w
    
    @property
    def swing_targets(self) -> torch.Tensor:
        return self.swing_targets_w
    
    @property
    def foot_positions(self) -> torch.Tensor:
        return self.foot_pos_w
    
    @property
    def foot_forces(self) -> torch.Tensor:
        return self.f_w
    
    @property
    def applied_forces(self) -> torch.Tensor:
        return self.applied_forces_w
    
    @property
    def swing_mask(self) -> torch.Tensor:
        return self._swing_mask
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.q_touchdown
    
    @property
    def joint_pos_desired(self) -> torch.Tensor:
        return self.q_desired

    """
    Command functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        pass 

    def _update_command(self):  
        act: RMPCAction = self._env.action_manager.get_term(self.cfg.action_name)
        self.f_w = act.foot_forces
        self.applied_forces_w = act.applied_forces
        self.foot_targets_w = act.foot_targets
        self.swing_targets_w = act.swing_targets
        self.foot_pos_w = act.foot_positions
        self.q_desired = act.joint_pos_desired
        self.q_touchdown = act.joint_pos
        self._swing_mask = act.leg_state
        self.desired_body_pos_w = act.desired_body_pos
        self.desired_body_quat_w = act.desired_body_quat

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.foot_target_visualizer.set_visibility(debug_vis)
        self.swing_target_visualizer.set_visibility(False)
        self.foot_forces_visualizer.set_visibility(False)
        self.applied_foot_forces_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.asset.is_initialized:
            return
        
        translations, marker_idx = self._resolve_foot_targets_to_spheres()

        self.foot_target_visualizer.visualize(
            translations=translations,
            marker_indices=marker_idx,
        )

        translations, marker_idx = self._resolve_swing_targets_to_spheres()

        self.swing_target_visualizer.visualize(
            translations=translations,
            marker_indices=marker_idx,
        )

        foot_pos_w, foot_quat_w, foot_scale, marker_idx = self._resolve_foot_forces_to_arrows(self.f_w)
        applied_foot_pos_w, applied_foot_quat_w, applied_foot_scale, marker_idx2 = self._resolve_foot_forces_to_arrows(self.applied_forces_w)

        self.foot_forces_visualizer.visualize(
            translations=foot_pos_w, 
            orientations=foot_quat_w,
            scales=foot_scale,
            marker_indices=marker_idx,
        )

        self.applied_foot_forces_visualizer.visualize(
            translations=applied_foot_pos_w, 
            orientations=applied_foot_quat_w,
            scales=applied_foot_scale,
            marker_indices=marker_idx2,
        )

    """
    Internal helpers.
    """

    def _resolve_swing_targets_to_spheres(self):
        """
        Draw spheres at foot target positions using POSITION_GOAL_MARKER_CFG.
        """
        E = self.num_envs
        L = len(self.foot_body_ids)
        ft_w = self.swing_targets_w

        translations = ft_w.reshape(E * L, 3)                    # (M,3)
        marker_indices = torch.full((E * L,), 0, dtype=torch.int64, device=self.device)  # 0 = red

        return translations, marker_indices

    def _resolve_foot_targets_to_spheres(self):
        """
        Draw spheres at foot target positions using POSITION_GOAL_MARKER_CFG.
        """
        E = self.num_envs
        L = len(self.foot_body_ids)
        ft_w = self.foot_targets_w
        z = torch.zeros((E, L, 1), device=self.device)
        ft_w = torch.cat([ft_w.reshape(E, L, 2), z], dim=-1)  # (E,L,3)

        translations = ft_w.reshape(E * L, 3)                    # (M,2)
        marker_indices = torch.full((E * L,), 0, dtype=torch.int64, device=self.device)  # 0 = red

        return translations, marker_indices
    
    def _quat_align_x_to_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Minimal-rotation quaternion that rotates +X axis onto vector v.
        v: (..., 3) world vector
        returns: (..., 4) quaternion (w, x, y, z) in WORLD frame
        """
        # normalize
        eps = torch.finfo(v.dtype).eps
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
        u = v / v_norm

        # reference axis = +X
        ex = torch.zeros_like(v)
        ex[..., 0] = 1.0

        # handle near-parallel numerically: if u ~ +ex => identity; if u ~ -ex => 180 deg about Y (or Z)
        dot = (ex * u).sum(dim=-1, keepdim=True)  # (...,1)
        # general case
        axis = torch.cross(ex, u, dim=-1)         # (...,3)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        # angle via atan2 for stability
        angle = torch.atan2(axis_norm, dot.clamp(-1.0, 1.0))     # (...,1)

        # default axis if parallel/antiparallel
        # when axis_norm ~ 0 and dot < 0 (antiparallel), rotate 180° about +Y
        fallback_axis = torch.zeros_like(axis)
        fallback_axis[..., 1] = 1.0

        safe_axis = torch.where(axis_norm > 1e-8, axis / axis_norm, fallback_axis)
        half = 0.5 * angle
        s = torch.sin(half)
        q = torch.cat([torch.cos(half), safe_axis * s], dim=-1)  # (w, x, y, z)
        return q

    def _resolve_foot_forces_to_arrows(self, F_w):
        """
        Build pose+scale for four foot arrows per env.
        Returns:
            pos_w:  (E, 4, 3)
            quat_w: (E, 4, 4)  (w,x,y,z), arrow +X -> force dir
            scale:  (E, 4, 3)  (length,x-width,y-width) in meters
        """
        E = self.num_envs
        L = len(self.foot_body_ids)

        if F_w.dim() == 2:
            F_w = torch.stack([
                torch.zeros_like(F_w),
                torch.zeros_like(F_w),
                F_w,
            ], dim=-1)  # (E, 4, 3)

        # foot world positions
        foot_pos_w = self.asset.data.body_pos_w[:, self.foot_body_ids].clone()  # (E,4,3)

        # arrow orientation: align +X with force vector (fallback to identity if ~0)
        quat_w = self._quat_align_x_to_v(F_w)                                        # (E,4,4)

        # arrow scale: length ∝ |F|, thickness from default marker scale
        default_scale = torch.tensor(
            self.foot_forces_visualizer.cfg.markers["fl"].scale,  # (3,)
            device=self.device, dtype=torch.float32
        )
        # repeat per foot
        scale = default_scale.view(1, 1, 3).repeat(F_w.shape[0], 4, 1)          # (E,4,3)
        F_mag = torch.linalg.norm(F_w, dim=-1)                                  # (E,4)

        scale[..., 0] = torch.clamp(0.25 * F_mag, min=0.01)

        marker_indices = torch.full((E * L,), 0, dtype=torch.int64, device=self.device)  # 0 = red

        return foot_pos_w.reshape(E*L, 3), quat_w.reshape(E*L, 4), scale.reshape(E*L, 3), marker_indices
