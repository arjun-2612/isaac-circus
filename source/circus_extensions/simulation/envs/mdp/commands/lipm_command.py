from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from circus_extensions.simulation.envs.mdp.actions import LIPMAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .lipm_command_cfg import LIPMCommandCfg

class LIPMCommand(CommandTerm):
    cfg: LIPMCommandCfg

    def __init__(self, cfg: LIPMCommandCfg, env: ManagerBasedEnv):
        self.foot_target_visualizer = VisualizationMarkers(cfg.foot_targets_visualizer_cfg)

        super().__init__(cfg, env)

        self.asset: Articulation = env.scene[cfg.asset_name]
        self._env = env 
        
        self.foot_body_ids = self.asset.find_bodies(cfg.foot_bodies)[0]

        self.foot_targets_w = torch.zeros((self.num_envs, 8), device=self.device)
        self.foot_pos_w = torch.zeros((self.num_envs, 12), device=self.device)
        self._swing_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        pass

    @property
    def foot_targets(self) -> torch.Tensor:
        return self.foot_targets_w
    
    @property
    def foot_positions(self) -> torch.Tensor:
        return self.foot_pos_w
    
    @property
    def swing_mask(self) -> torch.Tensor:
        return self._swing_mask

    """
    Command functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        pass 

    def _update_command(self):  
        act: LIPMAction = self._env.action_manager.get_term(self.cfg.action_name)
        self.foot_targets_w = act.foot_targets
        self.foot_pos_w = act.foot_positions
        self._swing_mask = act.leg_state

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.foot_target_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.asset.is_initialized:
            return
        
        translations, marker_idx = self._resolve_foot_targets_to_spheres()

        self.foot_target_visualizer.visualize(
            translations=translations,
            marker_indices=marker_idx,
        )

    """
    Internal helpers.
    """

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