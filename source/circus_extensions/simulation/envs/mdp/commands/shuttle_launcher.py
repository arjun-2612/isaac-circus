from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .shuttle_launcher_cfg import ShuttleLauncherCommandCfg

class ShuttleLauncherCommand(CommandTerm):
    cfg: ShuttleLauncherCommandCfg

    def __init__(self, cfg: ShuttleLauncherCommandCfg, env: ManagerBasedEnv):
        self.shuttle_target_visualizer = VisualizationMarkers(cfg.shuttle_target_visualizer_cfg)
        self.shuttle_pos_visualizer = VisualizationMarkers(cfg.shuttle_pos_visualizer_cfg)
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._env = env 
        
        # Buffers
        self.shuttle_init_pos_w = torch.zeros((self.num_envs, cfg.num_targets, 3), device=self.device)  # (E,3)
        self.shuttle_init_vel_w = torch.zeros((self.num_envs, cfg.num_targets, 3), device=self.device)  # (E,3)
        self.shuttle_interception_pos_w = torch.zeros((self.num_envs, cfg.num_targets, 3), device=self.device)
        self.shuttle_interception_vel_w = torch.zeros((self.num_envs, cfg.num_targets, 3), device=self.device)
        self.time_to_interception = torch.zeros((self.num_envs, cfg.num_targets), device=self.device)  # (E,)

        # Stepping tensors
        self.shuttle_pos_w = torch.zeros((self.num_envs, 3), device=self.device)  # (E,3)
        self.shuttle_vel_w = torch.zeros((self.num_envs, 3), device=self.device)  # (E,3)
        self.current_intercept_pos_w = torch.zeros((self.num_envs, 3), device=self.device)  # (E,3)
        self.current_intercept_vel_w = torch.zeros((self.num_envs, 3), device=self.device)  # (E,3)
        self.current_intercept_time = torch.zeros((self.num_envs,), device=self.device)  # (E,)
        self.at_intercept = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) # (E,)
        self.next_shuttle_pos_w = torch.zeros((self.num_envs, 3), device=self.device)  # (E,3)

        # Other variables
        self.targets_generated = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)  # (E,)
        self.time_since_last_resample = torch.zeros((self.num_envs,), device=self.device)
        self.k = torch.full((self.num_envs,), 0.55, device=self.device)
        self.total_targets = torch.full((self.num_envs,), cfg.num_targets, dtype=torch.int32, device=self.device)
        self.success_buffer = torch.zeros((self.num_envs, cfg.num_targets), device=self.device)
        self.level = torch.zeros((self.num_envs,), device=self.device)
        self.swing_speed_target = torch.full((self.num_envs,), 4.0, device=self.device)
        self.swing_speed_ema = torch.full((self.num_envs,), 4.0 / 1.25, device=self.device)
        self.achieved_racket_speed = torch.zeros((self.num_envs,), device=self.device)
        self.planned_intercept_time = torch.zeros((self.num_envs,), device=self.device)

        # Constants
        self.g = 9.81
        self.dt = self._env.step_dt
        self.exp_kdt = torch.exp(-self.k * self.dt)
        self.intercept_height = cfg.intercept_height
        self.time_between_targets = cfg.time_between_targets
        self.max_prediction_time = 3.5
        self.active_env_ids = torch.tensor([], device=self.device, dtype=torch.int64)

        # Metrics
        self.metrics["episode_success_rate"] = torch.zeros((self.num_envs,), device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        pass

    @property
    def interception_pos(self) -> torch.Tensor:
        return self.current_intercept_pos_w
    
    @property
    def interception_vel(self) -> torch.Tensor:
        return self.current_intercept_vel_w
    
    @property
    def shuttle_pos(self) -> torch.Tensor:
        return self.shuttle_pos_w
    
    @property
    def shuttle_vel(self) -> torch.Tensor:
        return self.shuttle_vel_w
    
    @property
    def time_to_intercept(self) -> torch.Tensor:
        return self.current_intercept_time
    
    @property
    def targets_remaining(self) -> torch.Tensor:
        return self.total_targets - self.targets_generated
    
    @property
    def next_target_pos(self) -> torch.Tensor:
        return self.next_shuttle_pos_w
    
    @property
    def time_since_intercept(self) -> torch.Tensor:
        # negative pre-impact, 0 at impact, positive after
        return self.time_since_last_resample - self.planned_intercept_time
    
    @property
    def at_intercept_time(self) -> torch.Tensor:
        # True for exactly one step per target, at the interception instant.
        return self.at_intercept

    """
    Command functions.
    """

    def _update_metrics(self):
        racket_pos_w = self._env.scene[self.cfg.ee_frame.name].data.target_pos_w.clone().squeeze(1)  # (E,3)
        pos_error = torch.norm(racket_pos_w - self.current_intercept_pos_w, dim=-1)  # (E,)

        at_int_time = self.at_intercept_time
        if at_int_time.any():
            env_ids = at_int_time.nonzero(as_tuple=False).squeeze(-1)
            target_id = self.targets_generated[env_ids].long()
            self.success_buffer[env_ids, target_id] = (pos_error[env_ids] < self.cfg.success_threshold).float()

        self.metrics["episode_success_rate"] = self.success_buffer.sum(dim=-1) / self.cfg.num_targets

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        standing_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_env_ratio
        new_standing = env_ids[standing_mask]
        new_active = env_ids[~standing_mask]
        keep_active = self.active_env_ids[~torch.isin(self.active_env_ids, env_ids)]
        self.active_env_ids = torch.cat([keep_active, new_active])

        self.time_since_last_resample[env_ids] = 0.0
        self.success_buffer[env_ids, :] = 0.0
        
        # Handle any new standing envs 
        if len(new_standing) > 0:
            self.targets_generated[new_standing] = self.total_targets[new_standing]  # Already "done"
            self.time_to_interception[new_standing] = self.max_prediction_time
            self.shuttle_interception_pos_w[new_standing, :, :] = self.robot.data.root_pos_w[new_standing].clone().unsqueeze(1)  # Just set to robot position
            self.shuttle_interception_pos_w[new_standing, :, 2] = self.intercept_height  # Set to intercept height
            self.shuttle_interception_vel_w[new_standing, :, :] = torch.zeros((len(new_standing), self.cfg.num_targets, 3), device=self.device)  # Just set to zero velocity

        # Handle any new active envs
        if len(new_active) > 0:
            lvl = self.level[new_active].clone().unsqueeze(-1)
            self.targets_generated[new_active] = 0
            # sample velocity commands
            r = torch.empty((len(new_active), self.cfg.num_targets), device=self.device)
            r_unit = torch.rand((len(new_active), self.cfg.num_targets), device=self.device)
            # vx: (-7.5, -7.5) -> (-10, -6)
            vx_lo = -7.5 - 2.5 * lvl
            vx_hi = -7.5 + 2.5 * lvl
            # pz: (1.5, 2.0) -> (0.5, 2.5)
            pz_lo = torch.clamp(1.5 - 1.0 * lvl, min=0.3)  # (N, 1)
            pz_hi = 2.0 + 0.5 * lvl  # (N, 1)
            # py: (0, 0) -> (-1, 1)
            py_lo = self.cfg.ranges.py[0] * lvl
            py_hi = self.cfg.ranges.py[1] * lvl
            # vy: (0, 0) -> (-2, 2)
            vy_lo = self.cfg.ranges.vy[0] * lvl
            vy_hi = self.cfg.ranges.vy[1] * lvl

            # -- position - x direction
            self.shuttle_init_pos_w[new_active, :, 0] = r.uniform_(*self.cfg.ranges.px)
            # -- position - y direction
            self.shuttle_init_pos_w[new_active, :, 1] = py_lo + r_unit * (py_hi - py_lo)
            # -- position - z direction
            self.shuttle_init_pos_w[new_active, :, 2] = pz_lo + r_unit * (pz_hi - pz_lo)
            # -- linear velocity - x direction
            self.shuttle_init_vel_w[new_active, :, 0] = vx_lo + r_unit * (vx_hi - vx_lo)
            # -- linear velocity - y direction
            self.shuttle_init_vel_w[new_active, :, 1] = vy_lo + r_unit * (vy_hi - vy_lo)
            # -- linear velocity - z direction
            self.shuttle_init_vel_w[new_active, :, 2] = r.uniform_(*self.cfg.ranges.vz)

            self.shuttle_init_pos_w[new_active, :, :2] += self.robot.data.root_pos_w[new_active, :2].unsqueeze(1)
            
            # Interception time as shuttle propagates
            self.time_to_interception[new_active] = self.time_to_height_linear_drag(
                z0=self.shuttle_init_pos_w[new_active, :, 2],
                vz0=self.shuttle_init_vel_w[new_active, :, 2],
                z_int=torch.full_like(self.shuttle_init_pos_w[new_active, :, 2], self.intercept_height),
                k=self.k[new_active].unsqueeze(-1),
                dt_max=self.max_prediction_time,
            )

            # Interception position and velocity XYZ of shuttle 
            k = self.k[new_active].unsqueeze(-1)
            exp_kt = torch.exp(-k * self.time_to_interception[new_active])
            
            self.shuttle_interception_pos_w[new_active, :, 0] = (
                self.shuttle_init_pos_w[new_active, :, 0] 
                + (self.shuttle_init_vel_w[new_active, :, 0] / k) 
                * (1.0 - exp_kt)
            )
            self.shuttle_interception_pos_w[new_active, :, 1] = (
                self.shuttle_init_pos_w[new_active, :, 1] 
                + (self.shuttle_init_vel_w[new_active, :, 1] / k) 
                * (1.0 - exp_kt)
            )
            self.shuttle_interception_pos_w[new_active, :, 2] = (
                self.shuttle_init_pos_w[new_active, :, 2]
                + (1.0 / k) * (self.shuttle_init_vel_w[new_active, :, 2] 
                + (self.g / k)) * (1.0 - exp_kt)
                - (self.g / k) * self.time_to_interception[new_active]
            )

            self.shuttle_interception_vel_w[new_active, :, 0] = (
                self.shuttle_init_vel_w[new_active, :, 0] * exp_kt
            )
            self.shuttle_interception_vel_w[new_active, :, 1] = (
                self.shuttle_init_vel_w[new_active, :, 1] * exp_kt
            )
            self.shuttle_interception_vel_w[new_active, :, 2] = (
                (self.shuttle_init_vel_w[new_active, :, 2] 
                + (self.g / k)) * exp_kt 
                - (self.g / k)
            )

        # Set the values for the first target
        self.shuttle_pos_w[env_ids] = self.shuttle_init_pos_w[env_ids, 0].clone()
        self.shuttle_vel_w[env_ids] = self.shuttle_init_vel_w[env_ids, 0].clone()
        self.next_shuttle_pos_w[env_ids] = self.shuttle_init_pos_w[env_ids, 1].clone()
        self.current_intercept_pos_w[env_ids] = self.shuttle_interception_pos_w[env_ids, 0].clone()
        self.current_intercept_vel_w[env_ids] = self.shuttle_interception_vel_w[env_ids, 0].clone()
        self.current_intercept_time[env_ids] = self.time_to_interception[env_ids, 0].squeeze(-1).clone()
        self.planned_intercept_time[env_ids] = self.current_intercept_time[env_ids].clone()
        self.achieved_racket_speed[env_ids] = 0.0
        self.at_intercept[env_ids] = False

    def _update_command(self):
        mask = self.time_since_last_resample >= self.time_between_targets
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()
        if env_ids.numel() > 0:
            common_mask = torch.isin(env_ids, self.active_env_ids)
            common_ids = env_ids[common_mask]
            self.targets_generated[common_ids] += 1
            self.time_since_last_resample[common_ids] = 0.0
            self.targets_generated = torch.clip(self.targets_generated, max=self.total_targets)
            # Only load a new shuttle if there is a valid next target (guard against
            # the last timer firing at episode end when targets_generated == total_targets).
            new_target_mask = self.targets_generated[common_ids] < self.total_targets[common_ids]
            valid_ids = common_ids[new_target_mask]
            if valid_ids.numel() > 0:
                t_idx = self.targets_generated[valid_ids]
                self.shuttle_pos_w[valid_ids] = self.shuttle_init_pos_w[valid_ids, t_idx].clone()
                self.shuttle_vel_w[valid_ids] = self.shuttle_init_vel_w[valid_ids, t_idx].clone()
                self.next_shuttle_pos_w[valid_ids] = self.shuttle_init_pos_w[valid_ids, torch.clip(t_idx + 1, max=self.total_targets[valid_ids] - 1)].clone()
                self.current_intercept_time[valid_ids] = self.time_to_interception[valid_ids, t_idx].squeeze(-1).clone()
                self.planned_intercept_time[valid_ids] = self.current_intercept_time[valid_ids].clone()
                self.current_intercept_pos_w[valid_ids] = self.shuttle_interception_pos_w[valid_ids, t_idx].clone()
                self.current_intercept_vel_w[valid_ids] = self.shuttle_interception_vel_w[valid_ids, t_idx].clone()

        # Step the shuttle trajectory
        self.shuttle_pos_w[self.active_env_ids, 0] += (self.shuttle_vel_w[self.active_env_ids, 0]/self.k[self.active_env_ids]) * (1.0 - self.exp_kdt[self.active_env_ids])
        self.shuttle_pos_w[self.active_env_ids, 1] += (self.shuttle_vel_w[self.active_env_ids, 1]/self.k[self.active_env_ids]) * (1.0 - self.exp_kdt[self.active_env_ids])
        self.shuttle_pos_w[self.active_env_ids, 2] += (1.0/self.k[self.active_env_ids])*(self.shuttle_vel_w[self.active_env_ids, 2] + (self.g/self.k[self.active_env_ids]))*(1.0 - self.exp_kdt[self.active_env_ids]) - (self.g/self.k[self.active_env_ids])*self.dt

        self.shuttle_vel_w[self.active_env_ids, 0] *= self.exp_kdt[self.active_env_ids]
        self.shuttle_vel_w[self.active_env_ids, 1] *= self.exp_kdt[self.active_env_ids]
        self.shuttle_vel_w[self.active_env_ids, 2] = (self.shuttle_vel_w[self.active_env_ids, 2] + (self.g/self.k[self.active_env_ids]))*self.exp_kdt[self.active_env_ids] - (self.g/self.k[self.active_env_ids])

        self.time_since_last_resample[self.active_env_ids] += self.dt

        before = self.current_intercept_time[self.active_env_ids]
        after = torch.clamp(before - self.dt, min=0.0)

        self.at_intercept[self.active_env_ids] = (before > 0.0) & (after <= 0.0)
        self.current_intercept_time[self.active_env_ids] = after

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.shuttle_target_visualizer.set_visibility(debug_vis)
        self.shuttle_pos_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        self.shuttle_pos_visualizer.visualize(self.shuttle_pos_w)
        self.shuttle_target_visualizer.visualize(self.current_intercept_pos_w)

    """
    Internal helpers.
    """

    def time_to_height_linear_drag(self, z0, vz0, z_int, k, dt_max=5.0, iters=20, g=9.81):
        k = torch.clamp(k, min=1e-6)

        def z_of_t(t):
            exp_term = torch.exp(-k * t)
            return z0 + (1.0 / k) * (vz0 + g / k) * (1.0 - exp_term) - (g / k) * t

        def f(t):
            return z_of_t(t) - z_int

        # apex time (where vz(t)=0)
        numer = k * vz0 + g  # = g + k*vz0
        valid_apex = numer > 1e-6
        t_apex = (1.0 / k) * torch.log(torch.clamp(numer / g, min=1e-6))
        t_apex = torch.where((vz0 > 0.0) & valid_apex, t_apex, torch.zeros_like(t_apex))
        t_start = torch.clamp(t_apex, 0.0, dt_max)

        # Bracket root on [t_start, dt_max]
        t_lo = t_start
        t_hi = torch.full_like(z0, dt_max)

        f_lo = f(t_lo)
        f_hi = f(t_hi)

        # Need sign change AND must be descending at the solution (searching after apex enforces that)
        has_down = (f_lo * f_hi) <= 0.0

        # Bisection only where has_down
        tL = torch.where(has_down, t_lo, t_hi)
        tH = torch.where(has_down, t_hi, t_hi)
        fL = torch.where(has_down, f_lo, f_hi)
        fH = torch.where(has_down, f_hi, f_hi)

        for _ in range(iters):
            tM = 0.5 * (tL + tH)
            fM = f(tM)
            left = (fL * fM) <= 0.0
            tH = torch.where(left, tM, tH)
            fH = torch.where(left, fM, fH)
            tL = torch.where(left, tL, tM)
            fL = torch.where(left, fL, fM)

        t_intercept = 0.5 * (tL + tH)
        t_intercept = torch.where(has_down, t_intercept, torch.full_like(t_intercept, dt_max))
        t_intercept = torch.clamp(t_intercept, 0.0, dt_max)

        return t_intercept