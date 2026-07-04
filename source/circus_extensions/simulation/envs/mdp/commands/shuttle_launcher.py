from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
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
        self.cfg.asset_cfg.resolve(env.scene)

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
        self.L = torch.full((self.num_envs,), cfg.aero_length, device=self.device)  # aerodynamic length
        self.total_targets = torch.full((self.num_envs,), cfg.num_targets, dtype=torch.int32, device=self.device)
        self.success_buffer = torch.zeros((self.num_envs, cfg.num_targets), device=self.device)
        self.level = torch.zeros((self.num_envs,), device=self.device)
        self.swing_speed_target = torch.full((self.num_envs,), cfg.racket_swing_speed_target, device=self.device)
        self.swing_speed_ema = torch.full((self.num_envs,), cfg.racket_swing_speed_target / 1.25, device=self.device)
        self.achieved_racket_speed = torch.zeros((self.num_envs, 3), device=self.device)
        self.planned_intercept_time = torch.zeros((self.num_envs,), device=self.device)

        # Predicted-target measurement bias (persistent per shuttle flight; see cfg meas_*_noise)
        self.meas_pos_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self.meas_vel_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self.meas_time_rel_bias = torch.zeros((self.num_envs,), device=self.device)

        # Constants
        self.g = 9.81
        self.g_vec = torch.tensor([0.0, 0.0, -self.g], device=self.device)  # gravity as a vector
        self.dt = self._env.step_dt
        self.intercept_height = cfg.intercept_height
        self.time_between_targets = cfg.time_between_targets
        self.restitution = torch.full((self.num_envs,), cfg.restitution, device=self.device)  # per-env, DR-able
        self.contact_radius = cfg.contact_radius
        self.meas_pos_noise = cfg.meas_pos_noise
        self.meas_vel_noise = cfg.meas_vel_noise
        self.meas_time_rel_noise = cfg.meas_time_rel_noise
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
    
    @property
    def racket_speed_at_intercept(self) -> torch.Tensor:
        return self.achieved_racket_speed

    @property
    def predicted_intercept_pos(self) -> torch.Tensor:
        # Emulated onboard-predictor interception (world frame): GT interception + a persistent
        # per-target measurement bias, with the velocity-error term scaled by time-to-go so the
        # estimate converges to the truth as the shuttle approaches. This is the target the policy
        # sees; the deployed robot computes it by integrating the drag model from the mocap state.
        return (
            self.current_intercept_pos_w
            + self.meas_pos_bias
            + self.meas_vel_bias * self.current_intercept_time.unsqueeze(-1)
        )

    @property
    def predicted_intercept_time(self) -> torch.Tensor:
        return self.current_intercept_time * (1.0 + self.meas_time_rel_bias)

    @property
    def intercept_pos_prediction_error(self) -> torch.Tensor:
        return self.predicted_intercept_pos - self.current_intercept_pos_w

    @property
    def is_shuttle_active(self) -> torch.Tensor:
        active = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        active[self.active_env_ids] = True
        return active & (self.targets_generated < self.total_targets)

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
            # vx: curriculum ramps from the cfg range's center outward to its full extent at lvl=1.
            vx_center = 0.5 * (self.cfg.ranges.vx[0] + self.cfg.ranges.vx[1])
            vx_half = 0.5 * (self.cfg.ranges.vx[1] - self.cfg.ranges.vx[0])
            vx_lo = vx_center - vx_half * lvl
            vx_hi = vx_center + vx_half * lvl
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

            # Interception time / pos / vel by forward-integrating the quadratic-drag dynamics
            # until each shuttle descends through the interception height.
            t_int, pos_int, vel_int = self._predict_interception(
                self.shuttle_init_pos_w[new_active],
                self.shuttle_init_vel_w[new_active],
                self.L[new_active],
            )
            self.time_to_interception[new_active] = t_int
            self.shuttle_interception_pos_w[new_active] = pos_int
            self.shuttle_interception_vel_w[new_active] = vel_int

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
        self._resample_measurement_bias(env_ids)

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
                self._resample_measurement_bias(valid_ids)

        # Step the shuttle trajectory one control step under quadratic drag.
        active = self.active_env_ids
        inv_L = (1.0 / self.L[active]).unsqueeze(-1)
        new_pos, new_vel = self._drag_rk4_step(
            self.shuttle_pos_w[active], self.shuttle_vel_w[active], self.dt, inv_L
        )
        self.shuttle_pos_w[active] = new_pos
        self.shuttle_vel_w[active] = new_vel

        self.time_since_last_resample[self.active_env_ids] += self.dt

        before = self.current_intercept_time[self.active_env_ids]
        after = torch.clamp(before - self.dt, min=0.0)

        self.current_intercept_time[self.active_env_ids] = after
        self.at_intercept[self.active_env_ids] = (before > 0.0) & (after <= 0.0)

        # Get racket speed at interception point, leave others unchanged
        self.achieved_racket_speed[self.active_env_ids] = torch.where(
            self.at_intercept[self.active_env_ids].unsqueeze(-1),
            self.robot.data.body_vel_w[self.active_env_ids, self.cfg.asset_cfg.body_ids, :3].squeeze(1),
            self.achieved_racket_speed[self.active_env_ids],
        )

        # Deflect the shuttle off the racket at the interception instant so the post-impact
        # trajectory reflects the actual swing (racket speed + face orientation).
        self._apply_racket_bounce(self.at_intercept.nonzero(as_tuple=False).flatten())

    def _resample_measurement_bias(self, env_ids: torch.Tensor):
        """Sample a fresh per-target measurement bias, held constant over the shuttle's flight.

        Emulates the onboard interception predictor's error at deployment: the bias is persistent
        per shuttle (a biased velocity estimate stays biased), and its effect on the predicted
        target is scaled by time-to-go (see ``predicted_intercept_pos``), so the target converges
        smoothly to the truth as the shuttle approaches rather than jittering independently.
        """
        n = len(env_ids)
        self.meas_pos_bias[env_ids] = (2.0 * torch.rand((n, 3), device=self.device) - 1.0) * self.meas_pos_noise
        self.meas_vel_bias[env_ids] = (2.0 * torch.rand((n, 3), device=self.device) - 1.0) * self.meas_vel_noise
        self.meas_time_rel_bias[env_ids] = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.meas_time_rel_noise

    def _apply_racket_bounce(self, hit_ids: torch.Tensor):
        """Deflect the shuttle off the racket at the moment of interception.

        Models the impact as an elastic collision against an infinitely-heavy moving plane
        (racket inertia >> shuttle inertia): in the racket's frame the shuttle-velocity
        component along the racket face normal is reversed (scaled by the restitution) while
        the tangential component is preserved, then the racket velocity is added back. Only
        shuttles physically within ``contact_radius`` of the racket sweet-spot are deflected;
        the rest pass through untouched (a real miss).
        """
        if hit_ids.numel() == 0:
            return

        body_ids = self.cfg.asset_cfg.body_ids

        # Only deflect shuttles the racket actually reaches. Use the sweet-spot frame that the
        # approach reward / success metric already key off of.
        racket_pos_w = self._env.scene[self.cfg.ee_frame.name].data.target_pos_w.squeeze(1)[hit_ids]
        contact = torch.norm(racket_pos_w - self.shuttle_pos_w[hit_ids], dim=-1) < self.contact_radius
        hit_ids = hit_ids[contact]
        if hit_ids.numel() == 0:
            return

        # Racket state at contact.
        ee_quat_w = self.robot.data.body_quat_w[hit_ids, body_ids, :].squeeze(1)   # (H,4)
        v_racket = self.robot.data.body_vel_w[hit_ids, body_ids, :3].squeeze(1)    # (H,3)

        # Racket face normal = local -Z rotated to world (matches ee_orientation_tracking_reward).
        local_normal = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(hit_ids.shape[0], 3)
        n = math_utils.quat_rotate(ee_quat_w, local_normal)
        n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-6)

        # Reflect in the racket's frame; orient the normal to the face meeting the shuttle so the
        # outgoing direction tracks the racket orientation regardless of which side makes contact.
        v_rel = self.shuttle_vel_w[hit_ids] - v_racket
        side = -torch.sign((v_rel * n).sum(dim=-1, keepdim=True))
        side = torch.where(side == 0.0, torch.ones_like(side), side)
        n_eff = n * side
        vn = (v_rel * n_eff).sum(dim=-1, keepdim=True) * n_eff
        v_rel_out = v_rel - (1.0 + self.restitution[hit_ids]).unsqueeze(-1) * vn

        self.shuttle_vel_w[hit_ids] = v_rel_out + v_racket

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

    def _drag_rk4_step(self, pos, vel, dt, inv_L):
        """One RK4 step of the quadratic-drag dynamics ``dv/dt = g - ||v|| * v / L``.

        ``pos`` / ``vel`` are (..., 3); ``inv_L`` (= 1 / L) is broadcastable to (..., 1).
        """
        def accel(v):
            speed = torch.norm(v, dim=-1, keepdim=True)
            return self.g_vec - speed * v * inv_L

        k1x, k1v = vel, accel(vel)
        k2x, k2v = vel + 0.5 * dt * k1v, accel(vel + 0.5 * dt * k1v)
        k3x, k3v = vel + 0.5 * dt * k2v, accel(vel + 0.5 * dt * k2v)
        k4x, k4v = vel + dt * k3v, accel(vel + dt * k3v)

        new_pos = pos + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        new_vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return new_pos, new_vel

    def _predict_interception(self, pos0, vel0, L):
        """Forward-integrate sampled shuttles until they descend through the interception height.

        Returns ``(time, pos, vel)`` at the first descending crossing of ``intercept_height`` for
        each shuttle, batched over (N, num_targets, ...). Shuttles that never cross within
        ``max_prediction_time`` fall back to their state at ``max_prediction_time``.
        """
        inv_L = (1.0 / L).view(-1, 1, 1)  # (N,1,1) -> broadcasts over targets & xyz
        z_int = self.intercept_height

        pos, vel = pos0.clone(), vel0.clone()
        found = torch.zeros(pos0.shape[:-1], dtype=torch.bool, device=self.device)  # (N,T)
        t_int = torch.full(pos0.shape[:-1], self.max_prediction_time, device=self.device)
        pos_int, vel_int = pos0.clone(), vel0.clone()

        n_steps = int(round(self.max_prediction_time / self.dt))
        t = 0.0
        for _ in range(n_steps):
            prev_pos, prev_vel = pos, vel
            pos, vel = self._drag_rk4_step(pos, vel, self.dt, inv_L)
            t += self.dt

            # First descending crossing of the interception height.
            crossed = (~found) & (prev_pos[..., 2] > z_int) & (pos[..., 2] <= z_int) & (vel[..., 2] < 0.0)
            if crossed.any():
                # Sub-step linear interpolation of the crossing point for a smoother estimate.
                denom = (prev_pos[..., 2] - pos[..., 2]).clamp(min=1e-6)
                frac = ((prev_pos[..., 2] - z_int) / denom).clamp(0.0, 1.0)  # (N,T)
                pos_c = prev_pos + (pos - prev_pos) * frac.unsqueeze(-1)
                vel_c = prev_vel + (vel - prev_vel) * frac.unsqueeze(-1)
                pos_int = torch.where(crossed.unsqueeze(-1), pos_c, pos_int)
                vel_int = torch.where(crossed.unsqueeze(-1), vel_c, vel_int)
                t_int = torch.where(crossed, (t - self.dt) + frac * self.dt, t_int)
                found = found | crossed
                if found.all():
                    break

        # Shuttles that never crossed: use the state at max_prediction_time.
        not_found = ~found
        if not_found.any():
            pos_int = torch.where(not_found.unsqueeze(-1), pos, pos_int)
            vel_int = torch.where(not_found.unsqueeze(-1), vel, vel_int)

        return t_int, pos_int, vel_int