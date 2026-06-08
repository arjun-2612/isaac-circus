from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_rotate,
    quat_box_minus,
    quat_rotate_inverse,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .wrench_assist_command_cfg import WrenchAssistCommandCfg


class WrenchAssistCommand(CommandTerm):
    cfg: WrenchAssistCommandCfg

    """Computes and applies an assistive wrench to the selected body."""

    def __init__(self, cfg: WrenchAssistCommandCfg, env: ManagerBasedEnv):

        # visualizer needs to be created first for `._set_debug_vis_imp() to be called.`
        self.force_assist_visualizer = VisualizationMarkers(cfg.force_assist_visualizer_cfg)
        # initialize the base class
        super().__init__(cfg, env)

        self.asset: Articulation = env.scene[cfg.asset_name]
        self.wrench_scale = torch.full((self.num_envs,), cfg.max_wrench_scale, device=self.device)
        self.command_level = torch.zeros(self.num_envs, device=self.device)
        self._forces_b = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torques_b = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self._vis_arrow_pitch = torch.full_like(self.wrench_scale, -math.pi / 2)

        # acquire body ID of target body to apply assistive force on
        self.body_ids, _ = self.asset.find_bodies(self.cfg.wrench_assist_body_name)
        if len(self.body_ids) > 1:
            raise ValueError(
                f"Multiple body IDs found for '{self.cfg.wrench_assist_body_name}': {self.body_ids}. Expected only one."
            )
        self._robot_weight = (torch.sum(self.asset.root_physx_view.get_masses()) / self._env.num_envs) * abs(
            self._env.sim.cfg.gravity[2]
        )

        total_mass, total_inertia_b, base_to_com_pos_b = self.compute_whole_body_info()
        self._mean_total_mass = torch.mean(total_mass, dim=0)
        self._mean_total_inertia_b = torch.mean(total_inertia_b, dim=0)
        self._mean_base_to_com_pos_b = torch.mean(base_to_com_pos_b, dim=0)
        self._mean_base_to_com_pos_b = self._mean_base_to_com_pos_b.repeat(self.num_envs, 1)
        gravity_w = torch.tensor(self._env.sim.cfg.gravity, device=self.device).repeat(self.num_envs, 1)
        self._gravitational_force_w = self._mean_total_mass * gravity_w

    @property
    def command(self) -> torch.Tensor:
        pass

    @property
    def forces_b(self) -> torch.Tensor:
        return self._forces_b.reshape(self.num_envs, 3)

    @property
    def torques_b(self) -> torch.Tensor:
        return self._torques_b.reshape(self.num_envs, 3)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.force_assist_visualizer.set_visibility(debug_vis)

    def _resolve_force_assist_to_arrow(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the force assist to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.force_assist_visualizer.cfg.markers["arrow"].scale
        # repeat arrow-scale for number of environment instances
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.wrench_scale.shape[0], 1)
        # scale proportionally to force assist magnitude
        arrow_scale[:, 0] *= self.wrench_scale * self.cfg.force_assist_arrow_scale
        # arrow-direction
        zeros = torch.zeros_like(self._vis_arrow_pitch)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, self._vis_arrow_pitch, zeros)

        return arrow_scale, arrow_quat

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.asset.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += self.cfg.force_assist_arrow_z_offset
        # -- resolve the scales and quaternions
        force_assist_arrow_scale, force_assist_arrow_quat = self._resolve_force_assist_to_arrow()
        # display markers
        self.force_assist_visualizer.visualize(base_pos_w, force_assist_arrow_quat, force_assist_arrow_scale)

    def _update_assistive_wrench(self):
        """
        Compute and apply the force and torque applied to specified robot bodies.
        This function calculates and applies an external wrench to help the policy.
        """
        motion_ref = self._env.command_manager.get_term(self.cfg.cmd_name)

        # get references
        p_ref = motion_ref.root_pos_w
        v_ref = motion_ref.root_lin_vel_w
        v_dot_ref = motion_ref.root_lin_acc_w
        quat_ref = motion_ref.root_quat_w
        w_ref = motion_ref.root_ang_vel_w
        w_dot_ref = motion_ref.root_ang_acc_w

        # get measurements
        p_meas = self.asset.data.root_pos_w - self._env.scene.env_origins
        v_meas = self.asset.data.root_lin_vel_w
        quat_meas = self.asset.data.root_quat_w
        quat_meas_inv = quat_conjugate(quat_meas)
        w_meas = self.asset.data.root_ang_vel_w

        # transform whole body inertia to world frame
        R_base = matrix_from_quat(quat_meas)
        base_inertia_w = R_base @ self._mean_total_inertia_b @ R_base.transpose(1, 2)

        # transform base-to-com position to world frame
        base_to_com_pos_w = quat_rotate(quat_meas, self._mean_base_to_com_pos_b)

        # compute force
        Kp_lin, Kd_lin = self.cfg.force_gain
        target_lin_acc_w = v_dot_ref + Kp_lin * (p_ref - p_meas) + Kd_lin * (v_ref - v_meas)
        force_w = self._mean_total_mass * target_lin_acc_w - self._gravitational_force_w

        # compute torque
        Kp_ang, Kd_ang = self.cfg.torque_gain
        target_ang_acc_w = w_dot_ref + Kd_ang * (w_ref - w_meas) + Kp_ang * quat_box_minus(quat_ref, quat_meas)

        torque_w = torch.bmm(target_ang_acc_w.unsqueeze(1), base_inertia_w).squeeze(1) # T = aI
        torque_w += torch.cross(torch.bmm(w_meas.unsqueeze(1), base_inertia_w).squeeze(1), w_meas, dim=-1) # T += wI x w
        torque_w -= torch.cross(base_to_com_pos_w, self._gravitational_force_w, dim=-1) # T -= T x g

        force_w *= self.wrench_scale.unsqueeze(-1)
        torque_w *= self.wrench_scale.unsqueeze(-1)

        # rotate forces and torques to base frame
        self._forces_b = quat_rotate(quat_meas_inv, force_w).unsqueeze(1)
        self._torques_b = quat_rotate(quat_meas_inv, torque_w).unsqueeze(1)

        # apply assistive wrench in base frame.
        self.asset.set_external_force_and_torque(self._forces_b, self._torques_b, body_ids=self.body_ids)

    def compute_whole_body_info(self):
        """
        Returns (shapes per env):
        total_mass        : (E,)
        total_inertia_b   : (E, 3, 3)   inertia about the base origin, expressed in base frame
        base_to_com_pos_b : (E, 3)      base->system COM vector, expressed in base frame
        """
        device = self.device
        av = self.asset.root_physx_view

        # -----------------------------
        # Fetch/link-shape to (E, L, ...)
        # -----------------------------
        # masses: (E, L)
        link_masses = av.get_masses().to(device)
        if link_masses.dim() == 1:
            # (E*L,) -> (E, L)
            link_masses = link_masses.view(self.num_envs, -1)

        # inertias at COM in link frame: (E, L, 3, 3)
        link_inertias = av.get_inertias().to(device).view(self.num_envs, -1, 3, 3)

        # COM offsets in link frame: (E, L, 3)
        link_com_l = av.get_coms().to(device)[..., :3]

        # link transforms: position (E, L, 3), quaternion (E, L, 4, wxyz)
        link_tf = av.get_link_transforms().to(device)       # (E, L, 7) [pos(0:3), quat(3:7)]
        link_pos_w = link_tf[..., 0:3]
        link_quat_w = link_tf[..., 3:7]

        # base (root) pose
        root_pos_w  = self.asset.data.root_pos_w.to(device)     # (E, 3)
        root_quat_w = self.asset.data.root_quat_w.to(device)    # (E, 4, wxyz)

        # -----------------------------------
        # Small helpers (use your local ones if you have them)
        # -----------------------------------
        def quat_to_mat_wxyz(q):
            """q: (..., 4) -> (..., 3, 3) mapping local->world."""
            w, x, y, z = q.unbind(-1)
            ww, xx, yy, zz = w*w, x*x, y*y, z*z
            wx, wy, wz = w*x, w*y, w*z
            xy, xz, yz = x*y, x*z, y*z
            r00 = ww + xx - yy - zz
            r01 = 2*(xy - wz)
            r02 = 2*(xz + wy)
            r10 = 2*(xy + wz)
            r11 = ww - xx + yy - zz
            r12 = 2*(yz - wx)
            r20 = 2*(xz - wy)
            r21 = 2*(yz + wx)
            r22 = ww - xx - yy + zz
            return torch.stack([
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ], dim=-2)

        # -----------------------------
        # World COM of each link: p_com_w = p_link_w + R_lw * com_l
        # -----------------------------
        R_lw = quat_to_mat_wxyz(link_quat_w)                        # (E, L, 3, 3)
        p_com_w = link_pos_w + torch.matmul(R_lw, link_com_l.unsqueeze(-1)).squeeze(-1)  # (E, L, 3)

        # -----------------------------
        # System COM in world & base->COM in base frame
        # -----------------------------
        m = link_masses                                             # (E, L)
        total_mass = m.sum(dim=1)                                   # (E,)
        com_w = (m.unsqueeze(-1) * p_com_w).sum(dim=1) / total_mass.unsqueeze(-1)  # (E, 3)

        base_to_com_w = com_w - root_pos_w                          # (E, 3)
        base_to_com_b = quat_rotate_inverse(root_quat_w, base_to_com_w)  # (E, 3)

        # -----------------------------
        # Whole-body inertia about base origin, expressed in base frame
        # 1) Rotate each link inertia (about its COM) into world:
        #    I_com_w = R_lw I_com_l R_lw^T
        # 2) Parallel-axis to base origin via d = (p_com_w - root_pos_w):
        #    I_about_base_w = I_com_w + m (||d||^2 I - d d^T)
        # 3) Express in base frame: I_b = R_wb I_w R_bw
        # -----------------------------
        I_com_l = link_inertias                                     # (E, L, 3, 3)
        I_com_w = torch.matmul(R_lw, torch.matmul(I_com_l, R_lw.transpose(-1, -2)))  # (E, L, 3, 3)

        d_w = p_com_w - root_pos_w.unsqueeze(1)                     # (E, L, 3)
        d2 = (d_w * d_w).sum(dim=-1, keepdim=True)                  # (E, L, 1)
        I3 = torch.eye(3, device=device, dtype=link_inertias.dtype).view(1, 1, 3, 3)

        ddT = d_w.unsqueeze(-1) @ d_w.unsqueeze(-2)                 # (E, L, 3, 3)
        par = m.view(self.num_envs, -1, 1, 1) * (d2.unsqueeze(-1) * I3 - ddT)  # (E, L, 3, 3)

        I_w_total = (I_com_w + par).sum(dim=1)                      # (E, 3, 3)

        # World->base rotation: R_wb = R_bw^T
        R_bw = quat_to_mat_wxyz(root_quat_w)                        # (E, 3, 3)
        R_wb = R_bw.transpose(-1, -2)
        total_inertia_b = torch.matmul(R_wb, torch.matmul(I_w_total, R_bw))  # (E, 3, 3)

        return total_mass, total_inertia_b, base_to_com_b

    def _update_command(self):
        """Updates the assistive force."""
        self._update_assistive_wrench()