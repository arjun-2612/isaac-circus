# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import math
import numpy as np
from qpth.qp import QPFunction
# import jax.numpy as jnp

import omni.log
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions import JointAction
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from isaaclab.managers import ObservationManager
from isaaclab.utils.assets import check_file_path, read_file
import isaaclab.utils.string as string_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from circus_extensions.simulation.envs import ManagerBasedRLEnv

    from . import actions_cfg


class QuadcopterAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.QuadcopterActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.QuadcopterActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._leg_actions = torch.zeros(self.num_envs, 12, device=self.device)

        self._root_id = self._asset.find_bodies(self.cfg.root_name, preserve_order=True)[0] # for the root body
        self._leg_joint_ids = self._asset.find_joints(self.cfg.leg_joint_names, preserve_order=True)[0] # for the legs

        self._legs_offset = torch.tensor([
            0.0, 0.0, 0.0, 0.0,  # bl_hx, br_hx, fl_hx, fr_hx
            0.698132, 0.698132, 0.698132, 0.698132,  # bl_hy, br_hy, fl_hy, fr_hy
            0.2, 0.2, 0.2, 0.2,  # bl_kn, br_kn, fl_kn, fr_kn
        ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    def apply_actions(self):
        actions = (self.raw_actions.clone() * self._scale).clamp(-1.0, 1.0) # (num_envs, 4)
        thrust = actions[:, 0]
        moments = actions[:, 1:]

        self._thrust[:, 0, 2] = self.cfg.robot_weight * self.cfg.thrust_to_weight_ratio * (thrust + 1.0) / 2.0 # (num_envs, 1, 1)
        self._moment[:, 0, :] = self.cfg.moment_scale * moments # (num_envs, 1, 3)

        self._asset.set_joint_position_target(self._leg_actions + self._legs_offset, joint_ids=self._leg_joint_ids) # send zero commands for all leg joints
        self._asset.set_external_force_and_torque(self._thrust, self._moment, self._root_id) # set force and momentum for root link


class ReferencePositionAction(JointAction):

    cfg: actions_cfg.ReferencePositionActionCfg

    def __init__(self, cfg: actions_cfg.ReferencePositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.reference_generator = env.command_manager.get_term(cfg.reference_name)
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        self._offset = self.reference_generator.joint_pos[..., self._joint_ids]
        self._processed_actions = self._offset + self._raw_actions * self._scale

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RMPCAction(JointAction):

    cfg: actions_cfg.RMPCActionCfg

    def __init__(self, cfg: actions_cfg.RMPCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env 
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.sensor_name]

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._ee_body_ids, self._ee_body_names = self._asset.find_bodies(cfg.ee_name)
        self.hip_body_ids, _ = self._asset.find_bodies(cfg.hx_bodies, preserve_order=True)
        self.thigh_body_ids, _ = self._asset.find_bodies(cfg.hy_bodies, preserve_order=True)
        self.foot_ids_contact_sensor = self.contact_sensor.find_bodies(cfg.ee_name, preserve_order=True)[0]

        self._jacobian_body_ids = self._ee_body_ids
        self._jacobian_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        print(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        print(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._ee_body_names} [{self._ee_body_ids}]"
        )

        self.has_ankle = cfg.has_ankle
        self.k_joint_ids = None 
        self.a_joint_ids = None

        if self.has_ankle:
            self.a_joint_ids, _ = self._asset.find_joints(cfg.a_joints, preserve_order=True)
            self.k_joint_ids, _ = self._asset.find_joints(cfg.k_joints, preserve_order=True)

            self._jacobian_k_jidx = [i + 6 for i in self.k_joint_ids]
            self._jacobian_a_jidx = [i + 6 for i in self.a_joint_ids]

        self.tau = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        if cfg.body_offset is not None:
            pos = torch.tensor(cfg.body_offset.pos, device=self.device, dtype=torch.float32)
            rot = torch.tensor(cfg.body_offset.rot, device=self.device, dtype=torch.float32)
            # Allow either per-leg or single offset in cfg
            if pos.ndim == 1:  # (3,)
                pos = pos.unsqueeze(0).repeat(len(self._ee_body_ids), 1)        # (L,3)
            if rot.ndim == 1:  # (4,)
                rot = rot.unsqueeze(0).repeat(len(self._ee_body_ids), 1)        # (L,4)
            # Now expand across envs
            self._offset_pos = pos.unsqueeze(0).repeat(self.num_envs, 1, 1).contiguous()  # (E,L,3)
            self._offset_rot = rot.unsqueeze(0).repeat(self.num_envs, 1, 1).contiguous()  # (E,L,4)
        else:
            self._offset_pos, self._offset_rot = None, None

        self.nf = 4
        self.nx = 13
        self.nu = self.nf*3
        self.qp_dtype = torch.float64

        self._dt = env.step_dt
        self._prev_p_ref = torch.zeros((self.num_envs, len(self._ee_body_ids), 3), device=self.device)
        self._prev_v_ref = torch.zeros((self.num_envs, len(self._ee_body_ids), 3), device=self.device)
        self._prev_Jv = self._compute_frame_jacobian()[:, :, 0:3, :]

        self._Ts = cfg.step_time
        self._trot_phase = torch.zeros((self.num_envs,), device=self.device)
        self._duty_cycle = cfg.duty_cycle
        self._desired_leg_state = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self._leg_state = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self._prev_leg_state = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self._prev_contact = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self.new_contact = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)

        self._sign_LR = torch.tensor([1, -1, 1, -1], device=self.device).unsqueeze(0).expand((self.num_envs, -1))
        self._phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0).expand((self.num_envs, -1))

        self.foot_targets_w = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.swing_targets_w = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.foot_pos_w = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.hip_pos_w = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.hip_pos = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.hip_to_thigh_offset_b = torch.zeros((self.num_envs, self.nf, 3), device=self.device)

        self._desired_com_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_com_speed = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_com_rpy = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_com_twist = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_com_pos[:, 2] = cfg.desired_height

        self._mu = cfg.friction_coeff
        self._fmin_ratio = cfg.f_min_ratio
        self._fmax_ratio = cfg.f_max_ratio
        self._mass_robot = cfg.robot_mass

        self.mpc_fw = torch.zeros((self.num_envs, self.nf, 3), device=self.device)
        self.q_desired = torch.zeros((self.num_envs, 12), device=self.device)
        self.q_touchdown = torch.zeros((self.num_envs, 12), device=self.device)
        self.applied_forces_w = torch.zeros((self.num_envs, 4, 3), device=self.device)

        self.G = torch.zeros((self.num_envs, self.nu*self.cfg.horizon_length, self.nu*self.cfg.horizon_length), device=self.device)
        self.p = torch.zeros((self.num_envs, self.nu*self.cfg.horizon_length), device=self.device)
        
        self.xt = torch.zeros((self.num_envs, self.nx), device=self.device)
        self.xd = torch.zeros((self.num_envs, self.cfg.horizon_length, self.nx), device=self.device)
        
        self.A_d = torch.zeros((self.num_envs, self.nx, self.nx), device=self.device)
        self.B_d = torch.zeros((self.num_envs, self.nx, self.nu), device=self.device)
        
        self.Q_diag = torch.tensor(self.cfg.q_diag, device=self.device)
        self.R_diag = torch.tensor(self.cfg.r_diag, device=self.device)
        
        self.A_qp = torch.zeros((self.num_envs, self.nx*self.cfg.horizon_length, self.nx), device=self.device)
        self.B_qp = torch.zeros((self.num_envs, self.nx*self.cfg.horizon_length, self.nu*self.cfg.horizon_length), device=self.device)

        self.dt_mpc = self._env.step_dt
        self.max_pos_error = 0.1
        self.a_g = 9.81

        self._counter = 0
        self._decimation = cfg.decimation

        self.vx_h_smooth = torch.zeros((self.num_envs,), device=self.device)
        self.vy_h_smooth = torch.zeros((self.num_envs,), device=self.device)
        self.wz_h_smooth = torch.zeros((self.num_envs,), device=self.device)
        self.alpha = cfg.smoothing_factor
        self.remaining_swing_time = 0.0

        self.roll_init = 0.0
        self.pitch_init = 0.0

        self._reach_min = torch.tensor(
            [
                [cfg.lxy_caps[0], cfg.lxy_caps[2]],     # BL: back, inner
                [cfg.rxy_caps[0], cfg.rxy_caps[2]],     # BR: back, outer
                [cfg.lxy_caps[0], cfg.lxy_caps[2]],     # FL: back, inner
                [cfg.rxy_caps[0], cfg.rxy_caps[2]],     # FR: back, outer
            ],
            device=self.device,
        )
        self._reach_max = torch.tensor(
            [
                [cfg.lxy_caps[1], cfg.lxy_caps[3]],     # BL: back, inner
                [cfg.rxy_caps[1], cfg.rxy_caps[3]],     # BR: back, outer
                [cfg.lxy_caps[1], cfg.lxy_caps[3]],     # FL: back, inner
                [cfg.rxy_caps[1], cfg.rxy_caps[3]],     # FR: back, outer
            ],
            device=self.device,
        )

        self.vel_cmd = torch.zeros((self.num_envs, 3), device=self.device)

    """
    Properties.
        """

    @property
    def foot_targets(self) -> torch.Tensor:
        return self.foot_targets_w[:, :, :2].flatten(start_dim=1)
    
    @property
    def leg_state(self) -> torch.Tensor:
        return self._leg_state
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.q_touchdown
    
    @property
    def desired_body_state(self) -> torch.Tensor:   
        return torch.cat([self.desired_body_pos, self.desired_body_quat], dim=-1)
    
    @property
    def desired_body_pos(self) -> torch.Tensor:
        return self.xd[:, 0, 3:6]
    
    @property
    def desired_body_quat(self) -> torch.Tensor:
        r = self.xd[:, 0, 0]
        p = self.xd[:, 0, 1]
        y = self.xd[:, 0, 2]
        return math_utils.quat_from_euler_xyz(r, p, y)
    
    @property
    def swing_targets(self) -> torch.Tensor:
        return self.swing_targets_w.flatten(start_dim=1)
    
    @property
    def foot_positions(self) -> torch.Tensor:
        return self._asset.data.body_pos_w[:, self._ee_body_ids].flatten(start_dim=1)
    
    @property
    def foot_forces(self) -> torch.Tensor:
        return self.mpc_fw
    
    @property
    def applied_forces(self) -> torch.Tensor:
        return self.applied_forces_w[..., 2]
    
    @property
    def joint_pos_desired(self) -> torch.Tensor:
        return self.q_desired

    @property
    def jacobian_w(self) -> torch.Tensor:
        J_all = self._asset.root_physx_view.get_jacobians().clone()
        J_bodies = J_all[:, self._jacobian_body_ids, :, :]
        J_joints = J_bodies[:, :, :, self._jacobian_joint_ids]

        # If we have passive ankles, the Jacobian needs to be reduced to apply torque correctly
        # Since knee and ankle are coupled by mimic, we can do J_kn -= J_ank and get the correct reduction
        if self.has_ankle:
            J_joints[:, :, :, self.k_joint_ids] -= J_bodies[:, :, :, self._jacobian_a_jidx]

        return J_joints
    
    """
    Class functions.
    """

    def apply_actions(self):
        # Uncomment all the lines commented out to run the MPC with Raibert. For RL, comment them back
        # state_changed = self._leg_state != self._prev_leg_state
        # if (self._counter % self._decimation == 0) or (torch.any(state_changed)):
            # Acquire all matrices necessary
            # self._compute_X()
            # self._compute_Xd()
            # self._compute_centroidal_dynamics_matrices()
            # self._compute_qp_mats()
            # self._compute_objective_horizon_matrix()
            # G_ineq, h_ineq = self._compute_constraint_horizon_matrices()

            # # Convert to double data type
            # G_ineq = G_ineq.to(self.qp_dtype)
            # h_ineq = h_ineq.to(self.qp_dtype)
            # G = self.G.to(self.qp_dtype)
            # p = self.p.to(self.qp_dtype)

            # Aeq = torch.empty((self.num_envs, 0, self.nu*self.cfg.horizon_length), device=self.device, dtype=self.qp_dtype)
            # beq = torch.empty((self.num_envs, 0), device=self.device, dtype=self.qp_dtype)

            # f_w = QPFunction(check_Q_spd=False, eps=1e-5, maxIter=150)(G, p, G_ineq, h_ineq, Aeq, beq)  # (E, 12H)
            # f_w0 = f_w[:, :self.nu]  # (E,12) -> only need the first step (receding horizon MPC)
            # self.mpc_fw = f_w0.view(-1, self.nf, 3).to(torch.float32)  # (E,4,3)

        self.update_command()
        self._prev_leg_state = self._leg_state
        
        # # Jacobian for torques
        # Jw = self._compute_frame_jacobian()
        # Jw_T = Jw[:, :, 0:3, :].transpose(-1, -2) # [E, 4, J, 3]

        # # Get the desired joint positions from the command directly
        # q_desired = self.joint_pos_desired
        # tau_ff = self._compute_feedforward_torque(Jw[:, :, 0:3, :])

        # # Torques for joint positions from IK
        # q_meas = self._asset.data.joint_pos[:, self._joint_ids]
        # dq_meas = self._asset.data.joint_vel[:, self._joint_ids]
        # tau_swing = self.cfg.kp * (q_desired - q_meas) + self.cfg.kd * (-dq_meas) + tau_ff

        # # Torques from QP foot forces
        # tau_stance = -torch.einsum('elnd,eld->en', Jw_T, self.mpc_fw)
        
        # # Swing mask to decide which torques to take for which feet
        # mask = torch.cat([self.leg_state, self.leg_state, self.leg_state], dim=1)
        # self.tau = torch.where(
        #     mask,
        #     tau_swing,
        #     tau_stance,
        # )

        # # Set target to actuator compute() function
        # self._asset.set_joint_effort_target(self.tau, joint_ids=self._joint_ids)

        # self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        
        # self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self.foot_pos_w[env_ids] = self._asset.data.body_pos_w[env_ids][:, self._ee_body_ids].clone()
        self.hip_pos_w[env_ids] = self._asset.data.body_pos_w[env_ids][:, self.hip_body_ids].clone()
        hip_pos_w = self._asset.data.body_pos_w[env_ids][:, self.hip_body_ids].clone()
        base_pos_w = self._asset.data.root_pos_w[env_ids].clone()
        base_quat_w = self._asset.data.root_quat_w[env_ids].clone()
        yaw_quat_w = math_utils.yaw_quat(base_quat_w)
        self.hip_pos[env_ids] = math_utils.quat_rotate_inverse(
            yaw_quat_w.unsqueeze(1), hip_pos_w - base_pos_w.unsqueeze(1),
        )
        hip_quat_w = self._asset.data.body_quat_w[env_ids][:, self.hip_body_ids].clone()
        thigh_pos_w = self._asset.data.body_pos_w[env_ids][:, self.thigh_body_ids].clone()

        # Set variables we want to use later on
        self._trot_phase[env_ids] = 0.0
        self.hip_to_thigh_offset_b[env_ids] = math_utils.quat_rotate_inverse(
            hip_quat_w, thigh_pos_w - hip_pos_w
        )

        self.swing_targets_w = self.foot_pos_w.clone()
        self.foot_targets_w = self.foot_pos_w.clone()
        self.roll_init = 0.0
        self.pitch_init = 0.0

    def update_command(self):
        ###################### CONTACT SCHEDULE ######################
        s = self._simple_contact_schedule()
        # s = self._smart_contact_schedule()

        ##################### UPDATE VARIABLES #####################
        base_pos_w = self._asset.data.root_pos_w.clone()
        base_quat_w = self._asset.data.root_quat_w.clone()
        yaw_only_quat_w = math_utils.yaw_quat(base_quat_w)
        hip_pos_w = self._asset.data.body_pos_w[:, self.hip_body_ids].clone()
        hip_quat_w = self._asset.data.body_quat_w[:, self.hip_body_ids].clone()

        try:
            desired_velocity = self._env.command_manager.get_command(self.cfg.cmd_vel_name)
        except KeyError:
            desired_velocity = self.vel_cmd.clone()

        desired_velocity_w = math_utils.quat_rotate(
            yaw_only_quat_w, 
            desired_velocity
        )
        observed_lin_vel_w = self._asset.data.root_lin_vel_w.clone()
        observed_ang_vel_w = self._asset.data.root_ang_vel_w.clone()
        observed_lin_vel_h = math_utils.quat_rotate_inverse(yaw_only_quat_w, observed_lin_vel_w)
        observed_ang_vel_h = math_utils.quat_rotate_inverse(base_quat_w, observed_ang_vel_w)

        vx_h = observed_lin_vel_h[:, 0].unsqueeze(1) # [E, 1]
        vy_h = observed_lin_vel_h[:, 1].unsqueeze(1) # [E, 1]
        wz_h = observed_ang_vel_h[:, 2].unsqueeze(1) # [E, 1]
        vx_des = desired_velocity[:, 0].unsqueeze(1) # [E, 1]
        vy_des = desired_velocity[:, 1].unsqueeze(1) # [E, 1]
        wz_des = desired_velocity[:, 2].unsqueeze(1) # [E, 1]
        vx_des_w = desired_velocity_w[:, 0].unsqueeze(1) # [E, 1]
        vy_des_w = desired_velocity_w[:, 1].unsqueeze(1) # [E, 1]
        wz_des_w = desired_velocity_w[:, 2].unsqueeze(1) # [E, 1]

        self.vx_h_smooth = (1-self.alpha) * self.vx_h_smooth + self.alpha * vx_h.squeeze(1)
        self.vy_h_smooth = (1-self.alpha) * self.vy_h_smooth + self.alpha * vy_h.squeeze(1)
        self.wz_h_smooth = (1-self.alpha) * self.wz_h_smooth + self.alpha * wz_h.squeeze(1)

        self._desired_com_speed[:, 0] = vx_des_w.squeeze(1)
        self._desired_com_speed[:, 1] = vy_des_w.squeeze(1)
        self._desired_com_twist[:, 2] = wz_des_w.squeeze(1)

        # Only update foot positions and hip positions on a gait switch event
        # So swing and stance will use latched positions
        new_foot_pos_w = self._asset.data.body_pos_w[:, self._ee_body_ids].clone()
        self.foot_pos_w = torch.where(
            self.new_contact.unsqueeze(-1),
            new_foot_pos_w,
            self.foot_pos_w
        )
        self.hip_pos_w = torch.where(
            self.new_contact.unsqueeze(-1),
            self._asset.data.body_pos_w[:, self.hip_body_ids].clone(),
            self.hip_pos_w
        )
        hip_pos_w_des_height = self.hip_pos_w.clone()
        hip_pos_w_des_height[..., 2] = 0.0
        hip_pos_wrt_b_in_w = hip_pos_w_des_height - base_pos_w.unsqueeze(1)
        hip_pos_h = math_utils.quat_rotate_inverse(yaw_only_quat_w.unsqueeze(1), hip_pos_wrt_b_in_w) # [E, 4, 3]
        # hip_pos_h[:, :2, 0] -= 0.05
        # hip_pos_h[:, 2:, 0] += 0.05
        # hip_pos_h[..., 1] += self._sign_LR * 0.03

        self._prev_contact = self.new_contact.clone()

        ##################### RAIBERT HEURISTIC + ICP #####################
        # [E, 4] + [E, 1] + [E, 1] + [E, 1] = [E, 4]
        to_swing = self._leg_state & ~self._prev_leg_state
        self.remaining_swing_time = torch.where(
            to_swing,
            (1.0 - self._duty_cycle) * self._Ts,
            self.remaining_swing_time - self._env.physics_dt,
        )
        self.remaining_swing_time = torch.clip(self.remaining_swing_time, 0.0)  # [E,4]

        raibert_x = (0.5 * self._duty_cycle * self._Ts * self.vx_h_smooth.unsqueeze(-1)) \
            - (self.cfg.kp_x * (vx_des - self.vx_h_smooth.unsqueeze(-1)))
        raibert_y = (0.5 * self._duty_cycle * self._Ts * self.vy_h_smooth.unsqueeze(-1)) \
            - (self.cfg.kp_y * (vy_des - self.vy_h_smooth.unsqueeze(-1)))

        swing_x_targets_h = self.hip_pos[..., 0] + raibert_x
        swing_y_targets_h = self.hip_pos[..., 1] + raibert_y
        swing_xy_targets_h = torch.stack([swing_x_targets_h, swing_y_targets_h], dim=-1)  # [E,4,2]

        # Yaw velocity corrections
        r_ib = self.hip_pos[..., :2]                 # [E,4,2]
        perp_r = torch.stack([-r_ib[..., 1], r_ib[..., 0]], dim=-1)  # [E,4,2]
        delta_yaw_ff = self.cfg.kp_w * self.remaining_swing_time.unsqueeze(-1) * wz_des.unsqueeze(-1) * perp_r  # [E,4,2]
        delta_yaw_fb = self.cfg.kp_w * (self.wz_h_smooth.unsqueeze(-1) - wz_des).unsqueeze(-1) * perp_r  # [E,4,2]
        swing_xy_targets_h = swing_xy_targets_h + delta_yaw_ff + delta_yaw_fb

        z = torch.zeros_like(swing_x_targets_h, device=self.device)
        swing_xyz_targets_h = torch.cat([swing_xy_targets_h, z.unsqueeze(-1)], dim=-1)    # [E,4,3]

        new_foot_targets_w = math_utils.quat_rotate(yaw_only_quat_w.unsqueeze(1), swing_xyz_targets_h) + base_pos_w.unsqueeze(1)
        new_foot_targets_w[:, :, 2] = 0.0
        tgt_xy = new_foot_targets_w[..., :2]

        ##################### KINEMATIC REACHABILITY #####################
        # Base pose for frame transforms
        base_pos_xy = self._asset.data.root_com_pos_w[:, :2]                       # [E,2]

        # Rotation from WORLD to BODY in XY (inverse yaw rotation)
        R_bw = torch.zeros(self.num_envs, 2, 2, device=self.device)
        yaw = self.quat_to_euler_xyz_world(base_quat_w)[:, 2]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_bw[:, 0, 0] = cy
        R_bw[:, 0, 1] = sy
        R_bw[:, 1, 0] = -sy
        R_bw[:, 1, 1] = cy

        # Hip positions in world XY
        hip_xy_w = self._asset.data.body_pos_w[:, self._ee_body_ids, :2]          # [E,4,2]

        # Relative to base in world frame
        hip_rel_w = hip_xy_w - base_pos_xy.unsqueeze(1)                           # [E,4,2]
        tgt_rel_w = tgt_xy   - base_pos_xy.unsqueeze(1)                           # [E,4,2]

        # Transform to BODY frame: [E,4,2] = R_bw[e] @ rel_w[e,leg]
        hip_rel_b = torch.einsum("eij,ekj->eki", R_bw, hip_rel_w)                 # [E,4,2]
        tgt_rel_b = torch.einsum("eij,ekj->eki", R_bw, tgt_rel_w)                 # [E,4,2]

        # Target relative to hip in BODY frame
        rel_to_hip_b = tgt_rel_b - hip_rel_b                                      # [E,4,2]

        # Broadcast reach limits: [E,4,2]
        reach_min = self._reach_min.unsqueeze(0)                                  # [1,4,2]
        reach_max = self._reach_max.unsqueeze(0)                                  # [1,4,2]

        # Clamp to reachable box per leg
        rel_clamped_b = torch.max(torch.min(rel_to_hip_b, reach_max), reach_min)  # [E,4,2]

        # Back to body-frame target positions relative to base
        tgt_clamped_b = hip_rel_b + rel_clamped_b                                 # [E,4,2]

        # Transform back to WORLD frame
        R_wb = R_bw.transpose(1, 2)                                               # [E,2,2]
        tgt_clamped_rel_w = torch.einsum("eij,ekj->eki", R_wb, tgt_clamped_b)     # [E,4,2]
        tgt_xy = tgt_clamped_rel_w + base_pos_xy.unsqueeze(1)                     # [E,4,2]
        new_foot_targets_w[..., :2] = tgt_xy

        ##################### APPLY FOOT TARGETS #####################
        self.foot_targets_w = torch.where(
            self._leg_state.unsqueeze(-1),
            new_foot_targets_w, 
            self.foot_targets_w
        )

        ##################### BEZIER TRAJECTORY #####################
        foot_targets_h = self._get_bezier_trajectory_at_phase(                # [E, 4, 3]
            s.unsqueeze(-1), hip_pos_h, swing_xy_targets_h
        )
        new_swing_targets_w = math_utils.quat_rotate(yaw_only_quat_w.unsqueeze(1), foot_targets_h) + base_pos_w.unsqueeze(1)
        self.swing_targets_w = torch.where(
            self._leg_state.unsqueeze(-1),
            new_swing_targets_w, 
            self.swing_targets_w
        )
        ###################### IK ######################
        q_abd, q_sag, q_kn = self._inverse_kinematics( # [E, 4]
            hip_pos_w=hip_pos_w,
            hip_quat_w=hip_quat_w,
            L1=self.cfg.hy_length,
            L2=self.cfg.kn_length,
            knee_sign=self.cfg.knee_sign,
            L3=self.cfg.ank_length,
            target=self.swing_targets_w,
        )
        q_abd = q_abd.squeeze(-1).reshape(self.num_envs, 4)
        q_sag = q_sag.squeeze(-1).reshape(self.num_envs, 4)
        q_kn  = q_kn.squeeze(-1).reshape(self.num_envs, 4)
        self.q_desired = torch.cat([q_abd, q_sag, q_kn], dim=-1)

        q_abd, q_sag, q_kn = self._inverse_kinematics( # [E, 4]
            hip_pos_w=hip_pos_w,
            hip_quat_w=hip_quat_w,
            L1=self.cfg.hy_length,
            L2=self.cfg.kn_length,
            knee_sign=self.cfg.knee_sign,
            L3=self.cfg.ank_length,
            target=self.foot_targets_w,
        )
        q_abd = q_abd.squeeze(-1).reshape(self.num_envs, 4)
        q_sag = q_sag.squeeze(-1).reshape(self.num_envs, 4)
        q_kn  = q_kn.squeeze(-1).reshape(self.num_envs, 4)
        self.q_touchdown = torch.cat([q_abd, q_sag, q_kn], dim=-1)

        # # Debug
        # print(f"X vel: {self.vx_h_smooth}")
        # print(f"Hip positions: {self.hip_pos[..., 0]}")
        # print(f"Raibert addition: {raibert_x}")
        # print(f"ICP x: {icp_x}")
        # print(f"Bias: {bx}")
        # print(f"Final position: {swing_x_targets_h}")
        # print("------------")

    """
    MPC Functions.
    """

    def _compute_X(self):
        # -- this is your 'states' block --
        robot_com_pos     = self._asset.data.root_com_pos_w.clone()
        robot_com_lin_vel = self._asset.data.root_com_lin_vel_w.clone()
        robot_com_ang_vel = self._asset.data.root_com_ang_vel_w.clone()
        q_w = self._asset.data.root_com_quat_w.clone()

        v_smooth = self.smooth_velocities(robot_com_lin_vel, self.xt[:, 9:12], alpha=self.alpha)
        w_smooth = self.smooth_velocities(robot_com_ang_vel, self.xt[:, 6:9], alpha=self.alpha)

        rpy = self.quat_to_euler_xyz_world(q_w)
        rpy = math_utils.wrap_to_pi(rpy)
        g_const = torch.full((self.num_envs, 1), -self.a_g, device=self.device, dtype=rpy.dtype)

        self.xt = torch.hstack([rpy, robot_com_pos, w_smooth, v_smooth, g_const])

    def _compute_Xd(self):
        # self._desired_com_pos[:, :2] = torch.where(
        #     (self._desired_com_pos[:, :2] - self.xt[:, :2]) > self.max_pos_error,
        #     self.xt[:, :2] + self.max_pos_error,
        #     self._desired_com_pos[:, :2],
        # )
        # self._desired_com_pos[:, :2] = torch.where(
        #     (self.xt[:, :2] - self._desired_com_pos[:, :2]) < self.max_pos_error,
        #     self.xt[:, :2] - self.max_pos_error,
        #     self._desired_com_pos[:, :2],
        # )

        self.pitch_init = torch.where(
            self.xt[:, 9] > 0.2,
            self.pitch_init + self.dt_mpc * (0.0 - self.xt[:, 1]) / self.xt[:, 9],
            self.pitch_init,
        )
        self.roll_init = torch.where(
            self.xt[:, 10] > 0.2,
            self.roll_init + self.dt_mpc * (0.0 - self.xt[:, 0]) / self.xt[:, 10],
            self.roll_init,
        )
        self.pitch_init = torch.clip(self.pitch_init, -0.25, 0.25)
        self.roll_init = torch.clip(self.roll_init, -0.25, 0.25)
        roll_comp = self.xt[:, 10] * self.roll_init
        pitch_comp = self.xt[:, 9] * self.pitch_init

        desired_states = torch.hstack([
            self._desired_com_rpy,   # (E,3)
            self._desired_com_pos,   # (E,3)
            self._desired_com_twist, # (E,3)  (ω*)
            self._desired_com_speed  # (E,3)  (v*)
        ])  # (E,12)

        for k in range(self.cfg.horizon_length):
            t = (k+1) * self.dt_mpc
            self.xd[:, k, :] = torch.stack([
                roll_comp,                            # roll*
                pitch_comp,                           # pitch*
                self.xt[:, 2] + desired_states[:, 8] * t,       # yaw: integrate desired yaw rate
                self.xt[:, 3] + desired_states[:, 9] * t,       # x: integrate desired vx
                self.xt[:, 4] + desired_states[:,10] * t,       # y: integrate desired vy
                desired_states[:, 5],                           # z*: (often fixed)
                desired_states[:, 6],                           # ωx*
                desired_states[:, 7],                           # ωy*
                desired_states[:, 8],                           # ωz* (yaw rate)
                desired_states[:, 9],                           # vx*
                desired_states[:,10],                           # vy*
                desired_states[:,11],                           # vz*
                torch.full_like(desired_states[:, 0], -self.a_g),   # -g  (constant)
            ], dim=-1)

    def _compute_centroidal_dynamics_matrices(self):
        device = self.device
        E = self.num_envs
        dt = self.dt_mpc

        # rpy rate transform T(roll,pitch,yaw)
        q_w  = self._asset.data.root_com_quat_w.clone()
        rpy  = self.quat_to_euler_xyz_world(q_w)
        cy, sy = torch.cos(rpy[:, 2]), torch.sin(rpy[:, 2])

        Rz = torch.zeros((E, 3, 3), device=self.device)
        Rz[:, 0, 0] = cy 
        Rz[:, 0, 1] = -sy 
        Rz[:, 1, 0] = sy 
        Rz[:, 1, 1] = cy 
        Rz[:, 2, 2] = 1.0

        eye3 = torch.eye(3, device=device).unsqueeze(0).expand(E,-1,-1)

        # Continuous-time A_c (13x13) and B_c (13x12)
        A_c = torch.zeros((E, self.nx, self.nx), device=device)
        B_c = torch.zeros((E, self.nx, self.nu), device=device)

        # rpy_dot = T * omega
        A_c[:, 0:3, 6:9]  = Rz.transpose(-1, -2)
        # p_dot = v
        A_c[:, 3:6, 9:12] = eye3
        # v_dot gets gravity from the -g state (index 12)
        A_c[:, 11, 12] = 1.0  # \dot{v_z} += (-g_state)

        # Build torque/force mapping
        invIw = self._compute_composite_inertia_world()  # (E,3,3)
        invM3 = (1.0 / self._mass_robot) * torch.eye(3, device=device)
        com_w = self._asset.data.root_com_pos_w.clone()
        r_w   = (self._contact_points_w() - com_w.unsqueeze(1))  # (E,4,3)

        z = torch.zeros_like(r_w[..., 0])
        rx = torch.stack([
            torch.stack([ z,            -r_w[..., 2],  r_w[..., 1]], dim=-1),
            torch.stack([ r_w[..., 2],  z,            -r_w[..., 0]], dim=-1),
            torch.stack([-r_w[..., 1],  r_w[..., 0],  z            ], dim=-1),
        ], dim=-2)  # (E,4,3,3)

        for leg in range(4):
            cols = slice(3*leg, 3*(leg+1))
            # omega_dot = I^{-1} (r x f)
            B_c[:, 6:9,  cols] = torch.bmm(invIw, rx[:, leg])
            # v_dot = (1/m) f
            B_c[:, 9:12, cols] = invM3

        # Build block matrix for ZOH exp: [[A,B],[0,0]] * dt
        nx, nu = self.nx, self.nu
        Z = torch.zeros((E, nx+nu, nx+nu), device=device)
        Z[:, :nx, :nx] = A_c * dt
        Z[:, :nx, nx:] = B_c * dt
        Zexp = torch.linalg.matrix_exp(Z)  # (E, nx+nu, nx+nu)

        self.A_d = Zexp[:, :nx, :nx]       # A_exp
        self.B_d = Zexp[:, :nx, nx:]       # B_exp

    def _compute_qp_mats(self):
        """
        Inputs:
            A_d: (E,13,13)
            B_d: (E,13,12)
        Returns:
            A_qp: (E, 13H, 13)     # [A; A^2; ...; A^H]
            B_qp: (E, 13H, 12H)    # block-lower-triangular with A^(i-j) B
        """
        E = self.num_envs
        H = self.cfg.horizon_length

        A_pows = [torch.eye(self.nx, device=self.device).expand(E, -1, -1)]
        for i in range(H):
            A_pows.append(torch.bmm(A_pows[i], self.A_d))

        for i in range(H):
            self.A_qp[:, self.nx*i:self.nx*(i+1), 0:self.nx] = A_pows[i+1]

            for j in range(H):
                if i >= j:
                    self.B_qp[:, self.nx*i:self.nx*(i+1), self.nu*j:self.nu*(j+1)] = torch.bmm(A_pows[i-j], self.B_d)

    def _compute_objective_horizon_matrix(self):
        """
        Minimizes u in 0.5 u^T G u + u^T p s.t constraints
        Implements G = 2(B^T Q B + R),  p = 2B^T Q(Ax0 - Xd).
        """
        H = self.cfg.horizon_length
        E = self.num_envs

        Xd_vec = self.xd.reshape(E, H*self.nx)                                  # (E,13H)
        Q = torch.diag_embed(self.Q_diag.to(device=self.device).repeat(E, H))   # (E, 13H, 13H)
        R = torch.diag_embed(self.R_diag.to(device=self.device).repeat(H))      # (1,12H,12H) broadcast

        AX0 = torch.bmm(self.A_qp, self.xt.unsqueeze(-1)).squeeze(-1)           # (E,13H)
        err = AX0 - Xd_vec                                                      # (E,13H)

        # Quadratic term: H = 2(B^T Q B + R)
        self.G = 2.0*(torch.bmm(self.B_qp.transpose(1, 2), torch.bmm(Q, self.B_qp)) + R)                # (E,13H,12H)
        I = torch.eye(self.G.size(-1), device=self.device).unsqueeze(0)
        self.G = 0.5*(self.G + self.G.transpose(1,2)) + 1e-8 * I
        # Linear term: g = 2B^T Q(Ax0 - Xd)
        self.p = 2.0*torch.bmm(self.B_qp.transpose(1, 2), torch.bmm(Q, err.unsqueeze(-1))).squeeze(-1)  # (E,13H)

    def _compute_constraint_horizon_matrices(self):
        """
        Inputs:
            contacts: (E,4) boolean (stance=1)
        Returns:
            G_ineq: (E, 24H, 12H)
            h_ineq: (E, 24H)
        """
        device = self.device
        E = self.num_envs
        H = self.cfg.horizon_length

        G_leg = torch.tensor([[ 1., 0., -self._mu],
                            [-1., 0., -self._mu],
                            [ 0., 1., -self._mu],
                            [ 0., -1., -self._mu],
                            [ 0., 0., -1.],
                            [ 0., 0., 1.]], device=self.device)
        G_stage = torch.kron(torch.eye(4, device=device), G_leg) \
                    .unsqueeze(0).expand(E, -1, -1).contiguous()    # (E,24,12)

        # Block-diag across horizon -> (E, 24H, 12H)
        _, M, N = G_stage.shape
        I_H = torch.eye(H, device=device).view(1, H, H, 1, 1)
        Gexp = G_stage.unsqueeze(1).unsqueeze(2)                    # (E,1,1,M,N)
        G_ineq = (I_H * Gexp).permute(0,1,3,2,4).reshape(E, H*M, H*N)

        stance = ~self.leg_state                                   # (E,4)
        n_stance = torch.clamp(stance.sum(dim=1, keepdim=True), min=1.0)
        fmin = (self._fmin_ratio * self._mass_robot * 9.81) / n_stance   # (E,1)
        fmax = (self._fmax_ratio * self._mass_robot * 9.81)

        h_stage = torch.zeros((E, 24), device=device)
        rows_fmin = torch.arange(0, 24, 6, device=device) + 4
        rows_fmax = torch.arange(0, 24, 6, device=device) + 5
        h_stage[:, rows_fmin] = -fmin * stance + 1e-7      # -fz <= -fmin (stance only)
        h_stage[:, rows_fmax] = fmax * stance + 1e-7      # +fz <=  fmax
        h_ineq = h_stage.repeat(1, H)         # (E,24H)

        return G_ineq.contiguous(), h_ineq.contiguous()
    
    """
    Swing leg functions.
    """

    def _simple_contact_schedule(self):
        self._trot_phase = (self._trot_phase + self._env.physics_dt/self._Ts) % 1.0
        leg_phase = (self._trot_phase.unsqueeze(1) + self._phase_offsets) % 1.0
        s = torch.where(
            leg_phase < self._duty_cycle,
            torch.clamp(leg_phase / self._duty_cycle, 0.0, 1.0),
            torch.clamp((leg_phase - self._duty_cycle) / (1.0 - self._duty_cycle), 0.0, 1.0),
        )

        self._desired_leg_state = leg_phase >= self._duty_cycle  # True = swing
        self._leg_state = self._desired_leg_state.clone()

        # contact purely from sensor
        self.new_contact = self._leg_state & (~self._prev_leg_state)

        return s

    def _smart_contact_schedule(self):
        self.applied_forces_w = self.contact_sensor.data.net_forces_w[:, self.foot_ids_contact_sensor].clone()
        leg_phase = self._trot_phase.unsqueeze(1) + self._phase_offsets
        s = torch.where(
            leg_phase < self._duty_cycle,
            torch.clamp(leg_phase / self._duty_cycle, 0.0, 1.0),
            torch.clamp((leg_phase - self._duty_cycle) / (1.0 - self._duty_cycle), 0.0, 1.0),
        )

        # Wait until all swing feet have made contact with the ground before ticking the time
        # Ensures 2 legs on and off at all times 
        in_contact = self.applied_forces_w[..., 2] > 1.0
        hold_phase = torch.where(
            s == 1.0,
            torch.where(
                in_contact,
                False,
                True,
            ),
            False,
        )

        self._trot_phase = torch.where(
            torch.any(hold_phase, dim=1),
            self._trot_phase,
            self._trot_phase % 1.0,
        )
        leg_phase = torch.where(
            hold_phase,
            leg_phase,
            leg_phase % 1.0,
        )

        self._trot_phase = torch.where(
            torch.any(hold_phase, dim=1),
            self._trot_phase,
            self._trot_phase + self._env.physics_dt/self._Ts,
        )

        self._desired_leg_state = leg_phase >= self._duty_cycle  # True = swing
        # Detecting early contact
        self._leg_state = torch.where(
            s > 0.1,
            torch.where(
                torch.logical_and(
                    self._desired_leg_state, 
                    in_contact,
                ),
                torch.logical_not(in_contact),
                self._desired_leg_state.clone(),
            ),
            self._desired_leg_state.clone(),
        )
        # Detecting early swing (loss of contact)
        self._leg_state = torch.where(
            s > 0.1,
            torch.where(
                torch.logical_and(
                    (~self._desired_leg_state), 
                    (~in_contact),
                ),
                in_contact,
                self._desired_leg_state.clone(),
            ),
            self._desired_leg_state.clone(),
        )
        self.new_contact = ~self._leg_state & self._prev_leg_state

        return s

    def _inverse_kinematics(
        self,
        hip_pos_w: torch.Tensor,
        hip_quat_w: torch.Tensor,
        target: torch.Tensor,
        L1: float,
        L2: float,
        knee_sign: float = -1.0,
        eps: float = 1e-9,
        L3: float = 0.0,
    ):
        ft_h = math_utils.quat_rotate_inverse(hip_quat_w, target - hip_pos_w)  # (E,4,3)
        l_hip = self.hip_to_thigh_offset_b[..., 1]

        x = ft_h[..., 0] - self.hip_to_thigh_offset_b[..., 0]
        y = ft_h[..., 1]
        z = ft_h[..., 2] - self.hip_to_thigh_offset_b[..., 2]

        if L3 != 0.0:
            L1 += L3

        c_kn = (x**2 + y**2 + z**2 - l_hip**2 - L2**2 - L1**2) / (2 * L2 * L1)
        c_kn = torch.clamp(c_kn, -1.0 + eps, 1.0 - eps)
        q_kn = knee_sign * torch.acos(c_kn)

        length = torch.sqrt(L1**2 + L2**2 + 2*L1*L2*torch.cos(q_kn))
        q_hip = torch.asin(torch.clamp(-x/(length + eps), -1.0 + eps, 1.0 - eps)) - q_kn/2
        
        s1 = length * torch.cos(q_hip + q_kn/2) * y + l_hip * z
        c1 = l_hip * y - length * torch.cos(q_hip + q_kn/2) * z
        q_abd = torch.atan2(s1, c1)

        # Updated angles based on husky-b structure
        if L3 != 0.0:
            return q_abd, -1.0 * q_hip, (torch.pi/2.0) - q_kn
        
        # For every other robot that has the Spot leg structure (<)
        return q_abd, q_hip, q_kn
    
    def _get_bezier_trajectory_at_phase(self, s, start_pos, goal_pos):
        u = 1.0 - s

        x = u[...,0]*start_pos[...,0] + s[...,0]*goal_pos[...,0]
        y = u[...,0]*start_pos[...,1] + s[...,0]*goal_pos[...,1]

        z0 = start_pos[...,2]
        if goal_pos.shape[-1] == 3:
            z1  = goal_pos[...,2]
        else:
            z1  = z0
        mid = torch.maximum(z0, z1) + self.cfg.foot_clearance

        B0 = u**4
        B1 = 4*u**3*s
        B2 = 6*u**2*s**2
        B3 = 4*u*s**3
        B4 = s**4
        z = (B0 + B1)[...,0]*z0 + B2[...,0]*mid + (B3 + B4)[...,0]*z1  # (E,4)

        return torch.stack([x, y, z], dim=-1)  # (E,4,3)

    """
    Helper functions.
    """

    def _set_velocity_cmd(self, v: torch.Tensor):
        self.vel_cmd = v.clone()

    def _compute_composite_inertia_world(self):
        """
        Compute the centroidal (composite) inertia about the whole-body COM in WORLD.
        Returns:
            Iw:    (E, 3, 3)
            invIw: (E, 3, 3)
        """
        E = self.num_envs

        # --- Per-body masses & local inertias (COM-local, body frame) from PhysX view ---
        av = self._asset.root_physx_view
        masses = av.get_masses().to(device=self.device)                 # (E, L) or (L,) → Isaac Lab returns (E,L)
        inertias_flat = av.get_inertias().to(device=self.device)        # (E, L, 9) flattened 3x3 in COM-local frame
        L = inertias_flat.shape[1]

        # reshape to (E,L,3,3)
        I_body_local = inertias_flat.view(E, L, 3, 3)

        # --- World poses of each link's COM (from ArticulationData) ---
        p_com_w      = self._asset.data.root_com_pos_w                      # (E, 3)
        p_link_com_w = self._asset.data.body_com_pos_w                      # (E, L, 3)
        q_link_com_w = self._asset.data.body_com_quat_w                     # (E, L, 4)

        # Rotations to world: (E,L,3,3)
        R_wi = math_utils.matrix_from_quat(q_link_com_w)

        # Rotate inertias to world: I_wi = R * I_local * R^T
        I_w_links = R_wi @ I_body_local @ R_wi.transpose(-1, -2)

        # Offsets from whole-body COM to link COM (world)
        r = p_link_com_w - p_com_w.unsqueeze(1)  # (E,L,3)

        # Skew(r) for parallel-axis: S(r) so that S @ S^T = [r]_x [r]_x^T
        z = torch.zeros_like(r[..., 0])
        S = torch.stack([
            torch.stack([ z,          -r[..., 2],  r[..., 1]], dim=-1),
            torch.stack([ r[..., 2],   z,         -r[..., 0]], dim=-1),
            torch.stack([-r[..., 1],   r[..., 0],  z        ], dim=-1),
        ], dim=-2)  # (E,L,3,3)

        # Parallel-axis term: m_i * S @ S^T
        m_e = masses.unsqueeze(-1).unsqueeze(-1)  # (E,L,1,1)
        PA  = torch.matmul(S, S.transpose(-1, -2)) * m_e  # (E,L,3,3)

        # Sum contributions over links
        Iw = (I_w_links + PA).sum(dim=1)  # (E,3,3)

        # Inverse (robust)
        invIw = torch.linalg.inv(Iw)

        return invIw

    def _contact_points_w(self) -> torch.Tensor:
        """
        Returns (E, L, 3) world positions of the *contact point* on each EE,
        applying the same per-leg offset pos/rot as used for the Jacobian shift.
        If no offset is configured, returns the EE body origins.
        """
        pos_w  = self.foot_pos_w      # (E,L,3)
        quat_w = self._asset.data.body_quat_w[:, self._ee_body_ids]     # (E,L,4)

        if self._offset_pos is None:
            return pos_w

        R_wL  = math_utils.matrix_from_quat(quat_w)                      # (E,L,3,3)
        R_off = math_utils.matrix_from_quat(self._offset_rot)            # (E,L,3,3)
        R_wT  = torch.einsum("elij,eljk->elik", R_wL, R_off)             # (E,L,3,3)
        r_L   = self._offset_pos                                         # (E,L,3)
        r_w   = torch.einsum("elij,elj->eli", R_wT, r_L)                 # (E,L,3)

        return pos_w + r_w                                               # (E,L,3)
    
    def _compute_frame_jacobian(self) -> torch.Tensor:
        J = self.jacobian_w.clone()              # (E,L,6,J)

        if self._offset_pos is not None:
            Jv = J[:, :, 0:3, :]                 # (E,L,3,J)
            Jw = J[:, :, 3:6, :]                 # (E,L,3,J)

            R_wL = math_utils.matrix_from_quat(
                self._asset.data.body_quat_w[:, self._jacobian_body_ids]
            )                                            # (E, L, 3, 3)

            r_L  = self._offset_pos                      # (E, L, 3)  (in link frame)
            R_off = math_utils.matrix_from_quat(self._offset_rot)           # (E, L, 3, 3)
            R_wT  = torch.einsum('elij,eljk->elik', R_wL, R_off)            # (E, L, 3, 3)
            r_w   = torch.einsum('elij,elj->eli', R_wT, r_L)   # (E, L, 3)

            # Point shift in WORLD axes: Jv' = Jv + (Jw × r_w)
            Jwxrw = torch.cross(Jw, r_w.unsqueeze(-1).expand_as(Jw), dim=-2)
            Jv = Jv + Jwxrw

            # Axes stay world-aligned: Jw unchanged
            J = torch.cat([Jv, Jw], dim=2)

        return J
    
    def _compute_feedforward_torque(self, Jv) -> torch.Tensor:
        p_ref = self.foot_targets_w.reshape(-1, 4, 3)  # (E,4,3)
        v_ref = (p_ref - self._prev_p_ref) / max(self._dt, 1e-6)
        a_ref = (v_ref - self._prev_v_ref) / max(self._dt, 1e-6)
        self._prev_v_ref = v_ref.clone()
        self._prev_p_ref = p_ref.clone()

        p_w, v_w = self._ee_state_w()         # (E,4,3) each
        e_p = p_ref - p_w
        e_v = v_ref - v_w
        a_cmd = a_ref + self.cfg.kp*e_p + self.cfg.kd*e_v  # (E,4,3)

        Jdot = (Jv - self._prev_Jv) / self._dt    # (E,4,3,J)
        self._prev_Jv = Jv.detach().clone()
        dq = self._asset.data.joint_vel[:, self._joint_ids]          # (E,J)
        Jdot_qdot = torch.einsum('elij,ej->eli', Jdot, dq)           # (E,4,3)

        # apply only on swing legs
        a_cmd_eff = a_cmd - Jdot_qdot   # (E,4,3)

        M = self._asset.root_physx_view.get_generalized_mass_matrices().clone() # (E,NJ,NJ)
        jid = torch.as_tensor(self._joint_ids, device=self.device)  # (J,)
        Msel = M.index_select(-1, jid).index_select(-2, jid)
        I = 1e-6 * torch.eye(Msel.shape[-1], device=self.device).unsqueeze(0)

        Minv = torch.linalg.pinv(Msel + I)                                 # (E,NJ,NJ)
        tmp = torch.einsum('elmj,ejk->elmk', Jv, Minv)
        JMJT = torch.einsum('elmk,elkn->elmn', tmp, Jv.transpose(-1, -2))  # (E,4,3,3)
        I3 = 1e-6 * torch.eye(3, device=self.device).view(1,1,3,3)

        Lambda = torch.linalg.inv(JMJT + I3)                        # (E,4,3,3)
        f_task = torch.einsum("elij,eli->elj", Lambda, a_cmd_eff)   # (E,4,3)
        lambda_efforts = torch.einsum("elij,elj->ei", Jv.transpose(-1, -2), f_task)

        return lambda_efforts

    def _ee_state_w(self):
        pos_w  = self._asset.data.body_pos_w[:, self._ee_body_ids]          # (E,L,3)
        lin_w  = self._asset.data.body_lin_vel_w[:, self._ee_body_ids]      # (E,L,3)
        ang_w  = self._asset.data.body_ang_vel_w[:, self._ee_body_ids]      # (E,L,3)

        if self._offset_pos is None:
            return pos_w, lin_w
        
        R_wL = math_utils.matrix_from_quat(self._asset.data.body_quat_w[:, self._ee_body_ids])  # (E,L,3,3)
        R_off = math_utils.matrix_from_quat(self._offset_rot)                                   # (E,L,3,3)
        R_wT = torch.einsum("elij,eljk->elik", R_wL, R_off)                                     # (E,L,3,3)
        r_w = torch.einsum("elij,elj->eli", R_wT, self._offset_pos)                             # (E,L,3)
        
        p_w = pos_w + r_w
        v_w = lin_w + torch.cross(ang_w, r_w, dim=-1)
        
        return p_w, v_w

    def smooth_velocities(self, vnow: torch.Tensor, vprev: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
        """
        Simple exponential moving average filter for smoothing velocity commands.
        velocities: (E,3)
        alpha: smoothing factor [0,1], higher is smoother
        """
        v_smooth = alpha * vnow + (1 - alpha) * vprev
        return v_smooth

    def quat_to_euler_xyz_world(self, q: torch.Tensor, degrees: bool = False) -> torch.Tensor:
        """
        Euler (roll_X, pitch_Y, yaw_Z) about WORLD axes (extrinsic XYZ).
        Equivalent to body-intrinsic ZYX (yaw then pitch then roll).
        q shape (...,4) in (w,x,y,z).
        """
        if q.shape[-1] != 4:
            raise ValueError("Expected q shape (..., 4).")
        w, x, y, z = q.unbind(dim=-1)

        # Rotation matrix (world R, rows first, cols second)
        R11 = 1 - 2*(y*y + z*z)
        R21 = 2*(x*y + w*z)
        R31 = 2*(x*z - w*y)
        R32 = 2*(y*z + w*x)
        R33 = 1 - 2*(x*x + y*y)

        # ZYX convention (roll about x, pitch about y, yaw about z)
        roll  = torch.atan2(R32, R33)
        pitch = torch.atan2(-R31, torch.hypot(R32, R33))  # stable vs asin
        yaw   = torch.atan2(R21, R11)

        eul = torch.stack((roll, pitch, yaw), dim=-1)
        if degrees:
            eul = eul * (180.0 / math.pi)
        return eul
    

class LIPMAction(JointAction):
    cfg: actions_cfg.LIPMActionCfg

    def __init__(self, cfg: actions_cfg.LIPMActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.nf = cfg.num_feet
        self._env = env
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._ee_body_ids, self._ee_body_names = self._asset.find_bodies(cfg.ee_name)

        print(f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]")
        print(f"Resolved body names for {self.__class__.__name__}: {self._ee_body_names} [{self._ee_body_ids}]")

        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._Ts = cfg.step_time 
        self._trot_phase = torch.zeros((self.num_envs,), device=self.device)
        self._phi_contact = torch.zeros((self.num_envs, self.nf), device=self.device)

        self._stance_bool = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self._stance_bool_prev = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)
        self._swing_start = torch.zeros((self.num_envs, self.nf), dtype=torch.bool, device=self.device)

        self.foot_targets_w = torch.zeros((self.num_envs, self.nf, 2), device=self.device)
        self.foot_pos_w = torch.zeros((self.num_envs, self.nf, 3), device=self.device)

        self._desired_com_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_com_pos[:, 2] = cfg.desired_height
        self._half_step = 0.5 * self._Ts
        self.w0 = torch.sqrt(torch.tensor(9.81 / self.cfg.desired_height, device=self.device))
        self.remaining_step_time = torch.full((self.num_envs, self.nf), 0.5*self._Ts, device=self.device)
        self.rdot_filt = torch.zeros((self.num_envs, 2), device=self.device)
        self.alpha = 0.5

        self.feet = {}
        # 0 = 'L', 1 = 'R'
        self.right = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # All 'R'
        self.left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # All 'L'

        # Change with num feet
        if self.nf == 4:
            self.sign_LR = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)
            self._phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            self.feet = {
                'BL': 0,
                'BR': 1,
                'FL': 2,
                'FR': 3,
            }
            self.p_feet_nominal = torch.tensor(
                [
                    [-0.2,  0.08, 0.0],    # BL
                    [-0.2, -0.08, 0.0],    # BR
                    [ 0.16,  0.08, 0.0],    # FL
                    [ 0.16, -0.08, 0.0],    # FR
                ],
                device=self.device,
            ).expand(self.num_envs, -1, -1)


            self._reach_min = torch.tensor(
                [
                    [cfg.lxy_caps[0], cfg.lxy_caps[2]],     # BL: back, inner
                    [cfg.rxy_caps[0], cfg.rxy_caps[2]],     # BR: back, outer
                    [cfg.lxy_caps[0], cfg.lxy_caps[2]],     # FL: back, inner
                    [cfg.rxy_caps[0], cfg.rxy_caps[2]],     # FR: back, outer
                ],
                device=self.device,
            )
            self._reach_max = torch.tensor(
                [
                    [cfg.lxy_caps[1], cfg.lxy_caps[3]],     # BL: back, inner
                    [cfg.rxy_caps[1], cfg.rxy_caps[3]],     # BR: back, outer
                    [cfg.lxy_caps[1], cfg.lxy_caps[3]],     # FL: back, inner
                    [cfg.rxy_caps[1], cfg.rxy_caps[3]],     # FR: back, outer
                ],
                device=self.device,
            )
        elif self.nf == 2:
            self.sign_LR = torch.tensor([1.0, -1.0], device=self.device)
            self._phase_offsets = torch.tensor([0.0, 0.5], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            self.feet = {
                'L': 0,
                'R': 1,
            }
            self.p_feet_nominal = torch.tensor(
                [
                    [0.0,  0.08, 0.0],    # L
                    [0.0, -0.08, 0.0],    # R
                ],
                device=self.device,
            ).expand(self.num_envs, -1, -1)

            self._reach_min = torch.tensor(
                [
                    [cfg.lxy_caps[0], cfg.lxy_caps[2]],     # Left: back, inner
                    [cfg.rxy_caps[0], cfg.rxy_caps[2]],     # Right: back, outer
                ],
                device=self.device,
            )
            self._reach_max = torch.tensor(
                [
                    [cfg.lxy_caps[1], cfg.lxy_caps[3]],     # Left: back, inner
                    [cfg.rxy_caps[1], cfg.rxy_caps[3]],     # Right: back, outer
                ],
                device=self.device,
            )


    """
    Properties.
    """

    @property
    def foot_targets(self):
        return self.foot_targets_w.flatten(start_dim=1)
    
    @property
    def foot_positions(self):
        return self.foot_pos_w.flatten(start_dim=1)

    @property
    def leg_state(self) -> torch.Tensor:
        return (~self._stance_bool)

    """
    Class functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self.foot_pos_w[env_ids] = self._asset.data.body_pos_w[env_ids][:, self._ee_body_ids].clone()
        self.foot_targets_w[env_ids] = self.foot_pos_w[env_ids, :, :2].clone()
        self._trot_phase[env_ids] = 0.0
        self._stance_bool[env_ids] = True
        self._stance_bool_prev[env_ids] = True
        self._swing_start[env_ids] = False
        self.remaining_step_time[env_ids] = self._half_step

    def apply_actions(self):
        cmd = self._env.command_manager.get_command(self.cfg.cmd_vel_name)
        yaw = self._yaw_from_quat_w(self._asset.data.root_quat_w)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cmd_x_w = cy * cmd[:, 0] - sy * cmd[:, 1]  # world X
        cmd_y_w = sy * cmd[:, 0] + cy * cmd[:, 1]  # world Y

        # ---------------- CONTACT SCHEDULE ----------------
        self._trot_phase = (self._trot_phase + (self._env.physics_dt / self._Ts)) % 1.0
        t = (self._trot_phase.unsqueeze(-1) + self._phase_offsets) % 1.0    # [E, nf]
        s = torch.sin(2 * torch.pi * t)
        self._phi_contact = s / torch.sqrt(s * s + 0.04)                    # [E, nf]

        self._stance_bool = self._phi_contact < 0.0                         # [E, nf]
        self._swing_start = (~self._stance_bool) & (self._stance_bool_prev) # [E, nf]
        self._stance_bool_prev = self._stance_bool.clone()                  # [E, nf]

        # ---------------- SUPPORT FOOT POSITIONS (CoP) ----------------
        stance_mask = self._stance_bool.unsqueeze(-1)                       # [E,nf,1]
        p_support_now = (self.p_feet_nominal * stance_mask).sum(1) / torch.clamp(stance_mask.sum(1), min=1)  # [E,3]
        p_support = torch.where(
            self._stance_bool.unsqueeze(-1),
            p_support_now.unsqueeze(1).expand(-1,self.nf,-1),
            self.p_feet_nominal.mean(1, keepdim=True).expand(-1,self.nf,-1),
        )  # [E,nf,3]

        # ---------------- LIP / ICP PLANNER ----------------
        r = self._asset.data.root_com_pos_w[:, :2]                          # [E, 2]
        rdot = self._asset.data.root_lin_vel_w[:, :2]                       # [E, 2]
        self.rdot_filt = self.alpha * self.rdot_filt + (1 - self.alpha) * rdot # [E, 2]
        zeta0 = r + self.rdot_filt / self.w0                                # [E, 2]

        e = torch.exp(self.w0 * self._half_step)                            # [E]

        zetaFx = e * zeta0[:, 0].unsqueeze(-1) + (1.0 - e) * p_support[:, :, 0] # [E, nf]
        zetaFy = e * zeta0[:, 1].unsqueeze(-1) + (1.0 - e) * p_support[:, :, 1] # [E, nf]

        sd = cmd_x_w.unsqueeze(-1) * self._half_step
        wd = (zetaFy - zeta0[:, 1].unsqueeze(-1)) \
            + 2.0 * (zeta0[:, 1].unsqueeze(-1) - p_support[:, :, 1]) \
            + (cmd_y_w.unsqueeze(-1) * self._half_step)
        
        bx = -sd / (e - 1.0)    # [E, nf]
        by = (wd / (e + 1.0)) * self.sign_LR    # [E, nf]

        # ---------------- APPLY TARGETS ----------------
        tgt_xy = torch.zeros((self.num_envs, self.nf, 2), device=self.device)
        tgt_xy[..., 0] = zetaFx + bx
        tgt_xy[..., 1] = zetaFy + by

        tgt_xy_clip = self.kinematic_reachability(tgt_xy)

        swing_mask = self._swing_start.unsqueeze(-1)                        # [E,nf,1]
        self.foot_targets_w = torch.where(
            swing_mask,
            tgt_xy_clip,
            self.foot_targets_w
        )
        self.foot_pos_w = self._asset.data.body_pos_w[:, self._ee_body_ids].clone()
        
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    """
    Helper functions.
    """

    def kinematic_reachability(self, tgt_xy: torch.Tensor) -> torch.Tensor:
        # Base pose for frame transforms
        base_pos_xy = self._asset.data.root_com_pos_w[:, :2]                       # [E,2]

        # Rotation from WORLD to BODY in XY (inverse yaw rotation)
        base_quat_w = self._asset.data.root_com_quat_w[:, :4]              # [E,4]
        yaw = self._yaw_from_quat_w(base_quat_w)                           # [E]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R_bw = torch.zeros(self.num_envs, 2, 2, device=self.device)
        R_bw[:, 0, 0] = cy
        R_bw[:, 0, 1] = sy
        R_bw[:, 1, 0] = -sy
        R_bw[:, 1, 1] = cy

        # Hip positions in world XY
        hip_xy_w = self._asset.data.body_pos_w[:, self._ee_body_ids, :2]          # [E,4,2]

        # Relative to base in world frame
        hip_rel_w = hip_xy_w - base_pos_xy.unsqueeze(1)                           # [E,4,2]
        tgt_rel_w = tgt_xy   - base_pos_xy.unsqueeze(1)                           # [E,4,2]

        # Transform to BODY frame: [E,4,2] = R_bw[e] @ rel_w[e,leg]
        hip_rel_b = torch.einsum("eij,ekj->eki", R_bw, hip_rel_w)                 # [E,4,2]
        tgt_rel_b = torch.einsum("eij,ekj->eki", R_bw, tgt_rel_w)                 # [E,4,2]

        # Target relative to hip in BODY frame
        rel_to_hip_b = tgt_rel_b - hip_rel_b                                      # [E,4,2]

        # Broadcast reach limits: [E,4,2]
        reach_min = self._reach_min.unsqueeze(0)                                  # [1,4,2]
        reach_max = self._reach_max.unsqueeze(0)                                  # [1,4,2]

        # Clamp to reachable box per leg
        rel_clamped_b = torch.max(torch.min(rel_to_hip_b, reach_max), reach_min)  # [E,4,2]

        # Back to body-frame target positions relative to base
        tgt_clamped_b = hip_rel_b + rel_clamped_b                                 # [E,4,2]

        # Transform back to WORLD frame
        R_wb = R_bw.transpose(1, 2)                                               # [E,2,2]
        tgt_clamped_rel_w = torch.einsum("eij,ekj->eki", R_wb, tgt_clamped_b)     # [E,4,2]
        tgt_xy = tgt_clamped_rel_w + base_pos_xy.unsqueeze(1)                     # [E,4,2]

        return tgt_xy

    def _yaw_from_quat_w(self, qwxyz: torch.Tensor) -> torch.Tensor:
        w = qwxyz[:, 0]
        x = qwxyz[:, 1]
        y = qwxyz[:, 2]
        z = qwxyz[:, 3]

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)
    

class QuadcopterPDAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.QuadcopterPDActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.QuadcopterPDActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._leg_actions = torch.zeros(self.num_envs, 12, device=self.device)

        self._root_id = self._asset.find_bodies(self.cfg.root_name, preserve_order=True)[0] # for the root body
        self._leg_joint_ids = self._asset.find_joints(self.cfg.leg_joint_names, preserve_order=True)[0] # for the legs

        self._legs_offset = torch.tensor([
            1.57, -1.57, 1.57, -1.57,  # bl_hx, br_hx, fl_hx, fr_hx
            0.698132, 0.698132, 0.698132, 0.698132,  # bl_hy, br_hy, fl_hy, fr_hy
            0.4, 0.4, 0.4, 0.4,  # bl_kn, br_kn, fl_kn, fr_kn
        ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        self._gravity_w = torch.tensor([0.0, 0.0, 9.81], device=self.device).repeat(self.num_envs, 1)
        self._base_to_com_pos_w = self._asset.data.root_com_pos_w - self._asset.data.root_pos_w
        self._gravity_force = self.cfg.robot_mass * self._gravity_w

        self.desired_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.desired_rpy_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.kp_lin = torch.tensor(cfg.kp_lin, device=self.device)
        self.kd_lin = torch.tensor(cfg.kd_lin, device=self.device)
        self.kp_ang = torch.tensor(cfg.kp_ang, device=self.device)
        self.kd_ang = torch.tensor(cfg.kd_ang, device=self.device)

        self._env_ids = []
        self._velocity_command = torch.zeros((self.num_envs, 4), device=self.device)
        self._max_forces = torch.tensor([12.0, 12.0, 70.0], device=self.device)

    """
    Properties.
    """

    @property
    def leg_positions(self):
        return self._legs_offset

    """
    Class Functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self.desired_pos_w[env_ids] = self._asset.data.root_pos_w[env_ids].clone()
        self.desired_rpy_w[env_ids, 2] = self.quat_to_rpy(self._asset.data.root_quat_w[env_ids].clone())[:, 2]
        
    def apply_actions(self):
        self._thrust.zero_()
        self._moment.zero_()
        self._base_to_com_pos_w = self._asset.data.root_com_pos_w - self._asset.data.root_pos_w

        if self._env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(self._env_ids, device=self.device, dtype=torch.long)

        if env_ids_t.numel() > 0:
            current_pos_w = self._asset.data.root_pos_w[env_ids_t].clone()
            current_quat = self._asset.data.root_quat_w[env_ids_t].clone()
            current_lin_vel = self._asset.data.root_lin_vel_w[env_ids_t].clone()
            current_ang_vel = self._asset.data.root_ang_vel_w[env_ids_t].clone()

            # desired_velocity = self._env.command_manager.get_command(self.cfg.cmd_vel_name)[env_ids_t]
            desired_velocity = self._velocity_command[env_ids_t]
            desired_lin_vel = desired_velocity[:, 0:3]
            desired_ang_vel = torch.cat(
                [
                    torch.zeros((env_ids_t.numel(), 1), device=self.device),
                    torch.zeros((env_ids_t.numel(), 1), device=self.device),
                    desired_velocity[:, 3].unsqueeze(-1),
                ],
                dim=1,
            )
            desired_lin_vel_w = math_utils.quat_rotate(current_quat, desired_lin_vel)
            desired_ang_vel_w = math_utils.quat_rotate(current_quat, desired_ang_vel)
            desired_ang_vel_w[:, :2] *= 0.0

            self.desired_pos_w[env_ids_t] = torch.where(
                (torch.abs(desired_lin_vel_w) > 0.05),
                self.desired_pos_w[env_ids_t] + desired_lin_vel_w * self._env.physics_dt,
                self.desired_pos_w[env_ids_t],
            )
            self.desired_rpy_w[env_ids_t, 2] = torch.where(
                (torch.abs(desired_ang_vel_w[:, 2]) > 0.1),
                self.desired_rpy_w[env_ids_t, 2] + desired_ang_vel_w[:, 2] * self._env.physics_dt,
                self.desired_rpy_w[env_ids_t, 2],
            )
            desired_quat_w = self.rpy_to_quat(self.desired_rpy_w[env_ids_t])
            # Ensure shortest path rotation
            dot = torch.sum(desired_quat_w * current_quat, dim=-1, keepdim=True)
            desired_quat_w = torch.where(dot < 0, -desired_quat_w, desired_quat_w)

            a_des = self.kp_lin * (self.desired_pos_w[env_ids_t] - current_pos_w) + \
                    self.kd_lin * (desired_lin_vel_w - current_lin_vel)
            a_des += self._gravity_w[env_ids_t]
            Iw = self._compute_composite_inertia_world()[env_ids_t]

            alpha_des = self.kp_ang * math_utils.quat_box_minus(desired_quat_w, current_quat) + \
                        self.kd_ang * (desired_ang_vel_w - current_ang_vel)

            force_w = self.cfg.robot_mass * a_des
            torque_w = torch.bmm(alpha_des.unsqueeze(1), Iw).squeeze(1) \
                + torch.cross(torch.bmm(current_ang_vel.unsqueeze(1), Iw).squeeze(1), current_ang_vel, dim=-1) \
                - torch.cross(self._base_to_com_pos_w[env_ids_t], self._gravity_force[env_ids_t], dim=-1)

            self._thrust[env_ids_t, 0, :] = math_utils.quat_rotate_inverse(
                current_quat, 
                torch.clamp(force_w, -self._max_forces, self._max_forces),
            )
            self._moment[env_ids_t, 0, :] = math_utils.quat_rotate_inverse(
                current_quat, 
                torch.clamp(torque_w, -10.0, 10.0),
            )

        # self._asset.set_joint_position_target(self._leg_actions + self._legs_offset, joint_ids=self._leg_joint_ids) # send zero commands for all leg joints
        self._asset.set_external_force_and_torque(self._thrust, self._moment, self._root_id) # set force and momentum for root link

    """
    Helper Functions.
    """

    def set_velocity_command(self, vel_cmd: torch.Tensor):
        self._velocity_command = vel_cmd

    def set_env_ids(self, ids):
        self._env_ids = ids

    def _compute_composite_inertia_world(self):
        """
        Compute the centroidal (composite) inertia about the whole-body COM in WORLD.
        Returns:
            Iw:    (E, 3, 3)
        """
        E = self.num_envs

        # --- Per-body masses & local inertias (COM-local, body frame) from PhysX view ---
        av = self._asset.root_physx_view
        masses = av.get_masses().to(device=self.device)                 # (E, L) or (L,) → Isaac Lab returns (E,L)
        inertias_flat = av.get_inertias().to(device=self.device)        # (E, L, 9) flattened 3x3 in COM-local frame
        L = inertias_flat.shape[1]

        # reshape to (E,L,3,3)
        I_body_local = inertias_flat.view(E, L, 3, 3)

        # --- World poses of each link's COM (from ArticulationData) ---
        p_com_w      = self._asset.data.root_com_pos_w                      # (E, 3)
        p_link_com_w = self._asset.data.body_com_pos_w                      # (E, L, 3)
        q_link_com_w = self._asset.data.body_com_quat_w                     # (E, L, 4)

        # Rotations to world: (E,L,3,3)
        R_wi = math_utils.matrix_from_quat(q_link_com_w)

        # Rotate inertias to world: I_wi = R * I_local * R^T
        I_w_links = R_wi @ I_body_local @ R_wi.transpose(-1, -2)

        # Offsets from whole-body COM to link COM (world)
        r = p_link_com_w - p_com_w.unsqueeze(1)  # (E,L,3)

        # Skew(r) for parallel-axis: S(r) so that S @ S^T = [r]_x [r]_x^T
        z = torch.zeros_like(r[..., 0])
        S = torch.stack([
            torch.stack([ z,          -r[..., 2],  r[..., 1]], dim=-1),
            torch.stack([ r[..., 2],   z,         -r[..., 0]], dim=-1),
            torch.stack([-r[..., 1],   r[..., 0],  z        ], dim=-1),
        ], dim=-2)  # (E,L,3,3)

        # Parallel-axis term: m_i * S @ S^T
        m_e = masses.unsqueeze(-1).unsqueeze(-1)  # (E,L,1,1)
        PA  = torch.matmul(S, S.transpose(-1, -2)) * m_e  # (E,L,3,3)

        # Sum contributions over links
        Iw = (I_w_links + PA).sum(dim=1)  # (E,3,3)

        return Iw
    
    def quat_to_rpy(self, q: torch.Tensor):
        qw = q[..., 0]
        qx = q[..., 1]
        qy = q[..., 2]
        qz = q[..., 3]

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        # clamp for numerical safety
        sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp_clamped)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        euler = torch.stack((roll, pitch, yaw), dim=-1)
        return euler
    
    def rpy_to_quat(self, rpy: torch.Tensor):
        roll  = rpy[..., 0]
        pitch = rpy[..., 1]
        yaw   = rpy[..., 2]

        half_roll  = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw   = yaw * 0.5

        cr = torch.cos(half_roll)
        sr = torch.sin(half_roll)
        cp = torch.cos(half_pitch)
        sp = torch.sin(half_pitch)
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)

        # ZYX composition: q = q_yaw * q_pitch * q_roll
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return quat
    

class HuskyBetaMultimodalJointPosAction(ActionTerm):
    cfg: actions_cfg.HuskyBetaMultimodalJointPosActionCfg
    _asset: Articulation
    _scale: torch.Tensor | float 

    def __init__(self, cfg: actions_cfg.HuskyBetaMultimodalJointPosActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._env = env

        """
        Actions: 
        [0:3]   - desired linear velocity x,y,z
        [3]     - desired angular velocity yaw
        [4]     - mode switch: walk
        [5]     - mode switch: flight
        6 total
        """
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        
        # State machine: 0 = walk, 1 = flight
        self._mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_mode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        ############### Action Loads ###############
        if not check_file_path(cfg.walk_policy_path):
            raise FileNotFoundError(f"Pretrained walk policy file not found at: {cfg.walk_policy_path}")
        
        walk_file_bytes = read_file(cfg.walk_policy_path)
        self.walk_policy = torch.jit.load(walk_file_bytes).to(env.device).eval()
        self._walk_policy_action_term: ActionTerm = cfg.walk_policy_actions.class_type(cfg.walk_policy_actions, env)
        self.walk_policy_actions = torch.zeros(self.num_envs, self._walk_policy_action_term.action_dim, device=self.device)
        self._flight_action = cfg.flight_action.class_type(cfg.flight_action, env)

        # UNIFIED JOINT TARGET BUFFER (this is what gets applied to the robot)
        num_joints = len(env.scene[cfg.asset_name].find_joints(cfg.walk_policy_actions.joint_names)[0])
        self._unified_joint_targets = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._joint_ids = env.scene[cfg.asset_name].find_joints(cfg.walk_policy_actions.joint_names)[0]
        
        ############### Observation Mutations ###############
        def last_action(action):
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                action[env.episode_length_buf == 0, :] = 0
            return action
        
        cfg.walk_policy_observations.actions.func = lambda dummy_env: last_action(self.walk_policy_actions)
        cfg.walk_policy_observations.actions.params = dict()
        cfg.walk_policy_observations.velocity_commands.func = lambda dummy_env: torch.cat([self.processed_actions[:, :2], self.processed_actions[:, 3].unsqueeze(-1)], dim=-1)
        cfg.walk_policy_observations.velocity_commands.params = dict()
        cfg.walk_policy_observations.foot_targets.params["command_name"] = "multimodal_command"
        cfg.walk_policy_observations.contact_schedule.params["command_name"] = "multimodal_command"

        self._low_level_obs_manager = ObservationManager(
            {
                "walk_policy": cfg.walk_policy_observations,
            },
            env,
        )

        self._walk_counter = 0
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 6
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def mode(self) -> torch.Tensor:
        return self._mode
    
    @property
    def prev_mode(self) -> torch.Tensor:
        return self._prev_mode

    @property
    def foot_targets(self) -> torch.Tensor:
        return self._walk_policy_action_term.foot_targets

    @property
    def swing_mask(self) -> torch.Tensor:
        return self._walk_policy_action_term.leg_state
    
    @property
    def time_left_s(self) -> torch.Tensor:
        return (self._env.max_episode_length_s - self._env.episode_length_buf * self._env.step_dt).unsqueeze(-1)

    """
    Class functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        # self._mode[env_ids] = 0 # maybe random sample this 
        self._prev_mode[env_ids] = self._mode[env_ids].clone()
        self._walk_counter = 0
        self._walk_policy_action_term.reset(env_ids)
        self._flight_action.reset(env_ids)
        self._low_level_obs_manager.reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions 
        self._processed_actions[:, :4] = torch.clip(
            self._raw_actions[:, :4] * float(self._scale),
            -1.0, 
            1.0,
        )
        self._processed_actions[:, 4:] = self._raw_actions[:, 4:]

        self._prev_mode = self._mode.clone()
        self._mode[:] = torch.argmax(self._processed_actions[:, 4:6], dim=-1).to(torch.int32)

    def apply_actions(self):        
        walk_des_vel = torch.cat([self.processed_actions[:, :2], self.processed_actions[:, 3].unsqueeze(-1)], dim=-1)
        self._walk_policy_action_term._set_velocity_cmd(walk_des_vel)

        start_flight = (self._mode == 1) & (self._prev_mode != 1)
        if start_flight.any():
            env_ids = start_flight.nonzero(as_tuple=False).squeeze(-1)
            self._flight_action.reset(env_ids)
        
        walk_targets = self._compute_walk_targets()           # (num_envs, num_joints)
        flight_targets = self._compute_flight_targets()       # Special: legs tucked + flight controller
        
        in_walk = self._mode == 0
        in_flight = self._mode == 1
        
        self._unified_joint_targets = torch.where(
            in_walk.unsqueeze(-1),
            walk_targets,
            self._unified_joint_targets
        )
        self._unified_joint_targets = torch.where(
            in_flight.unsqueeze(-1),
            flight_targets,
            self._unified_joint_targets
        )
        
        self.robot.set_joint_position_target(self._unified_joint_targets, joint_ids=self._joint_ids)
        
        self._flight_action.set_velocity_command(self._processed_actions[:, 0:4])
        if in_flight.any():
            env_ids = in_flight.nonzero(as_tuple=False).squeeze(-1).tolist()
            self._flight_action.set_env_ids(env_ids)
        else:
            env_ids = []
            self._flight_action.set_env_ids(env_ids)

        self._flight_action.apply_actions()

    """
    Helper functions.
    """

    def _compute_walk_targets(self) -> torch.Tensor:
        self._walk_policy_action_term.apply_actions()

        if self._walk_counter % self.cfg.low_level_decimation == 0:
            walk_obs = self._low_level_obs_manager.compute_group("walk_policy")
            self.walk_policy_actions[:] = self.walk_policy(walk_obs)
            self._walk_policy_action_term.process_actions(self.walk_policy_actions)
            self._walk_counter = 0
        
        self._walk_counter += 1
        
        return self._walk_policy_action_term.processed_actions

    def _compute_flight_targets(self) -> torch.Tensor:
        return self._flight_action.leg_positions
