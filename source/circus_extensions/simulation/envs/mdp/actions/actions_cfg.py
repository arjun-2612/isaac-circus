# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import ActionTerm
from isaaclab.envs.mdp.actions import JointActionCfg
from isaaclab.utils import configclass

from . import joint_actions

##
# Joint actions.
##


@configclass
class QuadcopterPDActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.QuadcopterPDAction

    leg_joint_names: list[str] = MISSING
    robot_mass: float = MISSING
    root_name: str = MISSING
    kp_lin: list = MISSING 
    kd_lin: list = MISSING 
    kp_ang: list = MISSING 
    kd_ang: list = MISSING

@configclass
class QuadcopterActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.QuadcopterAction

    thrust_to_weight_ratio: float = MISSING
    moment_scale: float = MISSING
    robot_weight: float = MISSING
    leg_joint_names: list[str] = MISSING
    root_name: str = MISSING

@configclass
class ReferencePositionActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = joint_actions.ReferencePositionAction

    reference_name: str = MISSING

@configclass
class RMPCActionCfg(JointActionCfg):

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = joint_actions.RMPCAction

    ee_name: str = MISSING
    """The end effector names, can specify by regex."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t the body frame. Defaults to None, in which case no offset is applied."""

    kp: float = MISSING
    """The Kp used by joints for PD control."""

    kd: float = MISSING
    """The Kd used by joints for PD control."""

    sensor_name: str = MISSING
    """Name of a contact/force sensor in env.sensors to detect stance vs swing. If None, all legs treated as swing."""

    cmd_vel_name: str = MISSING
    """Name of another CommandTerm that outputs desired planar velocity per env (expects at least [vx, vy])."""
    
    hx_bodies: str = MISSING 
    """Names of the hip bodies in the asset."""

    hy_bodies: str = MISSING 
    """Names of the hip bodies in the asset."""

    hy_length: float = MISSING

    kn_length: float = MISSING

    ank_length: float = 0.0

    knee_sign: float = MISSING

    ankle_length: float = 0.0

    k_joints: str = None 

    a_joints: str = None 

    has_ankle: bool = False

    # --- QP parameters ---
    desired_height: float = MISSING
    """Desired height of the robot COM (m)."""
    
    friction_coeff: float = MISSING
    """Friction coefficient to use for motion."""

    f_min_ratio: float = MISSING
    """Minimum force to exert."""

    f_max_ratio: float = MISSING
    """Maximum force to exert."""

    robot_mass: float = MISSING
    """Mass of the robot (kg)."""

    foot_clearance: float = 0.1
    """Clearance of the swing feet (m)."""

    # --- Heuristic parameters ---
    step_time: float = 0.7
    """Step period Ts (s) used in x_foot = x_hip + 0.5*Ts*v + k*(v - v_des)."""

    smoothing_factor: float = 1.0

    duty_cycle: float = 0.5

    kp_x: float = 0.10
    """Raibert gain for forward velocity correction."""

    kp_y: float = 0.10
    """Raibert gain for lateral velocity correction."""

    kp_w: float = 0.10
    """Raibert gain for angular velocity correction."""

    max_step_len: float = 0.25
    """Clamp for |foot_target - hip| in x/y (m)."""

    horizon_length: int = MISSING
    """Horizon length for MPC (number of timesteps)."""

    q_diag: list = MISSING
    """State cost weights for MPC (list of 12 floats)."""

    r_diag: list = MISSING
    """Control cost weights for MPC (list of 4 floats)."""

    decimation: int = MISSING 
    """The physics frequency / decimation = MPC frequency"""

    lxy_caps: list = MISSING 

    rxy_caps: list = MISSING


@configclass
class LIPMActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = joint_actions.LIPMAction

    ee_name: str = MISSING
    """The end effector names, can specify by regex."""

    cmd_vel_name: str = MISSING
    """Name of another CommandTerm that outputs desired planar velocity per env (expects at least [vx, vy])."""

    desired_height: float = MISSING
    """Desired height of the robot COM (m)."""

    step_time: float = 0.7
    """Step period Ts (s) used in x_foot = x_hip + 0.5*Ts*v + k*(v - v_des)."""

    step_width: float = 0.0

    lxy_caps: list = MISSING 

    rxy_caps: list = MISSING

    num_feet: int = MISSING