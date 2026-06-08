# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", 
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.65,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "knees": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=45.0,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)