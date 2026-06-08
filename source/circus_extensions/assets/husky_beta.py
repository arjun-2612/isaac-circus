# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from circus_extensions.simulation.actuators import PaceDCMotorCfg

import subprocess
import os 

##
# Configuration
##

JOINT_ORDER = [
    "bl_hx", "br_hx", "fl_hx", "fr_hx", 
    "bl_hy", "br_hy", "fl_hy", "fr_hy",
    "bl_kn", "br_kn", "fl_kn", "fr_kn",
]

HUSKY_B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/robotarium/husky_beta/usd/huskyb_full/huskyb_full.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "[fb]l_hx": 0.0,  # all left frontal hip joints
            "[fb]r_hx": 0.0,  # all left frontal hip joints
            "f[lr]_hy": 0.698132,  # all front sagittal hip joints
            "b[lr]_hy": 0.698132,  # all back sagittal hip joints
            "f[lr]_kn": 0.4,  # all front knee joints
            "b[lr]_kn": 0.4,  # all back knee joints
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": PaceDCMotorCfg(
            joint_names_expr=JOINT_ORDER,
            saturation_effort=9.9,
            effort_limit=9.9,
            velocity_limit=3.0,
            stiffness={".*_h[xy]": 35.0, ".*_kn": 40.0},
            damping={".*_h[xy]": 1.0, ".*_kn": 1.25},
            armature={".*_h[xy]": 0.0003, ".*_kn": 0.006},
            friction={".*_h[xy]": 0.057, ".*_kn": 0.25},
            encoder_bias=[0.0] * 12,
            max_delay=1,
        ),
        # "legs": DCMotorCfg(
        #     joint_names_expr=JOINT_ORDER,
        #     saturation_effort=9.9,
        #     effort_limit=9.9,
        #     velocity_limit=3.0,
        #     stiffness={".*_h[xy]": 35.0, ".*_kn": 40.0},
        #     damping={".*_h[xy]": 1.0, ".*_kn": 1.25},
        # ),
    },
)

"""Configuration for the Husky Beta robot."""