# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_CFG`: The Spot robot with delay PD and remote PD actuators.
"""

import isaaclab.sim as sim_utils
from circus_extensions.simulation.actuators.actuator_cfg import PowerLimitedPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
import subprocess 

##
# Configuration
##

JOINT_ORDER = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",  # hip_x
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",  # hip_y
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",  # knee
    "arm0_sh0", "arm0_sh1",  # arm shoulder yaw, pitch
    "arm0_el0", "arm0_el1",  # arm elbow
    "arm0_wr0", "arm0_wr1",  # arm wrist
]

SPOT_BADMINTON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/robotarium/spot/usd/spot_badminton/spot_badminton.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.525),
        joint_pos={
            ".*_hx": 0.0,  # all hip_x
            "f[rl]_hy": 0.9,  # front hip_y
            "h[rl]_hy": 1.1,  # hind hip_y
            ".*_kn": -1.5,  # all knees
            "arm0_sh0": 0.0,  # arm shoulder yaw
            "arm0_sh1": -3.14,  # arm shoulder pitch
            "arm0_el0": 2.7,  # arm elbow
            "arm0_el1": 0.0,  # arm elbow
            "arm0_wr0": -0.83,  # arm wrist
            "arm0_wr1": 0.0,  # arm wrist
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_arm": DelayedPDActuatorCfg(
            joint_names_expr=["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"],
            effort_limit={
                "arm0_sh0": 90.9,
                "arm0_sh1": 181.8,
                "arm0_el0": 90.9,
                "arm0_el1": 30.3,
                "arm0_wr0": 30.3,
                "arm0_wr1": 30.3,
            },
            stiffness=100.0,
            damping=2.0,
            friction=0.015,
            # motor_strength_range=(0.85, 1.0),
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_hips": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=45.0,
            stiffness=100.0,
            damping=2.0,
            friction=0.008,
            # motor_strength_range=(0.85, 1.0),
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_knees": PowerLimitedPDActuatorCfg(
            joint_names_expr=[".*_kn"],
            effort_limit=None,  # torque limits are handled based experimental data (`RemotizedPDActuatorCfg.data`)
            stiffness=100.0,
            damping=2.0,
            friction=0.018,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
            m1=-6.23,
            m2=-7.83,
            b1=155.93,
            b2=-173.98,
            max_torque=97.0,
            min_torque=-108.79,
        ),
    },
)
"""Configuration for the Boston Dynamics Spot robot."""
