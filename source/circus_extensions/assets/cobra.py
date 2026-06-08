# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.actuators as acts
from isaaclab.assets.articulation import ArticulationCfg
import subprocess
import os 

##
# Configuration
##

COBRA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/robotarium/cobra/usd/cobra.usda",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "joints": acts.DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=10.0,
            stiffness=25.0,
            damping=0.75,
            min_delay=0,
            max_delay=4
        )
        # "joints": acts.ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit=2.0,
        #     stiffness=25.0,
        #     damping=2.0,
        # )
    },
)