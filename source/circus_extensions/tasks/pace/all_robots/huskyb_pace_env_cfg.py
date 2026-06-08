# © 2026 Northeastern University, Silicon Synapse Lab
# Author: Arjun Viswanathan
# Licensed under the Apache License 2.0

import torch

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from circus_extensions.assets.husky_beta import HUSKY_B_CFG
from circus_extensions.simulation.actuators import PaceDCMotorCfg
from circus_extensions.tasks.pace.all_robots.pace_sim2real_env_cfg import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg

DYNAMIXEL_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hx", ".*_hy", ".*_kn"],
    saturation_effort=9.9,
    effort_limit=9.9,
    velocity_limit=3.0,
    stiffness={".*_h[xy]": 35.0, ".*_kn": 40.0},  # P gain in Nm/rad
    damping={".*_h[xy]": 1.0, ".*_kn": 1.25},  # D gain in Nm s/rad
    encoder_bias=[0.0] * 12,  # encoder bias in radians
    max_delay=10,  # max delay in simulation steps
)


@configclass
class HuskyBetaPaceCfg(PaceCfg):
    """Pace configuration for Husky Beta robot."""
    robot_name: str = "husky_beta_sim"
    data_dir: str = "husky_beta_sim/chirp_data.pt"  # located in pace_sim2real/data/anymal_d_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "bl_hx", "br_hx", "fl_hx", "fr_hx",
        "bl_hy", "br_hy", "fl_hy", "fr_hy",
        "bl_kn", "br_kn", "fl_kn", "fr_kn",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class HuskyBetaPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Husky Beta robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators={"legs": DYNAMIXEL_PACE_ACTUATOR_CFG})


@configclass
class HuskyBetaPaceEnvCfg(PaceSim2realEnvCfg):

    scene: HuskyBetaPaceSceneCfg = HuskyBetaPaceSceneCfg()
    sim2real: PaceCfg = HuskyBetaPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.005  # 200Hz simulation
        self.decimation = 1  # 200Hz control
