# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
import subprocess
import os 

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from circus_extensions.simulation.terrains import TerrainImporterCfg
import circus_extensions.tasks.locomotion.husky_beta.mdp as huskyb_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.husky_beta import HUSKY_B_CFG, JOINT_ORDER # isort: skip

PLAY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, 
            noise_range=(0.0, 0.0), 
            noise_step=0.01, 
            border_width=0.25, 
        ),
    },
)

@configclass
class HuskyBetaDTCCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = huskyb_mdp.isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=huskyb_mdp.isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-0.8, 0.8), 
        ),
    )

    # Debug vis to true throws a lot of warnings. For now, this is not important
    controller = huskyb_mdp.RMPCCommandCfg(
        asset_name="robot",
        debug_vis=False,
        resampling_time_range=(10.0, 10.0),
        action_name="joint_pos",
        foot_bodies=".*_foot",
    )


@configclass
class HuskyBetaDTCActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = huskyb_mdp.RMPCActionCfg(
        asset_name="robot",
        joint_names=JOINT_ORDER,
        scale=0.25,
        ee_name=".*_foot",
        preserve_order=True,
        kp=25.0, 
        kd=0.5,
        body_offset=huskyb_mdp.RMPCActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0], 
            rot=[0.9396, 0.0, 0.342, 0.0],
        ),
        sensor_name="contact_forces",
        cmd_vel_name="base_velocity",
        hx_bodies=".*_hip",
        hy_bodies=".*_uleg",
        hy_length=0.1746,
        kn_length=0.3225,
        ank_length=0.15,
        has_ankle=True,
        k_joints=".*_kn",
        a_joints=".*_ank",
        knee_sign=1.0,
        foot_clearance=0.25,
        duty_cycle=0.5,
        step_time=0.7,
        smoothing_factor=0.5,
        kp_x=0.15,
        kp_y=0.03,
        kp_w=0.05,
        max_step_len=0.5,
        desired_height=0.47,
        friction_coeff=0.6,
        f_min_ratio=0.03,
        f_max_ratio=1.5,
        robot_mass=6.11,
        horizon_length=12,
        q_diag=[5.0, 40.0, 1.0, 5.0, 5.0, 50.0, 0.01, 0.1, 0.2, 10.0, 10.0, 1.5, 0.0],
        r_diag=[5e-4, 5e-4, 1e-3, 5e-4, 5e-4, 1e-3, 5e-4, 5e-4, 1e-3, 5e-4, 5e-4, 1e-3],
        decimation=4,
        lxy_caps=[-0.1, 0.1, -0.025, 0.1],
        rxy_caps=[-0.1, 0.1, -0.1, 0.025],
    )


@configclass
class HuskyBetaDTCObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Proprioceptive observations
        base_lin_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_lin_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Gnoise(mean=0.0, std=0.1),
        )
        base_ang_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Gnoise(mean=0.0, std=0.02),
        )
        base_6do = ObsTerm(
            func=huskyb_mdp.body_6do, 
            params={"asset_cfg": SceneEntityCfg("robot"), "noise": (0.0, 0.02)},
        )
        velocity_commands = ObsTerm(func=huskyb_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=huskyb_mdp.isaac.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
            noise=Gnoise(mean=0.0, std=0.002),
        )
        joint_vel = ObsTerm(
            func=huskyb_mdp.isaac.joint_vel_rel, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)
            }, 
            noise=Gnoise(mean=0.0, std=0.05),
        )
        actions = ObsTerm(func=huskyb_mdp.isaac.last_action)

        # Model-based planner related
        foot_targets = ObsTerm(
            func=huskyb_mdp.foot_targets, 
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "controller"},
        )
        contact_schedule = ObsTerm(
            func=huskyb_mdp.contact_schedule,
            params={"command_name": "controller"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_6do = ObsTerm(
            func=huskyb_mdp.body_6do, 
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=huskyb_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=huskyb_mdp.isaac.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )
        joint_vel = ObsTerm(
            func=huskyb_mdp.isaac.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )
        actions = ObsTerm(func=huskyb_mdp.isaac.last_action)

        # Model-based planner related
        foot_targets = ObsTerm(
            func=huskyb_mdp.foot_targets, 
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "controller"},
        )
        contact_schedule = ObsTerm(
            func=huskyb_mdp.contact_schedule,
            params={"command_name": "controller"},
        )

        # Privileged observations
        foot_positions = ObsTerm(
            func=huskyb_mdp.foot_positions, 
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        base_height = ObsTerm(
            func=huskyb_mdp.isaac.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_wrench = ObsTerm(
            func=huskyb_mdp.isaac.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class HuskyBetaDTCEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=huskyb_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.25),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=huskyb_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.25, 0.25),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=huskyb_mdp.isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=huskyb_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER),
        },
    )

    # interval
    push_robot = EventTerm(
        func=huskyb_mdp.isaac.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
        },
    )

@configclass
class HuskyBetaDTCRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=huskyb_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.35,
            "velocity_threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=huskyb_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=huskyb_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=huskyb_mdp.foot_clearance_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    foot_target_tracking = RewardTermCfg(
        func=huskyb_mdp.foot_target_tracking_reward,
        weight=0.75,
        params={
            "command_name": "controller",
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- penalties
    base_height = RewardTermCfg(
        func=huskyb_mdp.isaac.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.53,
        },
    )
    action_smoothness = RewardTermCfg(func=huskyb_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=huskyb_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=huskyb_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=huskyb_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=huskyb_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_pos = RewardTermCfg(
        func=huskyb_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.1,
        },
    )
    # joint_vel = RewardTermCfg(
    #     func=huskyb_mdp.hb_rewards.joint_velocity_penalty,
    #     weight=-1.0e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    # joint_acc = RewardTermCfg(
    #     func=huskyb_mdp.hb_rewards.joint_acceleration_penalty,
    #     weight=-1.0e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    joint_torques = RewardTermCfg(
        func=huskyb_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
    )
    joint_power = RewardTermCfg(
        func=huskyb_mdp.power_penalty,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER), "sharpness": 0.15, "threshold": 50.0},
    )


@configclass
class HuskyBetaDynamicTrottingRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=huskyb_mdp.hb_rewards.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.35,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    contact_time = RewardTermCfg(
        func=huskyb_mdp.hb_rewards.contact_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.35,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=huskyb_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=huskyb_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=huskyb_mdp.foot_clearance_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    foot_target_tracking = RewardTermCfg(
        func=huskyb_mdp.foot_target_tracking_reward,
        weight=0.5,
        params={
            "command_name": "controller",
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- penalties
    base_height = RewardTermCfg(
        func=huskyb_mdp.isaac.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.53,
        },
    )
    action_smoothness = RewardTermCfg(func=huskyb_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=huskyb_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=huskyb_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=huskyb_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=huskyb_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_pos = RewardTermCfg(
        func=huskyb_mdp.hb_rewards.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_vel = RewardTermCfg(
        func=huskyb_mdp.hb_rewards.joint_velocity_penalty,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc = RewardTermCfg(
        func=huskyb_mdp.hb_rewards.joint_acceleration_penalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_torques = RewardTermCfg(
        func=huskyb_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
    )
    joint_power = RewardTermCfg(
        func=huskyb_mdp.power_penalty,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER), "sharpness": 0.15, "threshold": 50.0},
    )
    

@configclass
class HuskyBetaDTCTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=huskyb_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=huskyb_mdp.isaac.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["body", ".*_hip", ".*leg", ".*_ankle"],
            ), 
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=huskyb_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )


@configclass
class HuskyBetaDTCCurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class HuskyBetaDTCEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: HuskyBetaDTCObservationsCfg = HuskyBetaDTCObservationsCfg()
    actions: HuskyBetaDTCActionsCfg = HuskyBetaDTCActionsCfg()
    commands: HuskyBetaDTCCommandsCfg = HuskyBetaDTCCommandsCfg()

    # MDP setting
    # rewards: HuskyBetaRewardsCfg = HuskyBetaRewardsCfg()
    rewards: HuskyBetaDynamicTrottingRewardsCfg = HuskyBetaDynamicTrottingRewardsCfg()
    terminations: HuskyBetaDTCTerminationsCfg = HuskyBetaDTCTerminationsCfg()
    events: HuskyBetaDTCEventCfg = HuskyBetaDTCEventCfg()
    curriculum: HuskyBetaDTCCurriculumCfg = HuskyBetaDTCCurriculumCfg()

    # recorders: huskyb_mdp.RecorderManagerCfg = huskyb_mdp.RecorderManagerCfg(
    #     dataset_export_dir_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/source/sslab_extensions/simulation/isaaclab_data/huskyb/dtc_env",
    #     dataset_filename=f"dtc_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}",
    # )

    viewer = ViewerCfg(eye=(2.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4 # 50 Hz
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 1.0/200.0  # 200 Hz

        # switch robot to HuskyBeta-d
        self.scene.robot = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=PLAY_TERRAIN_CFG,
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None


class HuskyBetaDTCEnvCfg_PLAY(HuskyBetaDTCEnvCfg):

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = True
        self.events.push_robot = None
        self.commands.controller.debug_vis = True