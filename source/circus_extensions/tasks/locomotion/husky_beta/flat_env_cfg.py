# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import subprocess
from datetime import datetime
import math 

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import circus_extensions.tasks.locomotion.husky_beta.mdp as husky_beta_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.husky_beta import HUSKY_B_CFG, JOINT_ORDER  # isort: skip

@configclass
class HuskyBetaFlatActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = husky_beta_mdp.isaac.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=JOINT_ORDER, 
        scale=0.25, 
        use_default_offset=True, 
        preserve_order=True
    )

@configclass
class HuskyBetaFlatCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = husky_beta_mdp.isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=husky_beta_mdp.isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.25, 1.25), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
        ),
    )

@configclass
class HuskyBetaFlatObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        base_ang_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_6do = ObsTerm(
            func=husky_beta_mdp.body_6do, 
            params={"asset_cfg": SceneEntityCfg("robot"), "noise": (-0.1, 0.1)},
        )
        velocity_commands = ObsTerm(func=husky_beta_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=husky_beta_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=husky_beta_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=husky_beta_mdp.isaac.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_6do = ObsTerm(
            func=husky_beta_mdp.body_6do, 
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=husky_beta_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=husky_beta_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )
        joint_vel = ObsTerm(
            func=husky_beta_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )
        actions = ObsTerm(func=husky_beta_mdp.isaac.last_action)

        # privileged observations 
        base_height = ObsTerm(
            func=husky_beta_mdp.isaac.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_wrench = ObsTerm(
            func=husky_beta_mdp.isaac.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )
        joint_efforts = ObsTerm(
            func=husky_beta_mdp.joint_efforts,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class HuskyBetaFlatEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=husky_beta_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=husky_beta_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=husky_beta_mdp.isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=husky_beta_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=husky_beta_mdp.isaac.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-1.0, 1.0),
            },
        },
    )

@configclass
class HuskyBetaFlatRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=husky_beta_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=husky_beta_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=husky_beta_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=husky_beta_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=husky_beta_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "br_foot"), ("fr_foot", "bl_foot")),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    base_height = RewardTermCfg(
        func=husky_beta_mdp.base_height_abs,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.53,
        },
    )
    action_smoothness = RewardTermCfg(func=husky_beta_mdp.action_smoothness_penalty, weight=-1.5)
    air_time_variance = RewardTermCfg(
        func=husky_beta_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=husky_beta_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=husky_beta_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=husky_beta_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    # -- regularization at joint level
    joint_pos_hf = RewardTermCfg(
        func=husky_beta_mdp.joint_position_penalty,
        weight=-1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hx"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_pos = RewardTermCfg(
        func=husky_beta_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hy", ".*_kn"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_vel = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_velocity_penalty,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_acceleration_penalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_torques = RewardTermCfg(
        func=husky_beta_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_power = RewardTermCfg(
        func=husky_beta_mdp.power_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "sharpness": 0.15, "threshold": 60.0},
    )


@configclass
class HuskyBetaFlatDynamicTrottingRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    contact_time = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.contact_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=husky_beta_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={
            "std": 1.0, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    base_linear_velocity = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.base_linear_velocity_reward,
        weight=5.0,
        params={
            "std": 1.0, 
            "ramp_rate": 0.5, 
            "ramp_at_vel": 1.0, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    foot_clearance = RewardTermCfg(
        func=husky_beta_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "air_time": 0.3,
            "contact_time": 0.3,
            "terrain_type": "flat",
            "synced_feet_pair_names": (("fl_foot", "br_foot"), ("fr_foot", "bl_foot")),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=husky_beta_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=husky_beta_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(
        func=husky_beta_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=husky_beta_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=husky_beta_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_acceleration_penalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos_hf = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_position_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hx"),
        },
    )
    joint_pos_hs = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hy"),
        },
    )
    joint_pos_knee = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_kn"),
        },
    )
    joint_torques = RewardTermCfg(
        func=husky_beta_mdp.joint_torques_penalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=husky_beta_mdp.hb_rewards.joint_velocity_penalty,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


@configclass
class HuskyBetaFlatTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=husky_beta_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=husky_beta_mdp.isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
        body_names=["body", ".*_hip", ".*leg", ".*_ankle"]), 
        "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=husky_beta_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )


@configclass
class HuskyBetaFlatCurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class HuskyBetaFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: HuskyBetaFlatObservationsCfg = HuskyBetaFlatObservationsCfg()
    actions: HuskyBetaFlatActionsCfg = HuskyBetaFlatActionsCfg()
    commands: HuskyBetaFlatCommandsCfg = HuskyBetaFlatCommandsCfg()

    # MDP setting
    rewards: HuskyBetaFlatRewardsCfg = HuskyBetaFlatRewardsCfg()
    # rewards: HuskyBetaFlatDynamicTrottingRewardsCfg = HuskyBetaFlatDynamicTrottingRewardsCfg()
    terminations: HuskyBetaFlatTerminationsCfg = HuskyBetaFlatTerminationsCfg()
    events: HuskyBetaFlatEventCfg = HuskyBetaFlatEventCfg()
    curriculum: HuskyBetaFlatCurriculumCfg = HuskyBetaFlatCurriculumCfg()
    # Viewers
    viewer = ViewerCfg(eye=(15.0, -25.0, 5.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0/200.0  # 500 Hz
        self.sim.render_interval = self.decimation

        # switch robot to Huskyb
        self.scene.robot = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

class HuskyBetaFlatEnvCfg_PLAY(HuskyBetaFlatEnvCfg):
    recorders: husky_beta_mdp.RecorderManagerCfg = husky_beta_mdp.RecorderManagerCfg(
        dataset_export_dir_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/source/sslab_extensions/simulation/isaaclab_data/huskyb/flat_env",
        dataset_filename=f"flat_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}",
    )

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
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None