# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import circus_extensions.tasks.locomotion.go2.mdp as go2_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.unitree import UNITREE_GO2_CFG  # isort: skip

@configclass
class UnitreeGo2MMCommandsCfg:
    """Command specifications for the MDP."""

    motion_references = go2_mdp.MotionDatasetSamplerCfg(
        robot_asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        # data_dir="/home/arjun/Documents/sslab_isaaclab/source/sslab_extensions/simulation/blender/go2/go2_backflip.npz",
        data_dir="/workspace/source/sslab_extensions/simulation/blender/go2/go2_backflip.npz",
        score_std=10.0,
        max_extra_time=0.0,
        task_phase_min=0.0,
        task_phase_max=1.0,
    )
    wrench_assist = go2_mdp.WrenchAssistCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000, 1000),
        wrench_assist_body_name="base",
        max_wrench_scale=0.75,
        force_gain=(0.0, 10.0),
        torque_gain=(0.0, 10.0),
        debug_vis=True,
    )


@configclass
class UnitreeGo2MMActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = go2_mdp.ReferencePositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", 
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ],
        scale=0.25,
        reference_name="motion_references",
        preserve_order=True,
    )


@configclass
class UnitreeGo2MMObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # proprioception
        base_ang_vel = ObsTerm(
            func=go2_mdp.isaac.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        projected_gravity = ObsTerm(
            func=go2_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=go2_mdp.joint_pos_reference_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"), 
                "reference_name": "motion_references",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=go2_mdp.isaac.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=go2_mdp.isaac.last_action)

        # references
        base_pos_z = ObsTerm(
            func=go2_mdp.base_pos_z_reference, params={"reference_name": "motion_references"}
        )
        projected_gravity_ref = ObsTerm(
            func=go2_mdp.projected_gravity_reference,
            params={"asset_cfg": SceneEntityCfg("robot"), "reference_name": "motion_references"},
        )
        base_vel_ref = ObsTerm(
            func=go2_mdp.base_vel_reference, params={"reference_name": "motion_references"}
        )
        joint_pos_ref = ObsTerm(
            func=go2_mdp.joint_pos_reference,
            params={
                "asset_cfg": SceneEntityCfg("robot"), 
                "reference_name": "motion_references",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for policy group."""

        # observation terms (order preserved)
        projected_gravity_b = ObsTerm(func=go2_mdp.isaac.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel_b = ObsTerm(func=go2_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel_b = ObsTerm(func=go2_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_z = ObsTerm(func=go2_mdp.isaac.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")})
        body_wrench = ObsTerm(
            func=go2_mdp.isaac.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        foot_pos_wrt_body = ObsTerm(
            func=go2_mdp.body_relative_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        foot_vel_wrt_body = ObsTerm(
            func=go2_mdp.body_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        wrench_force = ObsTerm(
            func=go2_mdp.wrench_force,
            params={"command_name": "wrench_assist"},
        )
        wrench_torque = ObsTerm(
            func=go2_mdp.wrench_torque,
            params={"command_name": "wrench_assist"},
        )
        wrench_scale = ObsTerm(
            func=go2_mdp.wrench_scale,
            params={"command_name": "wrench_assist"},
        )
        similarity_metric = ObsTerm(
            func=go2_mdp.similarity_metric,
            params={"reference_name": "motion_references"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2MMEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=go2_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=go2_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # reset
    reset_task_phase = EventTerm(
        func=go2_mdp.reset_task_phase,
        mode="reset",
        params={
            "reference_name": "motion_references",
        },
    )

    reset_base = EventTerm(
        func=go2_mdp.reset_root_state_to_reference,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-3, 3), "y": (-3, 3), "z": (0.05, 0.05), "yaw": (-0.4, 0.4)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
            "reference_name": "motion_references",
        },
    )

    reset_robot_joints = EventTerm(
        func=go2_mdp.reset_joint_state_to_reference,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_name": "motion_references",
        },
    )

    # interval
    push_robot = EventTerm(
        func=go2_mdp.push_robot_under_wrench_assist,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "command_name": "wrench_assist",
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


##
# Reward manager.
##


@configclass
class UnitreeGo2MMRewardsCfg:
    # -- task rewards
    total_tracking_reward = RewardTermCfg(
        func=go2_mdp.total_tracking_reward, weight=1.0, params={"score_relative_weight": 5.0}
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=go2_mdp.action_smoothness_penalty, weight=-1.0)
    joint_torques = RewardTermCfg(
        func=go2_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    dof_pos_limits = RewardTermCfg(func=go2_mdp.isaac.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot")})
    applied_torque_limits = RewardTermCfg(
        func=go2_mdp.isaac.applied_torque_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    collision = RewardTermCfg(
        func=go2_mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )

    # -- to discourage early terminations
    bias = RewardTermCfg(
        func=go2_mdp.bias,
        weight=30.0,
        params={},
    )


@configclass
class UnitreeGo2MMTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=go2_mdp.isaac.time_out, time_out=True)

    foot_tangling_termination = DoneTerm(
        func=go2_mdp.bodies_distance_below_threshold,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "min_distance_threshold": 0.06},
        time_out=False,
    )
    feet_ground_penetration = DoneTerm(
        func=go2_mdp.body_ground_penetration,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"), "max_ground_penetration": 0.03},
        time_out=False,
    )
    damaging_contacts = DoneTerm(
        func=go2_mdp.isaac.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold": 12000.0,
        },
    )
    max_body_pos_dev_wrt_base = DoneTerm(
        func=go2_mdp.max_body_pos_dev_wrt_base,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "max_deviation": 0.50,
        },
        time_out=False,
    )
    root_pos_z_max_offset = DoneTerm(
        func=go2_mdp.root_pos_z_max_offset,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_deviation": 0.40},
        time_out=False,
    )
    projected_gravity_max_offset = DoneTerm(
        func=go2_mdp.projected_gravity_max_offset,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_deviation": 0.60},
        time_out=False,
    )
    feet_flight_phase = DoneTerm(
        func=go2_mdp.feet_still_in_flight_at_end_of_reference,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "reference_name": "motion_references",
            "foot_body_name": ".*_foot",
            "contact_force_threshold": 1.0,
            "max_num_steps_in_flight": 10,
        },
        time_out=False,
    )


@configclass
class UnitreeGo2MMCurriculumCfg:
    """Curriculum terms for the MDP."""

    update_wrench_assist_range = CurrTerm(
        func=go2_mdp.hand_of_god,
        params={
            "metric_command_term_name": "motion_references",
            "wrench_assist_command_term_name": "wrench_assist",
            "wrench_assist_decrease": 0.10,
            "wrench_assist_increase": 0.05,
        },
    )


##
# Environment configurations
##


@configclass
class UnitreeGo2MotionMimicEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: UnitreeGo2MMObservationsCfg = UnitreeGo2MMObservationsCfg()
    rewards: UnitreeGo2MMRewardsCfg = UnitreeGo2MMRewardsCfg()
    actions: UnitreeGo2MMActionsCfg = UnitreeGo2MMActionsCfg()
    commands: UnitreeGo2MMCommandsCfg = UnitreeGo2MMCommandsCfg()
    terminations: UnitreeGo2MMTerminationsCfg = UnitreeGo2MMTerminationsCfg()
    events: UnitreeGo2MMEventCfg = UnitreeGo2MMEventCfg()
    curriculum: UnitreeGo2MMCurriculumCfg = UnitreeGo2MMCurriculumCfg()

    viewer = ViewerCfg(eye=(5, -5, 2), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()
        # override sim settings
        self.decimation = 4  # 50 Hz
        self.sim_dt = 1.0/200.0
        self.episode_length_s = 3.25

        # switch robot to UnitreeGo2
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None


@configclass
class UnitreeGo2MotionMimicEnvCfg_PLAY(UnitreeGo2MotionMimicEnvCfg):
    """Make smaller Flat terrain environment for play"""

    def __post_init__(self):
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

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.wrench_assist.max_wrench_scale = 0.0
        self.commands.motion_references.task_phase_max = 0.0

        # self.terminations.projected_gravity_max_offset = None