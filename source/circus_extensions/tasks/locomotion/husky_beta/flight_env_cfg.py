# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import circus_extensions.tasks.locomotion.husky_beta.mdp as huskyb_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.husky_beta import HUSKY_B_CFG  # isort: skip


@configclass
class HuskyBetaFlightActionsCfg:
    """Action specifications for the MDP."""

    base_forces = huskyb_mdp.QuadcopterPDActionCfg(
        asset_name="robot", 
        joint_names=[".*_kn"], 
        leg_joint_names=[".*_hx", ".*_hy", ".*_kn"],
        scale=0.2, 
        robot_mass=6.11,
        kp_lin = [3.0, 3.0, 20.0],
        kd_lin = [6.0, 6.0, 15.0],
        kp_ang = [100.0, 100.0, 100.0],
        kd_ang = [10.0, 10.0, 10.0],
        root_name="body",
        # cmd_vel_name="base_velocity",
    )

@configclass
class HuskyBetaFlightCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = huskyb_mdp.Uniform3DVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.1,
        heading_command=False,
        debug_vis=True,
        ranges=huskyb_mdp.Uniform3DVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), lin_vel_z=(0.0, 0.25), ang_vel_z=(-0.5, 0.5), heading=(-3.14, 3.14)
        ),
    )

@configclass
class HuskyBetaFlightObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_lin_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=huskyb_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=huskyb_mdp.isaac.generated_commands, 
            params={"command_name": "base_velocity"},
        )
        actions = ObsTerm(func=huskyb_mdp.isaac.last_action)
        base_height = ObsTerm(
            func=huskyb_mdp.isaac.base_pos_z, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.015, n_max=0.015),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class HuskyBetaFlightEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=huskyb_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=huskyb_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.1, 0.1),
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
            # "velocity_range": {
            #     "x": (-1.5, 1.5),
            #     "y": (-1.0, 1.0),
            #     "z": (-0.5, 0.5),
            #     "roll": (-0.7, 0.7),
            #     "pitch": (-0.7, 0.7),
            #     "yaw": (-1.0, 1.0),
            # },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=huskyb_mdp.isaac.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    #     },
    # )

@configclass
class HuskyBetaFlightRewardsCfg:
    # -- task
    # living_reward = RewardTermCfg(func=huskyb_mdp.isaac.is_alive, weight=0.1)
    base_angular_velocity = RewardTermCfg(
        func=huskyb_mdp.base_angular_velocity_reward,
        weight=1.0,
        params={
            "std": 0.5, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    base_linear_velocity = RewardTermCfg(
        func=huskyb_mdp.base_linear_velocity_reward,
        weight=1.0,
        params={
            "std": 0.5, 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    # altitude_reward = RewardTermCfg(
    #     func=huskyb_mdp.base_altitude_reward,
    #     weight=3.0,
    #     params={
    #         "z_target": 1.0, 
    #         "std": 0.5, 
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     }
    # )

    # -- penalties
    termination_penalty = RewardTermCfg(func=huskyb_mdp.isaac.is_terminated, weight=-5.0)
    action_smoothness = RewardTermCfg(func=huskyb_mdp.action_smoothness_penalty, weight=-0.5)
    ang_vel_xy_l2 = RewardTermCfg(func=huskyb_mdp.isaac.ang_vel_xy_l2, weight=-2.0)
    base_orientation = RewardTermCfg(
        func=huskyb_mdp.base_orientation_penalty, 
        weight=-2.0, 
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class HuskyBetaFlightTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=huskyb_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=huskyb_mdp.isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
        body_names=["body", ".*leg", ".*_ankle", ".*_foot"]), 
        "threshold": 1.0},
    )


@configclass
class HuskyBetaFlightCurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class HuskyBetaFlightEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: HuskyBetaFlightObservationsCfg = HuskyBetaFlightObservationsCfg()
    actions: HuskyBetaFlightActionsCfg = HuskyBetaFlightActionsCfg()
    commands: HuskyBetaFlightCommandsCfg = HuskyBetaFlightCommandsCfg()

    # MDP setting
    rewards: HuskyBetaFlightRewardsCfg = HuskyBetaFlightRewardsCfg()
    terminations: HuskyBetaFlightTerminationsCfg = HuskyBetaFlightTerminationsCfg()
    events: HuskyBetaFlightEventCfg = HuskyBetaFlightEventCfg()
    curriculum: HuskyBetaFlightCurriculumCfg = HuskyBetaFlightCurriculumCfg()

    # Viewers
    viewer = ViewerCfg(eye=(10.0, -15.0, 2.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 1.0/200.0 # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to HuskyBeta-d
        self.scene.robot = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # no height scan
        self.scene.height_scanner = None


class HuskyBetaFlightEnvCfg_PLAY(HuskyBetaFlightEnvCfg):
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
            self.scene.terrain.terrain_generator.curriculum = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None