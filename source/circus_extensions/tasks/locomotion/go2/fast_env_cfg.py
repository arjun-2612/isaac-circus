# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import circus_extensions.tasks.locomotion.go2.mdp as go2_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.sensors import ImuCfg

##
# Pre-defined configs
##
from circus_extensions.assets.unitree import UNITREE_GO2_CFG  # isort: skip

FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=12,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25),
    },
)

@configclass
class UnitreeGo2FastActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = go2_mdp.isaac.JointPositionActionCfg(
        asset_name="robot", joint_names=[
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", 
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ], scale=0.25, use_default_offset=True, preserve_order=True
    )


@configclass
class UnitreeGo2FastCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = go2_mdp.ProgressiveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=go2_mdp.ProgressiveVelocityCommandCfg.Ranges(
            lin_vel_x=(-4.0, 6.0), lin_vel_y=(-0.0, 0.0), heading=(-3.14, 3.14), ang_vel_z=(-1.0, 1.0)
        ),
        use_curriculum=True,
    )


@configclass
class UnitreeGo2FastObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=go2_mdp.isaac.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.15, n_max=0.15),
            history_length=5,
        )
        base_orientation = ObsTerm(
            func=go2_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
        )
        velocity_commands = ObsTerm(func=go2_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=go2_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5,
        )
        joint_vel = ObsTerm(
            func=go2_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=go2_mdp.isaac.last_action, history_length=0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=go2_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_orientation = ObsTerm(
            func=go2_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=go2_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=go2_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=go2_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=go2_mdp.isaac.last_action, history_length=0)

        base_lin_vel = ObsTerm(
            func=go2_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_height = ObsTerm(
            func=go2_mdp.isaac.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class UnitreeGo2FastEventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=go2_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
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
    reset_base = EventTerm(
        func=go2_mdp.isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=go2_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (-1.5, 1.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=go2_mdp.isaac.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class UnitreeGo2FastRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=go2_mdp.air_time_reward,
        weight=1.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=go2_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=go2_mdp.base_linear_velocity_reward,
        weight=7.5,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    gait = RewardTermCfg(
        func=go2_mdp.GaitReward,
        weight=1.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=go2_mdp.action_smoothness_penalty, weight=-1.0)
    base_motion = RewardTermCfg(
        func=go2_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=go2_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=go2_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_pos = RewardTermCfg(
        func=go2_mdp.joint_position_penalty,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=go2_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_power = RewardTermCfg(
        func=go2_mdp.power_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "sharpness": 0.2, "threshold": 75.0},
    )


@configclass
class UnitreeGo2FastTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=go2_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=go2_mdp.isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=go2_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )


@configclass
class UnitreeGo2FastCurriculumsCfg:
    """Curriculum specifications for the MDP."""

    cmd_updater = CurrTerm(
        func=go2_mdp.update_cmd,
        params={"upper_threshold": 0.25, "lower_threshold": 0.15, "delta": 0.1},
    )


@configclass
class UnitreeGo2FastEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: UnitreeGo2FastObservationsCfg = UnitreeGo2FastObservationsCfg()
    actions: UnitreeGo2FastActionsCfg = UnitreeGo2FastActionsCfg()
    commands: UnitreeGo2FastCommandsCfg = UnitreeGo2FastCommandsCfg()

    # MDP setting
    rewards: UnitreeGo2FastRewardsCfg = UnitreeGo2FastRewardsCfg()
    terminations: UnitreeGo2FastTerminationsCfg = UnitreeGo2FastTerminationsCfg()
    events: UnitreeGo2FastEventCfg = UnitreeGo2FastEventCfg()
    curriculum: UnitreeGo2FastCurriculumsCfg = UnitreeGo2FastCurriculumsCfg()

    viewer = ViewerCfg(eye=(15.0, -25.0, 5.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 1.0/200.0
        self.decimation = 4

        # switch robot to UnitreeGo2
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.height_scanner = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=FLAT_TERRAIN_CFG,
            max_init_terrain_level=FLAT_TERRAIN_CFG.num_rows - 1,
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
            debug_vis=True,
        )
        
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.005,
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

class UnitreeGo2FastEnvCfg_PLAY(UnitreeGo2FastEnvCfg):
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
        self.commands.base_velocity.use_curriculum = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None