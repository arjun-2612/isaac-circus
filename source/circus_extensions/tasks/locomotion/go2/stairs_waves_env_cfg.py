# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
import isaaclab.terrains as isaac_terrains
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sensors import RayCasterCfg, patterns

from circus_extensions.simulation.terrains import TerrainImporterCfg
import circus_extensions.tasks.locomotion.go2.mdp as go2_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.unitree import UNITREE_GO2_CFG  # isort: skip

STAIRS_WAVES_TERRAIN_CFG = isaac_terrains.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=14,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": isaac_terrains.MeshPlaneTerrainCfg(),
        "stairs_v1": isaac_terrains.HfPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.45,
            platform_width=3.0,
        ),
        "stairs_v2": isaac_terrains.HfPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.35,
            platform_width=3.0,
        ),
        "stairs_v3": isaac_terrains.HfPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.25,
            platform_width=3.0,
        ),
        "inverted_stairs_v1": isaac_terrains.HfInvertedPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.45,
            platform_width=3.0,
        ),
        "inverted_stairs_v2": isaac_terrains.HfInvertedPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.35,
            platform_width=3.0,
        ),
        "inverted_stairs_v3": isaac_terrains.HfInvertedPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.02, 0.25),
            step_width=0.25,
            platform_width=3.0,
        ),
        "platforms_v1": isaac_terrains.HfDiscreteObstaclesTerrainCfg(
            obstacle_width_range=(2.5, 3.0),
            obstacle_height_range=(0.0, 0.25),
            num_obstacles=5,
        ),
        "platforms_v2": isaac_terrains.HfDiscreteObstaclesTerrainCfg(
            obstacle_width_range=(2.5, 3.0),
            obstacle_height_range=(0.0, 0.25),
            num_obstacles=3,
        ),
        "platforms_v3": isaac_terrains.HfDiscreteObstaclesTerrainCfg(
            obstacle_width_range=(2.5, 3.0),
            obstacle_height_range=(0.0, 0.25),
            num_obstacles=2,
        ),
    },
)

TEST_TERRAIN_CFG = isaac_terrains.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "stairs": isaac_terrains.HfPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.15, 0.25),
            step_width=0.35,
            platform_width=3.0,
        ),
        "inverted_stairs": isaac_terrains.HfInvertedPyramidStairsTerrainCfg(
            border_width=1.0,
            step_height_range=(0.15, 0.25),
            step_width=0.35,
            platform_width=3.0,
        ),
        "platforms": isaac_terrains.HfDiscreteObstaclesTerrainCfg(
            obstacle_width_range=(2.5, 3.0),
            obstacle_height_range=(0.15, 0.3),
            num_obstacles=5,
        ),
    },
)

@configclass
class UnitreeGo2SWActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = go2_mdp.isaac.JointPositionActionCfg(
        asset_name="robot", joint_names=[
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", 
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", 
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ], scale=0.25, use_default_offset=True, preserve_order=True
    )

@configclass
class UnitreeGo2SWCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = go2_mdp.isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=go2_mdp.isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.5, 1.5), heading=(-math.pi, math.pi),
        ),
    )

@configclass
class UnitreeGo2SWObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=go2_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=go2_mdp.isaac.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        base_orientation = ObsTerm(
            func=go2_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=go2_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=go2_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=go2_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=go2_mdp.isaac.last_action)
        height_scan = ObsTerm(
            func=go2_mdp.isaac.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            noise=Gnoise(mean=0.0, std=0.007),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=go2_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=go2_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, history_length=5,
        )
        base_orientation = ObsTerm(
            func=go2_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            history_length=5,
        )
        velocity_commands = ObsTerm(func=go2_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=go2_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, history_length=5,
        )
        joint_vel = ObsTerm(
            func=go2_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=go2_mdp.isaac.last_action)
        height_scan = ObsTerm(
            func=go2_mdp.isaac.height_scan,  # set the dim equal to raycaster sensor rays manually
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )

        # privileged observations
        base_height = ObsTerm(
            func=go2_mdp.isaac.base_pos_z, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_efforts = ObsTerm(
            func=go2_mdp.joint_efforts,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        contact_forces = ObsTerm(
            func=go2_mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2SWEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=go2_mdp.isaac.randomize_rigid_body_material,
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
        func=go2_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=go2_mdp.isaac.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

@configclass
class UnitreeGo2SWRewardsCfg:
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
        weight=5.0,
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
    collision = RewardTermCfg(
        func=go2_mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},
    )
    foot_trip = RewardTermCfg(
        func=go2_mdp.foot_trip_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
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
    foot_near_edge = RewardTermCfg(
        func=go2_mdp.foot_near_edge_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    action_smoothness = RewardTermCfg(func=go2_mdp.action_smoothness_penalty, weight=-1.0)
    base_orientation = RewardTermCfg(
        func=go2_mdp.base_orientation_penalty, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_rp_motion = RewardTermCfg(
        func=go2_mdp.base_rp_motion_penalty, weight=-0.025, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_z_motion = RewardTermCfg(
        func=go2_mdp.base_z_motion_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    hxy_penalty = RewardTermCfg(
        func=go2_mdp.joint_position_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"]),
            "stand_still_scale": 15.0,
            "velocity_threshold": 0.5,
        },
    )
    kn_penalty = RewardTermCfg(
        func=go2_mdp.joint_position_penalty,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "sharpness": 0.2, "threshold": 100.0},
    )
    

@configclass
class UnitreeGo2SWTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=go2_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=go2_mdp.isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=go2_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
        time_out=True,
    )


@configclass
class UnitreeGo2SWCurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=go2_mdp.isaac.terrain_levels_vel)


@configclass
class UnitreeGo2StairsWavesEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: UnitreeGo2SWObservationsCfg = UnitreeGo2SWObservationsCfg()
    actions: UnitreeGo2SWActionsCfg = UnitreeGo2SWActionsCfg()
    commands: UnitreeGo2SWCommandsCfg = UnitreeGo2SWCommandsCfg()

    # MDP setting
    rewards: UnitreeGo2SWRewardsCfg = UnitreeGo2SWRewardsCfg()
    terminations: UnitreeGo2SWTerminationsCfg = UnitreeGo2SWTerminationsCfg()
    events: UnitreeGo2SWEventCfg = UnitreeGo2SWEventCfg()
    curriculum: UnitreeGo2SWCurriculumCfg = UnitreeGo2SWCurriculumCfg()

    # Viewers
    viewer = ViewerCfg(eye=(-70.0, 0.0, 10.0), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0/200.0  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to UnitreeGo2-d
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=STAIRS_WAVES_TERRAIN_CFG,
            max_init_terrain_level=3,
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

        # no height scan
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 0.8]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


class UnitreeGo2StairsWavesEnvCfg_PLAY(UnitreeGo2StairsWavesEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TEST_TERRAIN_CFG,
            max_init_terrain_level=3,
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

        # disable randomization for play
        self.observations.policy.enable_corruption = True