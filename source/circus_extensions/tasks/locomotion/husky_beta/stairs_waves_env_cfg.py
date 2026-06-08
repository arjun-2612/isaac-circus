# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os 
import subprocess 
from datetime import datetime 

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import RayCasterCfg, patterns

import circus_extensions.tasks.locomotion.husky_beta.mdp as husky_beta_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from circus_extensions.assets.husky_beta import HUSKY_B_CFG  # isort: skip

STAIRS_WAVES_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=12,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
    },
)

@configclass
class HuskyBetaSWActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = husky_beta_mdp.isaac.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_h[fs]_joint", ".*_k_joint"], 
        scale=0.25, 
        use_default_offset=True, 
        preserve_order=True
    )

@configclass
class HuskyBetaSWCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = husky_beta_mdp.isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=husky_beta_mdp.isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )

@configclass
class HuskyBetaSWObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=husky_beta_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=husky_beta_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=husky_beta_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        joint_vel = ObsTerm(
            func=husky_beta_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        height_scan = ObsTerm(
            func=husky_beta_mdp.isaac.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05)
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
            func=husky_beta_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=husky_beta_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=husky_beta_mdp.isaac.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=husky_beta_mdp.isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=husky_beta_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        joint_vel = ObsTerm(
            func=husky_beta_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        height_scan = ObsTerm(
            func=husky_beta_mdp.isaac.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        actions = ObsTerm(func=husky_beta_mdp.isaac.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HuskyBetaSWEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=husky_beta_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=husky_beta_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=husky_beta_mdp.isaac.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

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
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class HuskyBetaSWRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=husky_beta_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Foot_.*"),
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
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=husky_beta_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="Foot_.*"),
        },
    )
    gait = RewardTermCfg(
        func=husky_beta_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("Foot_fl", "Foot_hr"), ("Foot_fr", "Foot_hl")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=husky_beta_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=husky_beta_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Foot_.*")},
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Foot_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Foot_.*"),
            "threshold": 1.0,
        },
    )
    hip_penalty = RewardTermCfg(
        func=husky_beta_mdp.joint_position_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[fs]_joint"),
            "stand_still_scale": 15.0,
            "velocity_threshold": 0.5,
        },
    )
    knee_penalty = RewardTermCfg(
        func=husky_beta_mdp.joint_position_penalty,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_k_joint"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=husky_beta_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    joint_power = RewardTermCfg(
        func=husky_beta_mdp.power_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"), "sharpness": 0.15, "threshold": 60.0},
    )
    

@configclass
class HuskyBetaSWTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=husky_beta_mdp.isaac.time_out, time_out=True)
    body_contact = DoneTerm(
        func=husky_beta_mdp.isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", 
        body_names=["base_link", "Hip_.*", "Upper_leg_.*", "Lower_leg_.*"]), 
        "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=husky_beta_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )


@configclass
class HuskyBetaSWCurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=husky_beta_mdp.isaac.terrain_levels_vel)
    # pass


@configclass
class HuskyBetaStairsWavesEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings'
    observations: HuskyBetaSWObservationsCfg = HuskyBetaSWObservationsCfg()
    actions: HuskyBetaSWActionsCfg = HuskyBetaSWActionsCfg()
    commands: HuskyBetaSWCommandsCfg = HuskyBetaSWCommandsCfg()

    # MDP setting
    rewards: HuskyBetaSWRewardsCfg = HuskyBetaSWRewardsCfg()
    terminations: HuskyBetaSWTerminationsCfg = HuskyBetaSWTerminationsCfg()
    events: HuskyBetaSWEventCfg = HuskyBetaSWEventCfg()
    curriculum: HuskyBetaSWCurriculumCfg = HuskyBetaSWCurriculumCfg()

    # Viewers
    viewer = ViewerCfg(eye=(-70.0, 0.0, 10.0), origin_type="world", env_index=0, asset_name="robot")
    # viewer = ViewerCfg(eye=(35.0, -10.0, 25.0), origin_type="world", env_index=0, asset_name="robot")
    # viewer = ViewerCfg(eye=(35.0, -10.0, 15.0), origin_type="world", env_index=0, asset_name="robot")

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

        # switch robot to HuskyBeta-d
        self.scene.robot = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


class HuskyBetaStairsWavesEnvCfg_PLAY(HuskyBetaStairsWavesEnvCfg):
    recorders: husky_beta_mdp.RecorderManagerCfg = husky_beta_mdp.RecorderManagerCfg(
        dataset_export_dir_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/source/sslab_extensions/simulation/isaaclab_data/huskyb/stairs_waves_env",
        dataset_filename=f"stairs_waves_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}",
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
            self.scene.terrain.terrain_generator.curriculum = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None