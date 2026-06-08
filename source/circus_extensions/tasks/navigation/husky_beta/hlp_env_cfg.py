# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
import subprocess
import os
import math

# IsaacLab Imports
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Custom SS Lab Imports
from circus_extensions.simulation.envs import ManagerBasedRLEnvCfg
from circus_extensions.simulation.terrains import TerrainImporterCfg
import sslab_extensions.tasks.navigation.husky_beta.mdp as huskyb_mdp
from circus_extensions.assets.husky_beta import HUSKY_B_CFG # isort: skip
from circus_extensions.tasks.locomotion.husky_beta.dtc_env_cfg import HuskyBetaDTCEnvCfg # isort: skip
from circus_extensions.tasks.locomotion.husky_beta.flight_env_cfg import HuskyBetaFlightEnvCfg

# Constants
root_path = f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}"
walk_policy_folder = "2026-03-20_12-52-39"

flat_patch_sampling_dict = {
    "init_pos": FlatPatchSamplingCfg(num_patches=8, patch_radius=1.28, max_height_diff=0.1),
    "target": FlatPatchSamplingCfg(num_patches=16, patch_radius=0.35, max_height_diff=0.1),
}

HLP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, 
            noise_range=(0.0, 0.0), 
            noise_step=0.01, 
            border_width=0.25, 
            flat_patch_sampling=flat_patch_sampling_dict,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.9, 
            obstacle_width_range=(2.0, 3.0), 
            obstacle_height_range=(0.25, 1.75), 
            num_obstacles=3, 
            platform_width=2.0, 
            flat_patch_sampling=flat_patch_sampling_dict,
        ),
    },
)

PLAY_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0, 
            obstacle_width_range=(2.0, 3.0), 
            obstacle_height_range=(0.25, 1.75), 
            num_obstacles=4, 
            platform_width=2.0, 
            flat_patch_sampling=flat_patch_sampling_dict,
        ),
    },
)

WALK_ENV_CFG = HuskyBetaDTCEnvCfg()
FLIGHT_ENV_CFG = HuskyBetaFlightEnvCfg()


@configclass
class HuskyBetaHLPSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HLP_TERRAIN_CFG,
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
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = HUSKY_B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class HuskyBetaHLPCommandsCfg:
    """Command specifications for the MDP."""

    pose_command = huskyb_mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        required_settled_steps=15,
        max_goal_speed=0.25,
        pos_success_threshold=0.15,
        yaw_success_threshold=0.5,
        debug_vis=True,
        ranges=huskyb_mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi),
        ),
    )

    multimodal_command = huskyb_mdp.HuskyBetaMultimodalJointPosCommandCfg(
        asset_name="robot",
        debug_vis=True,
        resampling_time_range=(10.0, 10.0),
        action_name="multimodal_action",
    )


@configclass
class HuskyBetaHLPActionsCfg:
    """Action specifications for the MDP."""

    multimodal_action = huskyb_mdp.HuskyBetaMultimodalJointPosActionCfg(
        asset_name="robot",
        scale=0.5,
        walk_policy_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/source/sslab_extensions/simulation/isaacsim_scripts/huskyb/working_policies/dtc/{walk_policy_folder}/policy.pt",
        walk_policy_actions=WALK_ENV_CFG.actions.joint_pos,
        walk_policy_observations=WALK_ENV_CFG.observations.policy,
        low_level_decimation=WALK_ENV_CFG.decimation,
        flight_action=FLIGHT_ENV_CFG.actions.base_forces,
    )


@configclass
class HuskyBetaHLPObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot state
        base_lin_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            clip=(-5.0, 5.0),
        )
        base_ang_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-7.0, 7.0),
        )
        base_6do = ObsTerm(
            func=huskyb_mdp.body_6do,
            params={"asset_cfg": SceneEntityCfg("robot"), "noise": (-0.1, 0.1)},
        )

        # -- commands
        pose_command = ObsTerm(
            func=huskyb_mdp.isaac.generated_commands, 
            params={"command_name": "pose_command"},
        )
        commanded_velocity = ObsTerm( # maybe split into lin and ang vel?
            func=huskyb_mdp.commanded_velocity,
            params={"command_name": "multimodal_command"},
        )

        # -- mode
        observed_mode = ObsTerm(
            func=huskyb_mdp.observed_mode,
            params={"command_name": "multimodal_command"},
        )

        # -- terrain
        terrain_representation = ObsTerm(
            func=huskyb_mdp.isaac.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot state
        base_lin_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
        )
        base_ang_vel = ObsTerm(
            func=huskyb_mdp.isaac.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-7.0, 7.0),
        )
        base_6do = ObsTerm(
            func=huskyb_mdp.body_6do,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # -- commands
        pose_command = ObsTerm(
            func=huskyb_mdp.isaac.generated_commands, 
            params={"command_name": "pose_command"},
        )
        commanded_velocity = ObsTerm(
            func=huskyb_mdp.commanded_velocity,
            params={"command_name": "multimodal_command"},
        )

        # -- mode
        observed_mode = ObsTerm(
            func=huskyb_mdp.observed_mode,
            params={"command_name": "multimodal_command"},
        )

        # -- terrain
        terrain_representation = ObsTerm(
            func=huskyb_mdp.isaac.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-5.0, 5.0),
        )

        # -- privileged observations 
        base_pos = ObsTerm(
            func=huskyb_mdp.pos_rel_env_origin, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        time_left = ObsTerm(
            func=huskyb_mdp.time_left_in_episode,
            params={"command_name": "multimodal_command"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HuskyBetaHLPEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=huskyb_mdp.isaac.randomize_rigid_body_material,
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
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    randomize_flight_spawn = EventTerm(
        func=huskyb_mdp.randomize_flight_spawn,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_name": "multimodal_action",
            "flight_spawn_prob": 0.25,
            "flight_spawn_height": 0.6,
        },
    )


@configclass
class HuskyBetaHLPRewardsCfg:
    # -- task rewards
    position_tracking = RewardTermCfg(
        func=huskyb_mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine = RewardTermCfg(
        func=huskyb_mdp.position_command_error_tanh,
        weight=5.0,
        params={"std": 0.5, "command_name": "pose_command"},
    )
    orientation_tracking = RewardTermCfg(
        func=huskyb_mdp.heading_command_error_abs,
        weight=-2.0,
        params={"command_name": "pose_command"},
    )
    goal_reached = RewardTermCfg(
        func=huskyb_mdp.goal_reached_bonus,
        weight=10.0,
        params={"command_name": "pose_command"},
    )

    # -- mode related
    mode_persistence = RewardTermCfg(
        func=huskyb_mdp.mode_persistence_penalty,
        weight=-0.5,
        params={"command_name": "multimodal_command"},
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=huskyb_mdp.action_smoothness_penalty_hlp, weight=-1.0)
    flight_mode_cost = RewardTermCfg(
        func=huskyb_mdp.flight_mode_cost,
        weight=-0.25,
        params={
            "command_name": "multimodal_command",
        },
    )
    # foot_near_edge = RewardTermCfg(
    #     func=huskyb_mdp.foot_near_edge_penalty,
    #     weight=-5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )
    # foot_trip = RewardTermCfg(
    #     func=huskyb_mdp.foot_trip_penalty,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )
    undesired_body_contact = RewardTermCfg(
        func=huskyb_mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_lleg", ".*_ankle"]), 
            "threshold": 1.0,
        },
    )
    termination_penalty = RewardTermCfg(
        func=huskyb_mdp.isaac.is_terminated,
        weight=-200.0,
    )


@configclass
class HuskyBetaHLPTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=huskyb_mdp.isaac.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=huskyb_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )
    goal_reached = DoneTerm(
        func=huskyb_mdp.goal_reached, 
        params={"command_name": "pose_command"},
        time_out=True,
    )

    body_contact = DoneTerm(
        func=huskyb_mdp.isaac.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["body", ".*_hip", ".*_uleg"],
            ), 
            "threshold": 1.0,
        },
    )
    damaging_contacts = DoneTerm(
        func=huskyb_mdp.isaac.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold": 300.0,
        },
    )


@configclass
class HuskyBetaHLPCurriculumCfg:
    """Curriculum terms for the MDP."""
    
    terrain_levels = CurrTerm(
        func=huskyb_mdp.terrain_levels_pose,
        params={
            "move_up_distance": 0.3,
            "move_down_distance": 1.0,
        }
    )


@configclass
class HuskyBetaHLPEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for Husky-b HLP task."""

    # Scene settings'
    scene: HuskyBetaHLPSceneCfg = HuskyBetaHLPSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings'
    observations: HuskyBetaHLPObservationsCfg = HuskyBetaHLPObservationsCfg()
    actions: HuskyBetaHLPActionsCfg = HuskyBetaHLPActionsCfg()
    commands: HuskyBetaHLPCommandsCfg = HuskyBetaHLPCommandsCfg()

    # MDP setting
    rewards: HuskyBetaHLPRewardsCfg = HuskyBetaHLPRewardsCfg()
    terminations: HuskyBetaHLPTerminationsCfg = HuskyBetaHLPTerminationsCfg()
    events: HuskyBetaHLPEventCfg = HuskyBetaHLPEventCfg()
    curriculum: HuskyBetaHLPCurriculumCfg = HuskyBetaHLPCurriculumCfg()

    viewer = ViewerCfg(eye=(15.0, -15.0, 3.0), lookat=(0.0, 0.0, 0.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = WALK_ENV_CFG.decimation * 10  # 5 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0/200.0  # 200 Hz
        self.sim.render_interval = WALK_ENV_CFG.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                WALK_ENV_CFG.decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class HuskyBetaHLPEnvCfg_PLAY(HuskyBetaHLPEnvCfg):
    # recorders: huskyb_mdp.RecorderManagerCfg = huskyb_mdp.RecorderManagerCfg(
    #     dataset_export_dir_path=f"{'/workspace' if os.path.exists('/workspace') else subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()}/source/sslab_extensions/simulation/isaaclab_data/huskyb/hlp_env",
    #     dataset_filename=f"hlp_{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}",
    # )

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # disable randomization for play
        self.observations.policy.enable_corruption = True
        self.events.push_robot = None
        self.curriculum.terrain_levels = None
        self.episode_length_s = 1000.0
        self.events.randomize_flight_spawn = None
        self.terminations.body_contact = None 
        self.terminations.time_out = None 
        self.terminations.terrain_out_of_bounds = None

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