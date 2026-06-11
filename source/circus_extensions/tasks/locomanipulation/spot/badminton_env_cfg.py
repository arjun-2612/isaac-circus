# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg

import circus_extensions.tasks.locomanipulation.spot.mdp as spot_mdp
from circus_extensions.assets.spot import SPOT_BADMINTON_CFG, JOINT_ORDER # isort: skip

@configclass
class SpotBadmintonSceneCfg(InteractiveSceneCfg):
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
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
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # frame transformer 
    spot_badminton_ee = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm0_racket_face",
                name="spot_racket_ee",
                offset=OffsetCfg(pos=[0.15, 0.0, 0.0]),
            ),
        ],
    )


@configclass
class SpotBadmintonCommandsCfg:
    """Command specifications for the MDP."""
    
    shuttle_launcher = spot_mdp.ShuttleLauncherCommandCfg(
        asset_name="robot",
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=spot_mdp.ShuttleLauncherCommandCfg.Ranges(
            px=(6.0, 7.0), py=(-1.0, 1.0), 
            vy=(-2.0, 2.0), vz=(4.0, 8.0),
        ),
        standing_env_ratio=0.1,
        time_between_targets=2.0,
        intercept_height=1.5,
        num_targets=6,
        ee_frame=SceneEntityCfg("spot_badminton_ee"),
        success_threshold=0.3,
    )

@configclass
class SpotBadmintonActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = spot_mdp.isaac.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=JOINT_ORDER, 
        scale=0.25, 
        use_default_offset=True, 
        preserve_order=True
    )

@configclass
class SpotBadmintonObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Body state
        base_lin_vel = ObsTerm(
            func=spot_mdp.isaac.base_lin_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=spot_mdp.isaac.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_6do = ObsTerm(
            func=spot_mdp.body_6do,
            params={"asset_cfg": SceneEntityCfg("robot"), "noise": (-0.1, 0.1)},
        )

        # Joint state
        joint_pos = ObsTerm(
            func=spot_mdp.isaac.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=spot_mdp.isaac.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=spot_mdp.isaac.last_action)

        # End effector state
        ee_position = ObsTerm(
            func=spot_mdp.end_effector_position,
            params={"asset_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("spot_badminton_ee")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        ee_velocity = ObsTerm(
            func=spot_mdp.end_effector_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        ee_orientation = ObsTerm(
            func=spot_mdp.end_effector_orientation,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face")},
        )

        # Swing target command (position relative to base, time, velocity)
        swing_target_pos = ObsTerm(
            func=spot_mdp.swing_target_position,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "shuttle_launcher"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        time_to_swing = ObsTerm(
            func=spot_mdp.time_to_swing, 
            params={"command_name": "shuttle_launcher"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # Body state
        base_lin_vel = ObsTerm(
            func=spot_mdp.isaac.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=spot_mdp.isaac.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_6do = ObsTerm(
            func=spot_mdp.body_6do,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Joint state
        joint_pos = ObsTerm(
            func=spot_mdp.isaac.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=spot_mdp.isaac.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=spot_mdp.isaac.last_action)

        # End effector state
        ee_position = ObsTerm(
            func=spot_mdp.end_effector_position,
            params={"asset_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("spot_badminton_ee")},
        )
        ee_velocity = ObsTerm(
            func=spot_mdp.end_effector_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face")},
        )
        ee_orientation = ObsTerm(
            func=spot_mdp.end_effector_orientation,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face")},
        )

        # Swing target command (position relative to base, time, velocity)
        swing_target_pos = ObsTerm(
            func=spot_mdp.swing_target_position,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "shuttle_launcher"},
        )
        time_to_swing = ObsTerm(
            func=spot_mdp.time_to_swing, 
            params={"command_name": "shuttle_launcher"},
        )

        # Privileged information
        shuttle_pos = ObsTerm(
            func=spot_mdp.shuttlecock_position_gt,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "shuttle_launcher"},
        )
        shuttle_vel = ObsTerm(
            func=spot_mdp.shuttlecock_velocity_gt,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "shuttle_launcher"},
        )
        targets_remaining = ObsTerm(
            func=spot_mdp.targets_remaining, params={"command_name": "shuttle_launcher"})
        next_target_dist = ObsTerm(
            func=spot_mdp.next_target_dist, params={"command_name": "shuttle_launcher"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class SpotBadmintonEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=spot_mdp.isaac.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=spot_mdp.isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=spot_mdp.isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-0.5235, 0.5235)},
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

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=spot_mdp.isaac.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-1.0, 1.0),
            },
        },
    )

@configclass
class SpotBadmintonRewardsCfg:
    # -- task: approach and orientation
    ee_approach = RewardTermCfg(
        func=spot_mdp.ee_approach_reward,
        weight=2.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("spot_badminton_ee"),
            "std": 0.5, 
        },
    )
    ee_orientation = RewardTermCfg(
        func=spot_mdp.ee_orientation_tracking_reward,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face"),
            "std": 0.3,
            "half_width": 0.25, 
        },
    )

    # -- task: swing and follow through
    ee_velocity = RewardTermCfg(
        func=spot_mdp.ee_velocity_tracking_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face"),
            "swing_speed": 12.0, 
            "half_width": 0.25,
            "perp_std": 3.0,
        },
    )
    ee_follow_through = RewardTermCfg(
        func=spot_mdp.ee_follow_through_reward,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_racket_face"),
            "swing_speed": 12.0, 
            "window": 0.2,
            "perp_std": 3.0,
        },
    )

    # -- auxiliary
    face_net = RewardTermCfg(
        func=spot_mdp.face_net_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stand_still_no_target = RewardTermCfg(
        func=spot_mdp.stand_still_reward,
        weight=16.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.25,
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_height = RewardTermCfg(
        func=spot_mdp.isaac.base_height_l2, 
        weight=-5.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.525,
        },
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    collision = RewardTermCfg(
        func=spot_mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*leg"), "threshold": 1.0},
    )

    # joint level regularizations
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_dev_from_default,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="[fh][lr]_.*"),
        },
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.isaac.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    

@configclass
class SpotBadmintonTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=spot_mdp.isaac.time_out, time_out=True)
    tipped_over = DoneTerm(
        func=spot_mdp.fallen_over,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 1.2,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=spot_mdp.isaac.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
        time_out=True,
    )
    # missed = DoneTerm(
    #     func=spot_mdp.missed_shuttle,
    #     params={
    #         "command_name": "shuttle_launcher",
    #         "ee_frame_cfg": SceneEntityCfg("spot_badminton_ee"),
    #         "success_threshold": 0.3,      
    #     },
    # )


@configclass
class SpotBadmintonCurriculumCfg:
    """Curriculum terms for the MDP."""
    
    badminton_intercept_difficulty = CurrTerm(
        func=spot_mdp.shuttle_distance_curriculum,
        params={
            "command_name": "shuttle_launcher",
            "increase_difficulty_threshold": 0.8,
            "decrease_difficulty_threshold": 0.3,
            "level_step": 0.1,
        }
    )


@configclass
class SpotBadmintonEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Spot Badminton environment."""

    # Scene settings
    scene: SpotBadmintonSceneCfg = SpotBadmintonSceneCfg(num_envs=4096, env_spacing=8.0)
    # Basic settings
    observations: SpotBadmintonObservationsCfg = SpotBadmintonObservationsCfg()
    actions: SpotBadmintonActionsCfg = SpotBadmintonActionsCfg()
    commands: SpotBadmintonCommandsCfg = SpotBadmintonCommandsCfg()

    # MDP setting
    rewards: SpotBadmintonRewardsCfg = SpotBadmintonRewardsCfg()
    terminations: SpotBadmintonTerminationsCfg = SpotBadmintonTerminationsCfg()
    events: SpotBadmintonEventCfg = SpotBadmintonEventCfg()
    curriculum: SpotBadmintonCurriculumCfg = SpotBadmintonCurriculumCfg()

    viewer = ViewerCfg(eye=(6.0, 6.0, 1.5), lookat=(0.0, 0.0, 0.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 5  # 100 Hz
        self.episode_length_s = 12.0

        # simulation settings
        self.sim.dt = 1.0/500.0

        # switch robot to Spot-d
        self.scene.robot = SPOT_BADMINTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


class SpotBadmintonEnvCfg_PLAY(SpotBadmintonEnvCfg):
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