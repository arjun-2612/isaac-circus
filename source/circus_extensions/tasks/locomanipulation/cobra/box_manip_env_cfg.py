# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from . import mdp
from .mdp.commands import BoxManipulationCommandCfg

##
# Asset paths
##

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUBE_USD = os.path.join(_PKG_DIR, "assets", "usd", "colored_cube.usd")
CUBE_FRAME_USD = os.path.join(_PKG_DIR, "assets", "usd", "colored_cube_frame.usd")

from circus_extensions.assets.cobra import COBRA_CFG  # isort:skip

##
# Sampling configuration
##

CUBE_HALF = 0.1
CUBE_HEIGHT = CUBE_HALF + 0.01

XY_RANGE = ((-1.2, 0.0), (0.0, 0.6))
ROBOT_EXCLUSION = ((-1.6, 0.2), (-0.15, 0.15))
YAW_RANGE = (-180.0, 180.0)
MIN_BOX_GOAL_DIST = 0.3


##
# Scene
##


@configclass
class CobraMasterBoxSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = COBRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.06),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, CUBE_HEIGHT)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    box_goal = BoxManipulationCommandCfg(
        asset_name="object",
        xy_range=XY_RANGE,
        robot_exclusion=ROBOT_EXCLUSION,
        yaw_range=YAW_RANGE,
        min_box_goal_dist=MIN_BOX_GOAL_DIST,
        object_height=CUBE_HEIGHT,
        pos_success_threshold=0.05,
        orientation_success_threshold=0.40,
        max_goal_speed=0.05,
        required_settled_time=0.5,
        goal_visualizer_cfg=sim_utils.UsdFileCfg(
            usd_path=CUBE_FRAME_USD,
            scale=(1.0, 1.0, 1.0),
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.2,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(std=0.05))
        vc_observations = ObsTerm(
            func=mdp.VirtualChassisObservations,
            noise=Gnoise(std=0.005),
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "box_goal",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    track_pose_global = RewTerm(
        func=mdp.box_keypoint_l2,
        weight=-10.0,
        params={"command_name": "box_goal", "half_side": 0.1},
    )
    success_bonus = RewTerm(
        func=mdp.manipulation_success_bonus,
        weight=5.0,
        params={"command_name": "box_goal"},
    )
    time_penalty = RewTerm(func=mdp.time_penalty, weight=-0.5)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.005)
    goal_completion = RewTerm(
        func=mdp.goal_completion_bonus,
        weight=100.0,
        params={"command_name": "box_goal"},
    )
    # box_flip_penalty = RewTerm(
    #     func=mdp.box_upright,
    #     weight=-1.0,
    # )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "box_goal"},
    )
    # box_flipped = DoneTerm(
    #     func=mdp.box_flipped,
    # )


@configclass
class CurriculumCfg:
    goal_threshold = CurrTerm(
        func=mdp.BoxManipGoalThresholdCurriculum,
        params={
            "command_name": "box_goal",
            "initial_pos_threshold": 0.2,
            "final_pos_threshold": 0.05,
            "initial_orientation_threshold": 0.8,
            "final_orientation_threshold": 0.30,
            "success_rate_threshold": 0.7,
            "ramp_rate": 0.01,
            "ema_alpha": 0.05,
        },
    )
    anneal_success_bonus = CurrTerm(
        func=mdp.BoxManipAnnealRewardWeight,
        params={
            "reward_name": "success_bonus",
            "target_weight": 0.0,
            "success_rate_threshold": 0.7,
            "ramp_rate": 0.01,
            "ema_alpha": 0.05,
        },
    )


##
# Full environment config
##


@configclass
class CobraMasterBoxEnvCfg(ManagerBasedRLEnvCfg):

    scene: CobraMasterBoxSceneCfg = CobraMasterBoxSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 3.0, 4.5)
        self.viewer.lookat = (0.0, -1.2, 0.0)


@configclass
class CobraMasterBoxEnvCfg_PLAY(CobraMasterBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
