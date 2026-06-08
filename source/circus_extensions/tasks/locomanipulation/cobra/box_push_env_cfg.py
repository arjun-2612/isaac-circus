# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.commands import PlanarManipulationCommandCfg

##
# Asset paths
##

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUBE_USD = os.path.join(_PKG_DIR, "assets", "usd", "colored_cube.usd")
CUBE_FRAME_USD = os.path.join(_PKG_DIR, "assets", "usd", "colored_cube_frame.usd")

##
# Pre-defined configs
##

from circus_extensions.assets.cobra import COBRA_CFG  # isort:skip

##
# Spawn ranges (relative to env origin, which coincides with head at reset)
##

OBJ_X_RANGE = (-1.2, -0.5)
OBJ_Y_RANGE = (0.2, 0.3)
OBJ_YAW_RANGE = (-45.0, 45.0)

GOAL_X_RANGE = (-1.2, -0.5)
GOAL_Y_RANGE = (0.5, 0.7)
GOAL_YAW_RANGE = (-45.0, 45.0)

CUBE_HALF = 0.1  # half-extent of 0.2 m cube

# Default init position for the object (centre of spawn range)
_OBJ_INIT_X = 0.5 * (OBJ_X_RANGE[0] + OBJ_X_RANGE[1])
_OBJ_INIT_Y = 0.5 * (OBJ_Y_RANGE[0] + OBJ_Y_RANGE[1])


##
# Scene
##


@configclass
class CobraBoxPushSceneCfg(InteractiveSceneCfg):
    """Scene: ground plane, cobra robot, pushable box, lighting."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = COBRA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.06),  # head collision radius = 0.05
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_OBJ_INIT_X, _OBJ_INIT_Y, CUBE_HALF + 0.01),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Planar (x, y, yaw) goal for the box — sampled once per episode at reset."""

    box_goal = PlanarManipulationCommandCfg(
        asset_name="object",
        goal_x_range=GOAL_X_RANGE,
        goal_y_range=GOAL_Y_RANGE,
        goal_yaw_range=GOAL_YAW_RANGE,
        object_height=CUBE_HALF + 0.01,
        pos_success_threshold=0.05,
        yaw_success_threshold=0.4,
        goal_visualizer_cfg=sim_utils.UsdFileCfg(
            usd_path=CUBE_FRAME_USD,
            scale=(1.0, 1.0, 1.0),
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Joint position targets (offset from default)."""

    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=0.5,
    #     use_default_offset=True,
    # )

    # joint_pos = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=0.5,
    # )

    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.2,
    )

@configclass
class ObservationsCfg:
    """28-D observation: joint state (22) + box pose (3) + goal pose (3)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # box_pose = ObsTerm(
        #     func=mdp.box_pose_env,
        #     params={
        #         "object_cfg": SceneEntityCfg("object"),
        #     },
        # )
        # goal_pose = ObsTerm(
        #     func=mdp.goal_pose_env,
        #     params={
        #         "command_name": "box_goal",
        #     },
        # )
        vc_observations = ObsTerm(
            func=mdp.VirtualChassisObservations,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "box_goal",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: robot to default pose, box to random pose in spawn region."""

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

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (OBJ_X_RANGE[0] - _OBJ_INIT_X, OBJ_X_RANGE[1] - _OBJ_INIT_X),
                "y": (OBJ_Y_RANGE[0] - _OBJ_INIT_Y, OBJ_Y_RANGE[1] - _OBJ_INIT_Y),
                "yaw": (math.radians(OBJ_YAW_RANGE[0]), math.radians(OBJ_YAW_RANGE[1])),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for planar box pushing."""

    # track_pos = RewTerm(
    #     func=mdp.track_pos_l2,
    #     weight=-10.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "command_name": "box_goal",
    #     },
    # )
    # track_yaw = RewTerm(
    #     func=mdp.track_yaw_l1,
    #     weight=-5.0,
    #     params={"command_name": "box_goal"},
    # )
    
    track_pose_global = RewTerm(
        func=mdp.box_keypoint_l2,
        weight=-10.0,
        params={"command_name": "box_goal", "half_side": 0.1},
    )
    # track_pose_precision = RewTerm(
    #     func=mdp.box_keypoint_sqrt,
    #     weight=-10.0,
    #     params={"command_name": "box_goal", "half_side": 0.1},
    # )


    # approach_obj = RewTerm(
    #     func=mdp.approach_object,
    #     weight=2.0,
    #     params={
    #         "std": 0.5,
    #         "object_cfg": SceneEntityCfg("object"),
    #         "robot_cfg": SceneEntityCfg("robot"), 
    #     },
    # )
    # box_stationary = RewTerm(
    #     func=mdp.box_stationary_penalty,
    #     weight=-2.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "threshold": 0.05},
    # )
    # box_vel_to_goal = RewTerm(
    #     func=mdp.box_vel_toward_goal,
    #     weight=3.0,
    #     params={"command_name": "box_goal"},
    # )
    # activity = RewTerm(
    #     func=mdp.joint_activity,
    #     weight=1.0,
    # )
    success_bonus = RewTerm(
        func=mdp.manipulation_success_bonus,
        weight=5.0,
        params={"command_name": "box_goal"},
    )
    time_penalty = RewTerm(func=mdp.time_penalty, weight=-0.5)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.005)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    goal_completion = RewTerm(
        func=mdp.goal_completion_bonus,
        weight=100.0,
        params={"command_name": "box_goal"},
    )


@configclass
class TerminationsCfg:
    """Termination: timeout or goal reached."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "box_goal"},
    )

@configclass
class CurriculumCfg:
    """Curriculum: tighten goal thresholds as success rate improves."""

    # goal_threshold = CurrTerm(
    #     func=mdp.BoxPushGoalThresholdCurriculum,
    #     params={
    #         "command_name": "box_goal",
    #         "initial_pos_threshold": 0.15,
    #         "final_pos_threshold": 0.05,
    #         "initial_yaw_threshold": 0.50,
    #         "final_yaw_threshold": 0.20,
    #         "success_rate_threshold": 0.7,
    #         "ramp_rate": 0.01,
    #         "ema_alpha": 0.05,
    #     },
    # )
    anneal_success_bonus = CurrTerm(
        func=mdp.BoxPushAnnealRewardWeight,
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
class CobraBoxPushEnvCfg(ManagerBasedRLEnvCfg):
    """Cobra planar box-pushing environment."""

    scene: CobraBoxPushSceneCfg = CobraBoxPushSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 3.0, 4.5)
        self.viewer.lookat = (0.0, -1.2, 0.0)


@configclass
class CobraBoxPushEnvCfg_PLAY(CobraBoxPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # self.terminations.time_out = None
