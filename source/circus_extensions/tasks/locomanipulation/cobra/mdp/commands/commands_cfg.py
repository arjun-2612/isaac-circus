# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .planar_command import PlanarManipulationCommand
from .box_manipulation_command import BoxManipulationCommand


@configclass
class PlanarManipulationCommandCfg(CommandTermCfg):
    """Configuration for the planar (x, y, yaw) goal command.

    The goal is sampled once at episode reset and held fixed for the
    entire episode.  The episode terminates on success or timeout.
    """

    class_type: type = PlanarManipulationCommand
    
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # never resample on time

    asset_name: str = MISSING
    """Name of the manipulated object in the scene."""

    goal_x_range: tuple[float, float] = MISSING
    goal_y_range: tuple[float, float] = MISSING
    goal_yaw_range: tuple[float, float] = MISSING
    """Sampling ranges for the goal (x, y in metres; yaw in degrees)."""

    object_height: float = 0.1
    """Z height of the goal position (so the marker sits on the ground)."""

    pos_success_threshold: float = MISSING
    """Position error (m) below which the goal is considered reached."""

    yaw_success_threshold: float = MISSING
    """Yaw error (rad) below which the goal is considered reached."""

    goal_visualizer_cfg: sim_utils.UsdFileCfg | None = None
    """Optional USD to visualise the goal frame."""

    required_settled_time: float = 0.5
    """Time (seconds) the box must remain settled at the goal."""

    max_goal_speed: float = 0.05
    """Maximum box xy speed (m/s) to count as settled."""


@configclass
class BoxManipulationCommandCfg(CommandTermCfg):
    """Configuration for the full 3D box manipulation goal command.

    Samples box and goal poses at reset: random xy (avoiding robot footprint),
    random cube face, random yaw. Goal must also be far enough from box.
    """

    class_type: type = BoxManipulationCommand

    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_name: str = MISSING
    """Name of the manipulated object in the scene."""

    xy_range: tuple[tuple[float, float], tuple[float, float]] = MISSING
    """((x_min, x_max), (y_min, y_max)) sampling bounds for box and goal."""

    robot_exclusion: tuple[tuple[float, float], tuple[float, float]] = MISSING
    """((x_min, x_max), (y_min, y_max)) rectangular zone around robot to avoid."""

    yaw_range: tuple[float, float] = (-180.0, 180.0)
    """Yaw sampling range in degrees (applied on top of face rotation)."""

    min_box_goal_dist: float = 0.3
    """Minimum xy distance between box and goal positions."""

    object_height: float = 0.105
    """Z height of box/goal centre (cube half-extent + ground clearance)."""

    pos_success_threshold: float = MISSING
    """Position error (m) below which the goal is considered reached."""

    orientation_success_threshold: float = MISSING
    """Orientation error (rad) below which the goal is considered reached."""

    max_goal_speed: float = 0.05
    """Maximum box xy speed (m/s) to count as settled."""

    required_settled_time: float = 0.5
    """Time (seconds) the box must remain settled at the goal."""

    goal_visualizer_cfg: sim_utils.UsdFileCfg | None = None
    """Optional USD to visualise the goal frame."""