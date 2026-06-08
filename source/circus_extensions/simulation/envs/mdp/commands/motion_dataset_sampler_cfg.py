from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .motion_dataset_sampler import MotionDatasetSampler


@configclass
class MotionDatasetSamplerCfg(CommandTermCfg):
    """Configuration for motion dataset."""

    class_type: type = MotionDatasetSampler

    robot_asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    data_dir: str = MISSING

    task_phase_min: float = 0.0
    """Task phase start value."""

    task_phase_max: float = 1.0
    """Task phase end value."""

    score_std: float = 10.0
    """Exponential kernel std scaling factor for score metrics."""

    max_extra_time: float = 5.0
    """Maximum wait time at the end of the trajectory before resetting to a new phase"""

    tracking_term_tolerance: dict[str, float] = {
        "projected_gravity": 0.3,
        "root_lin_vel": 0.2,
        "root_ang_vel": 0.3,
        "joint_pos": 0.2,
        "link_pos": 0.2,
    }
    """A fixed set of tolerances for all tracking terms of interest needed to define the tracking metrics."""
