from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .wrench_assist_command import WrenchAssistCommand


@configclass
class WrenchAssistCommandCfg(CommandTermCfg):
    """Configuration for force assist command term."""

    class_type: type = WrenchAssistCommand

    asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    cmd_name: str = MISSING
    """Name of the command term from which to read the reference motion data."""

    wrench_assist_body_name: str = MISSING
    """Name of the robot root link for which assistive wrenches are applied to."""

    max_wrench_scale: float = 1.0
    """The maximum allowable wrench assist factor: W_assist = max_wrench_scale * W_assist"""

    min_wrench_scale: float = 0.0
    """The minimum threshold of the mean wrench assist scale below which we set the wrench scale to exactly 0.0"""

    force_gain: tuple[float, float] = (0.0, 0.0)
    """Gain applied to setting assistive force"""

    torque_gain: tuple[float, float] = (0.0, 0.0)
    """Gain applied to setting the assistive torque"""

    force_assist_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Curriculum/force_assist"
    )
    """The configuration for the current force assist visualization marker. Defaults to RED_ARROW_X_MARKER_CFG. This visualization is only applied to the force component of the assistive wrench."""

    force_assist_arrow_z_offset: float = 1.25
    """Z height offset for the position of the force assist visualization arrow."""

    force_assist_arrow_scale: float = 10
    """Scaling factor for the size of the force assist visualization arrow."""

    force_assist_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)