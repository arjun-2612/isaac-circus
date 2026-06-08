from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from .multimodal_command import HuskyBetaMultimodalJointPosCommand

@configclass
class HuskyBetaMultimodalJointPosCommandCfg(CommandTermCfg):
    """Configuration for Raibert foot-placement command term."""

    class_type: type = HuskyBetaMultimodalJointPosCommand

    asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    action_name: str = MISSING
    """Name of the MM action."""

    # --- Visualization ---
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to RED_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)