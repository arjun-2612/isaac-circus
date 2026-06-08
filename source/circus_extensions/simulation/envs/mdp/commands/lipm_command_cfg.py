from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .lipm_command import LIPMCommand

@configclass
class LIPMCommandCfg(CommandTermCfg):
    """Configuration for Raibert foot-placement command term."""

    class_type: type = LIPMCommand

    asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    action_name: str = MISSING
    """Name of the RQP action that outputs desired planar velocity per env (expects at least [vx, vy])."""

    foot_bodies: str = MISSING 
    """Names of the foot bodies in the asset."""

    # --- Visualization ---
    foot_targets_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/foot_position_targets",
        markers={
            "fl": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "fr": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "rl": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "rr": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )