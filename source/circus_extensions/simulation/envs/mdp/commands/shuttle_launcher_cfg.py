from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from .shuttle_launcher import ShuttleLauncherCommand

@configclass
class ShuttleLauncherCommandCfg(CommandTermCfg):
    """Configuration for Shuttle launcher command term."""

    class_type: type = ShuttleLauncherCommand

    asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    intercept_height: float = MISSING 
    """Height at which the shuttle is to be intercepted [m]."""

    time_between_targets: float = MISSING
    """Time between successive targets [s]."""

    num_targets: int = 4
    """Number of targets to be generated."""

    standing_env_ratio: float = 0.1
    """Ratio of environments that have no targets."""

    ee_frame: SceneEntityCfg = MISSING
    """Config for the end-effector frame used for metrics."""

    asset_cfg: SceneEntityCfg = MISSING
    """Config for the asset used for the command."""

    success_threshold: float = 0.3
    """Distance threshold for successful interception [m]."""

    racket_swing_speed_target: float = 4.0
    """Target swing speed [m/s]."""

    restitution: float = 0.4
    """Coefficient of restitution for the racket-shuttle impact (1.0 = perfectly elastic).

    Applied to the normal component of the shuttle velocity in the racket's frame. Tune /
    system-ID against real return speeds later.
    """

    contact_radius: float = 0.25
    """Max distance [m] between the racket sweet-spot and the shuttle for a hit to register.

    Interceptions where the racket is farther than this are treated as misses: the shuttle
    passes through untouched instead of being deflected.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        px: tuple[float, float] = MISSING  # min max [m]
        py: tuple[float, float] = MISSING  # min max [m]
        vy: tuple[float, float] = MISSING  # min max [m/s]
        vz: tuple[float, float] = MISSING  # min max [m/s]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    # --- Visualization ---
    shuttle_target_visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/shuttle_position_target"
    )

    shuttle_target_visualizer_cfg.markers["hit"].radius = 0.05

    shuttle_pos_visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/shuttle_position"
    )

    shuttle_pos_visualizer_cfg.markers["hit"].radius = 0.05
    shuttle_pos_visualizer_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)