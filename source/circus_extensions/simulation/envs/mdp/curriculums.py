"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_of_god(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    metric_command_term_name: str,
    wrench_assist_command_term_name: str,
    wrench_assist_decrease: float,
    wrench_assist_increase: float,
):
    command_term = env.command_manager.get_term(metric_command_term_name)
    wrench_command_term = env.command_manager.get_term(wrench_assist_command_term_name)
    wrench_scale = wrench_command_term.wrench_scale
    max_wrench_scale = wrench_command_term.cfg.max_wrench_scale

    decrease_difficulty, increase_difficulty = command_term.get_curriculum_info(env_ids)
    wrench_scale[env_ids] += decrease_difficulty * wrench_assist_increase
    wrench_scale[env_ids] -= increase_difficulty * wrench_assist_decrease
    wrench_scale[env_ids] = torch.clip(wrench_scale[env_ids], min=0.0, max=max_wrench_scale)
    wrench_scale[wrench_scale < wrench_command_term.cfg.min_wrench_scale] = 0.0
    wrench_command_term.command_level[env_ids] = 1.0 - (wrench_scale[env_ids] / max_wrench_scale)

    return torch.mean(wrench_scale)


def update_cmd(env: ManagerBasedRLEnv, env_ids: Sequence[int], upper_threshold: float, lower_threshold: float, delta: float):
    command_term = env.command_manager.get_term("base_velocity")
    metric_xy = command_term.metrics["error_vel_xy"]

    decrease_difficulty = (metric_xy > upper_threshold) + env.termination_manager.terminated
    increase_difficulty = (metric_xy < lower_threshold) * env.termination_manager.time_outs * ~decrease_difficulty

    command_term.scale_factor[env_ids] += increase_difficulty[env_ids] * delta
    command_term.scale_factor[env_ids] -= decrease_difficulty[env_ids] * delta
    command_term.scale_factor[env_ids] = torch.clip(command_term.scale_factor[env_ids], min=0.0, max=1.0)

    return torch.mean(command_term.scale_factor)


def terrain_levels_pose(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    move_up_distance: float = 0.2,
    move_down_distance: float = 0.7,
) -> torch.Tensor:
    """Curriculum based on how close the robot is to the goal pose. 

    Robots that reach the pose and get within a threshold of the pose go up, while robots that do not reach 
    the pose go down. 

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("pose_command")
    # compute the distance to the goal pose (base frame)
    distance_to_goal = torch.norm(command[env_ids, :3], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance_to_goal <= move_up_distance
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance_to_goal > (move_down_distance)
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
