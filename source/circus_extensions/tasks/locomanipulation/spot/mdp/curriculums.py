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

def shuttle_distance_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    increase_difficulty_threshold: float = 0.8, # >= 5/6 success
    decrease_difficulty_threshold: float = 0.3, # < 2/6 success
    level_step: float = 0.1,
) -> float:
    """
    Curriculum that widens shuttle intercept spread based on EE success rate.
    
    Returns current difficulty level for logging.
    """
    cmd = env.command_manager.get_term(command_name)
    metric_success = cmd.metrics["episode_success_rate"]
    increase_difficulty = (metric_success > increase_difficulty_threshold)
    decrease_difficulty = (metric_success < decrease_difficulty_threshold)
    
    cmd.level[env_ids] += increase_difficulty[env_ids] * level_step
    cmd.level[env_ids] -= decrease_difficulty[env_ids] * level_step
    cmd.level = torch.clip(cmd.level, min=0.0, max=1.0)
    
    return torch.mean(cmd.level)


# def swing_speed_curriculum(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     command_name: str,
#     min_swing_speed: float = 4.0,
#     max_swing_speed: float = 14.0,
#     headroom: float = 1.25,
#     ema_decay: float = 0.995,
#     max_step_per_update: float = 1.0,
# ) -> float:
#     """Auto-calibrating swing-speed target for the velocity/follow-through rewards.

#     Tracks the racket speed actually achieved along the swing direction at impact,
#     and keeps ``swing_speed`` ~``headroom`` above it. This keeps the reward gradient
#     steep (reward ~= achieved / target ~= 1/headroom) instead of saturating near 0,
#     so there is always positive pressure to swing faster -- up to ``max_swing_speed``.
#     """
#     cmd = env.command_manager.get_term(command_name)

#     cmd.swing_speed_ema[env_ids] = ema_decay * cmd.swing_speed_ema[env_ids] + (1.0 - ema_decay) * cmd.achieved_racket_speed[env_ids]
#     desired = min(max(cmd.swing_speed_ema[env_ids] * headroom, min_swing_speed), max_swing_speed)
#     delta = max(-max_step_per_update, min(max_step_per_update, desired - cmd.swing_speed_target[env_ids]))
#     cmd.swing_speed_target[env_ids] += delta

#     return torch.mean(cmd.swing_speed_target)