"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
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
#     asset_cfg: SceneEntityCfg,
#     command_name: str,
#     reward_term_names: Sequence[str] = ("ee_velocity", "ee_follow_through"),
#     racket_body_name: str = "arm0_racket_face",
#     init_swing_speed: float = 4.0,
#     min_swing_speed: float = 4.0,
#     max_swing_speed: float = 14.0,
#     headroom: float = 1.25,
#     ema_decay: float = 0.995,
#     max_step_per_update: float = 0.02,
# ) -> float:
#     """Auto-calibrating swing-speed target for the velocity/follow-through rewards.

#     Tracks the racket speed actually achieved along the swing direction at impact,
#     and keeps ``swing_speed`` ~``headroom`` above it. This keeps the reward gradient
#     steep (reward ~= achieved / target ~= 1/headroom) instead of saturating near 0,
#     so there is always positive pressure to swing faster -- up to ``max_swing_speed``.
#     """
#     cmd = env.command_manager.get_term(command_name)
#     asset = env.scene[asset_cfg.name]

#     # lazy state
#     if not hasattr(env, "_swing_speed_target"):
#         env._swing_speed_target = float(init_swing_speed)
#         env._swing_speed_ema = float(init_swing_speed) / headroom
#         env._racket_body_id = env.scene["robot"].find_bodies(racket_body_name)[0][0]

#     # measure achieved racket speed along the swing dir, only at the impact step
#     at_impact = cmd.at_intercept_time  # (E,) bool, true one step per target
#     if at_impact.any():
#         racket_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids, :3].squeeze(1)

#         swing_dir = -cmd.interception_vel.clone()
#         swing_dir[:, 2] = swing_dir[:, 2] + 1.0
#         swing_dir[:, 1] = 0.0
#         swing_dir = swing_dir / (torch.norm(swing_dir, dim=-1, keepdim=True) + 1e-6)

#         speed_along = torch.sum(racket_vel_w * swing_dir, dim=-1)
#         achieved = speed_along[at_impact].clamp(min=0.0).mean().item()
#         env._swing_speed_ema = ema_decay * env._swing_speed_ema + (1.0 - ema_decay) * achieved

#     # move the target toward achieved * headroom, rate-limited
#     desired = min(max(env._swing_speed_ema * headroom, min_swing_speed), max_swing_speed)
#     delta = max(-max_step_per_update, min(max_step_per_update, desired - env._swing_speed_target))
#     env._swing_speed_target += delta

#     # push the new target into the reward terms (get_term_cfg returns the live cfg)
#     for name in reward_term_names:
#         env.reward_manager.get_term_cfg(name).params["swing_speed"] = env._swing_speed_target

#     return env._swing_speed_target