# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry augmentation for Spot badminton swing task (left-right limb mirroring)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


def spot_badminton_left_right_symmetry(
    env: VecEnv = None,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Mirror left/right limbs for Spot badminton task (Mittal et al. 2403.04359).

    Data augmentation: returns concatenated [original; mirrored] for 2x batch size.
    Mirror loss: PPO compares policy outputs on original vs mirrored observations.

    Spot has 18 joints with natural left-right symmetry:
        0-3:   fl_hx, fr_hx, hl_hx, hr_hx  (hip_x, abduction/adduction)
        4-7:   fl_hy, fr_hy, hl_hy, hr_hy  (hip_y, pitch)
        8-11:  fl_kn, fr_kn, hl_kn, hr_kn  (knee)
        12-17: arm joints (no symmetry)

    Left-right mirroring:
    - Hip_x (0-3): swap with sign flip (−fl ↔ fr, −hl ↔ hr)
    - Hip_y (4-7), Knee (8-11): swap without sign flip
    - Arm: unchanged
    - Observations: mirror body velocity y-component, orientation y/z axes, EE pose y-component,
      shuttle position y-component, joint pos/vel by same leg swap rules

    Args:
        env: Environment object (passed by PPO for context).
        obs: Observation batch [num_envs, obs_dim], or None.
        actions: Action batch [num_envs, action_dim], or None.
        obs_type: "policy" or "critic" — determines which obs structure to use.

    Returns:
        Tuple of (aug_obs, aug_actions) where each is [2*num_envs, ...] by concatenating
        [original; mirrored]. Either can be None if input is None.
    """

    # Mirror actions: create the left-right flipped version
    if actions is not None:
        mirrored_action = actions.clone()

        # Hip_x (0-3): swap with sign flip
        mirrored_action[:, 0] = -actions[:, 1]   # fl_hx <- -fr_hx
        mirrored_action[:, 1] = -actions[:, 0]   # fr_hx <- -fl_hx
        mirrored_action[:, 2] = -actions[:, 3]   # hl_hx <- -hr_hx
        mirrored_action[:, 3] = -actions[:, 2]   # hr_hx <- -hl_hx

        # Hip_y (4-7): swap without sign flip
        mirrored_action[:, 4] = actions[:, 5]    # fl_hy <- fr_hy
        mirrored_action[:, 5] = actions[:, 4]    # fr_hy <- fl_hy
        mirrored_action[:, 6] = actions[:, 7]    # hl_hy <- hr_hy
        mirrored_action[:, 7] = actions[:, 6]    # hr_hy <- hl_hy

        # Knee (8-11): swap without sign flip
        mirrored_action[:, 8] = actions[:, 9]    # fl_kn <- fr_kn
        mirrored_action[:, 9] = actions[:, 8]    # fr_kn <- fl_kn
        mirrored_action[:, 10] = actions[:, 11]  # hl_kn <- hr_kn
        mirrored_action[:, 11] = actions[:, 10]  # hr_kn <- hl_kn

        # Arm (12-17): no symmetry, unchanged

        # Data augmentation: concatenate [original; mirrored] for 2x batch size
        aug_action = torch.cat([actions, mirrored_action], dim=0)
    else:
        aug_action = None

    # Mirror observations: create the left-right flipped version
    if obs is not None:
        mirrored_obs = obs.clone()

        # Policy observation structure: [base_lin_vel(3), base_ang_vel(3), base_6do(6), joint_pos(18),
        # joint_vel(18), actions(18), ee_position(3), ee_velocity(3), ee_orientation(4),
        # swing_target_pos(3), time_to_swing(1), shuttle_pos_history(15), shuttle_detected(1)]
        # Total: 117 dimensions

        if obs_type == "policy":
            o = 0

            # base_lin_vel (o=0-2): mirror y-component
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # base_ang_vel (o=3-5): mirror z-component (yaw)
            mirrored_obs[:, o+2] = -obs[:, o+2]
            o += 3

            # base_6do (o=6-11): rotation matrix [R00, R10, R20, R01, R11, R21]
            # Y-axis flip: negate columns that have Y dependence
            mirrored_obs[:, o+1] = -obs[:, o+1]  # -R10
            mirrored_obs[:, o+2] = -obs[:, o+2]  # -R20
            mirrored_obs[:, o+3] = -obs[:, o+3]  # -R01
            o += 6

            # joint_pos (o=12-29): mirror leg joints (same swap as actions)
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # joint_vel (o=30-47): mirror leg joints (same swap as actions)
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # actions (o=48-65): last action obs (same swap as actions)
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # ee_position (o=66-68): mirror y-component
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # ee_velocity (o=69-71): mirror y-component
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # ee_orientation (o=72-75): quaternion [w, x, y, z] -> [w, x, -y, -z] for Y-axis flip
            mirrored_obs[:, o+2] = -obs[:, o+2]  # flip y
            mirrored_obs[:, o+3] = -obs[:, o+3]  # flip z
            o += 4

            # swing_target_pos (o=76-78): mirror y-component
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # time_to_swing (o=79): no change
            o += 1

            # shuttle_pos_history (o=80-94): mirror y for each of 5 historic positions
            for i in range(5):
                mirrored_obs[:, o + i*3 + 1] = -obs[:, o + i*3 + 1]
            o += 15

            # shuttle_detected (o=95): no change
            # o += 1

        elif obs_type == "critic":
            # Critic observation structure (93 dims): [base_lin_vel(3), base_ang_vel(3), base_6do(6),
            # joint_pos(18), joint_vel(18), actions(18), ee_position(3), ee_velocity(3), ee_orientation(4),
            # swing_target_pos(3), time_to_swing(1), shuttle_pos(3), shuttle_vel(3), targets_remaining(1),
            # next_target_dist(3), swing_target_prediction_error(3)]
            # Note: critic has shuttle_pos(3), NOT shuttle_pos_history(15) like policy
            o = 0

            # base_lin_vel (o=0-2): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # base_ang_vel (o=3-5): mirror z
            mirrored_obs[:, o+2] = -obs[:, o+2]
            o += 3

            # base_6do (o=6-11): mirror Y-dependence
            mirrored_obs[:, o+1] = -obs[:, o+1]
            mirrored_obs[:, o+2] = -obs[:, o+2]
            mirrored_obs[:, o+3] = -obs[:, o+3]
            o += 6

            # joint_pos (o=12-29): mirror leg joints
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # joint_vel (o=30-47): mirror leg joints
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # actions (o=48-65): mirror leg actions
            _mirror_leg_obs(mirrored_obs, obs, o, o)
            o += 18

            # ee_position (o=66-68): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # ee_velocity (o=69-71): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # ee_orientation (o=72-75): quaternion, mirror y/z
            mirrored_obs[:, o+2] = -obs[:, o+2]
            mirrored_obs[:, o+3] = -obs[:, o+3]
            o += 4

            # swing_target_pos (o=76-78): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # time_to_swing (o=79): no change
            o += 1

            # shuttle_pos (o=80-82): mirror y (NOTE: 3 dims, not 15!)
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # shuttle_vel (o=83-85): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # targets_remaining (o=86): no change
            o += 1

            # next_target_dist (o=87-89): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            o += 3

            # swing_target_prediction_error (o=90-92): mirror y
            mirrored_obs[:, o+1] = -obs[:, o+1]
            # o += 3

        # Data augmentation: concatenate [original; mirrored] for 2x batch size
        aug_obs = torch.cat([obs, mirrored_obs], dim=0)
    else:
        aug_obs = None

    return aug_obs, aug_action


def _mirror_leg_obs(mirrored_obs: torch.Tensor, obs: torch.Tensor, aug_offset: int, obs_offset: int):
    """Mirror leg joint observations (positions, velocities, or action history).

    Applies the same left-right swap rules as actions:
    - Hip_x (0-3): swap with sign flip
    - Hip_y (4-7): swap without sign flip
    - Knee (8-11): swap without sign flip
    - Arm (12-17): no change
    """
    # Hip_x: flip sign and swap (abduction/adduction)
    mirrored_obs[:, aug_offset+0] = -obs[:, obs_offset+1]  # fl_hx <- -fr_hx
    mirrored_obs[:, aug_offset+1] = -obs[:, obs_offset+0]  # fr_hx <- -fl_hx
    mirrored_obs[:, aug_offset+2] = -obs[:, obs_offset+3]  # hl_hx <- -hr_hx
    mirrored_obs[:, aug_offset+3] = -obs[:, obs_offset+2]  # hr_hx <- -hl_hx

    # Hip_y: swap without sign flip
    mirrored_obs[:, aug_offset+4] = obs[:, obs_offset+5]   # fl_hy <- fr_hy
    mirrored_obs[:, aug_offset+5] = obs[:, obs_offset+4]   # fr_hy <- fl_hy
    mirrored_obs[:, aug_offset+6] = obs[:, obs_offset+7]   # hl_hy <- hr_hy
    mirrored_obs[:, aug_offset+7] = obs[:, obs_offset+6]   # hr_hy <- hl_hy

    # Knee: swap without sign flip
    mirrored_obs[:, aug_offset+8] = obs[:, obs_offset+9]    # fl_kn <- fr_kn
    mirrored_obs[:, aug_offset+9] = obs[:, obs_offset+8]    # fr_kn <- fl_kn
    mirrored_obs[:, aug_offset+10] = obs[:, obs_offset+11]  # hl_kn <- hr_kn
    mirrored_obs[:, aug_offset+11] = obs[:, obs_offset+10]  # hr_kn <- hl_kn

    # Arm: no symmetry, keep as-is
    mirrored_obs[:, aug_offset+12:aug_offset+18] = obs[:, obs_offset+12:obs_offset+18]
