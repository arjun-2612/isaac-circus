# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared sampling utilities for 3D box manipulation."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

# Six quaternions (w, x, y, z) — each places a different cube face upward
FACE_QUATERNIONS = torch.tensor([
    [1.0000, 0.0000, 0.0000, 0.0000],   # +Z up  (identity)
    [0.0000, 1.0000, 0.0000, 0.0000],   # -Z up  (180° about X)
    [0.7071, 0.0000, 0.7071, 0.0000],   # +X up  ( 90° about Y)
    [0.7071, 0.0000, -0.7071, 0.0000],  # -X up  (-90° about Y)
    [0.7071, -0.7071, 0.0000, 0.0000],  # +Y up  (-90° about X)
    [0.7071, 0.7071, 0.0000, 0.0000],   # -Y up  ( 90° about X)
])


# def sample_face_yaw_quat(n: int, yaw_range: tuple[float, float], device: str) -> torch.Tensor:
#     """Sample n quaternions: random face × random yaw. Shape (n, 4)."""
#     face_idx = torch.randint(0, 6, (n,), device=device)
#     face_q = FACE_QUATERNIONS.to(device)[face_idx]

#     yaw = torch.empty(n, device=device).uniform_(
#         math.radians(yaw_range[0]), math.radians(yaw_range[1])
#     )
#     zeros = torch.zeros_like(yaw)
#     yaw_q = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

#     return math_utils.quat_mul(yaw_q, face_q)

def sample_face_yaw_quat(n: int, yaw_range: tuple[float, float], device: str) -> torch.Tensor:
    """Sample n quaternions: +Z up with random yaw. Shape (n, 4)."""
    yaw = torch.empty(n, device=device).uniform_(
        math.radians(yaw_range[0]), math.radians(yaw_range[1])
    )
    zeros = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

def sample_valid_xy(
    n: int,
    xy_range: tuple[tuple[float, float], tuple[float, float]],
    robot_exclusion: tuple[tuple[float, float], tuple[float, float]],
    device: str,
    avoid_xy: torch.Tensor | None = None,
    min_dist: float = 0.0,
    max_attempts: int = 50,
) -> torch.Tensor:
    """Vectorised rejection sampling for positions outside the robot zone.

    Args:
        n: Number of positions to sample.
        xy_range: ((x_min, x_max), (y_min, y_max)) sampling bounds.
        robot_exclusion: ((x_min, x_max), (y_min, y_max)) rectangular zone to avoid.
        device: Torch device.
        avoid_xy: Optional (n, 2) tensor of positions to keep distance from.
        min_dist: Minimum distance from avoid_xy positions.
        max_attempts: Maximum rejection sampling iterations.

    Returns:
        (n, 2) tensor of valid xy positions.
    """
    (x_lo, x_hi), (y_lo, y_hi) = xy_range
    (rx_lo, rx_hi), (ry_lo, ry_hi) = robot_exclusion

    xy = torch.zeros(n, 2, device=device)
    remaining = torch.ones(n, dtype=torch.bool, device=device)

    for _ in range(max_attempts):
        k = remaining.sum().item()
        if k == 0:
            break

        cand = torch.zeros(k, 2, device=device)
        cand[:, 0].uniform_(x_lo, x_hi)
        cand[:, 1].uniform_(y_lo, y_hi)

        # reject if inside robot exclusion rectangle
        in_robot = (
            (cand[:, 0] > rx_lo) & (cand[:, 0] < rx_hi)
            & (cand[:, 1] > ry_lo) & (cand[:, 1] < ry_hi)
        )
        valid = ~in_robot

        # reject if too close to avoid points
        if avoid_xy is not None and min_dist > 0:
            dist = torch.norm(cand - avoid_xy[remaining], dim=-1)
            valid = valid & (dist > min_dist)

        accepted = remaining.nonzero(as_tuple=False).squeeze(-1)[valid]
        xy[accepted] = cand[valid]
        remaining[accepted] = False

    # fallback: place at far corner (guaranteed outside robot zone)
    if remaining.any():
        xy[remaining, 0] = x_hi
        xy[remaining, 1] = y_hi

    return xy