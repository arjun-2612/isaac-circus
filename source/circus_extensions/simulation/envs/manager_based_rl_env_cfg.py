# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from circus_extensions.simulation.envs.manager_based_rl_env_window import CircusManagerBasedRLEnvWindow


@configclass
class CircusManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    # ui settings
    ui_window_class_type: type | None = CircusManagerBasedRLEnvWindow
