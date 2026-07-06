# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv variant that uses the CircusActionManager (tracks a[t-2]).

The stock env hardcodes ``ActionManager`` in ``load_managers``, so to get ``prev_prev_action``
(needed for second-order / jerk action penalties) we override ``load_managers`` and swap in the
CircusActionManager. Re-instantiating re-parses the (side-effect-free) action terms; the gym action
space is unchanged since the action dimension is identical.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from circus_extensions.simulation.managers.action_manager import CircusActionManager


class CircusManagerBasedRLEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` whose action manager also tracks a[t-2] (``prev_prev_action``)."""

    def load_managers(self):
        super().load_managers()
        # swap the stock ActionManager for the jerk-tracking one (same cfg, same action dim)
        self.action_manager = CircusActionManager(self.cfg.actions, self)
