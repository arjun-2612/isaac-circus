# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActionManager variant that additionally tracks the action from two steps ago (a[t-2]).

This lives in isaac-circus (NOT the IsaacLab submodule) so we never edit official IsaacLab code.
The stock ``ActionManager`` only keeps ``action`` (a[t]) and ``prev_action`` (a[t-1]); adding
``prev_prev_action`` (a[t-2]) makes a second-order / jerk action-smoothness penalty trivial:

    jerk = a[t] - 2*a[t-1] + a[t-2]  ==  action - 2*prev_action + prev_prev_action
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers.action_manager import ActionManager


class CircusActionManager(ActionManager):
    """:class:`ActionManager` that also exposes ``prev_prev_action`` (a[t-2])."""

    def __init__(self, cfg: object, env):
        super().__init__(cfg, env)
        # extra one-step-deeper history buffer (a[t-2])
        self._prev_prev_action = torch.zeros_like(self._action)

    @property
    def prev_prev_action(self) -> torch.Tensor:
        """The action from two steps ago, a[t-2]. Shape is (num_envs, total_action_dim)."""
        return self._prev_prev_action

    def process_action(self, action: torch.Tensor):
        # shift the deeper history first: prev_prev <- prev (a[t-2] <- old a[t-1]),
        # then the base shifts prev <- current and current <- new action.
        self._prev_prev_action[:] = self._prev_action
        super().process_action(action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        result = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prev_prev_action[env_ids] = 0.0
        return result
