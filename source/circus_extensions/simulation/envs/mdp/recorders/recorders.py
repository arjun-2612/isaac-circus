# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.managers.recorder_manager import RecorderTerm

class FootForcesRecorder(RecorderTerm):

    def record_post_step(self):
        computed_and_applied_forces = dict()
        computed_forces = self._env.command_manager.get_term("controller").foot_forces
        applied_forces = self._env.command_manager.get_term("controller").applied_forces
        computed_and_applied_forces["computed_forces"] = computed_forces
        computed_and_applied_forces["applied_forces"] = applied_forces
        return "foot_forces", computed_and_applied_forces
    
class FootPositionsRecorder(RecorderTerm):

    def record_post_step(self):
        commanded_and_observed_foot_positions = dict()
        commanded_foot_positions = self._env.command_manager.get_term("controller").swing_targets
        observed_foot_positions = self._env.command_manager.get_term("controller").foot_positions
        commanded_and_observed_foot_positions["commanded_foot_positions"] = commanded_foot_positions
        commanded_and_observed_foot_positions["observed_foot_positions"] = observed_foot_positions
        return "foot_positions", commanded_and_observed_foot_positions