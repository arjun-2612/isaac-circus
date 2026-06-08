# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.envs.mdp.recorders import PostStepStatesRecorderCfg, PreStepActionsRecorderCfg, PreStepFlatPolicyObservationsRecorderCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class FootForcesRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.FootForcesRecorder


@configclass
class FootPositionsRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.FootPositionsRecorder


@configclass
class RecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    # foot_forces_recorder = FootForcesRecorderCfg()
    foot_positions_recorder = FootPositionsRecorderCfg()
    post_step_states_recorder = PostStepStatesRecorderCfg()
    pre_step_actions_recorder = PreStepActionsRecorderCfg()
    pre_step_flat_policy_observations_recorder = PreStepFlatPolicyObservationsRecorderCfg()
