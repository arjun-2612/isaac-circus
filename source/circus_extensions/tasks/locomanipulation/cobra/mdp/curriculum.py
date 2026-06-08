# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the cobra box-push environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BoxPushGoalThresholdCurriculum(ManagerTermBase):
    """Tightens goal thresholds as the agent's success rate improves.

    Tracks success rate via exponential moving average. When it exceeds
    a threshold, difficulty ramps up, linearly interpolating the position
    and yaw thresholds from loose initial values to tight final values.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._cmd_term = env.command_manager.get_term(cfg.params["command_name"])

        self._initial_pos = cfg.params["initial_pos_threshold"]
        self._final_pos = cfg.params["final_pos_threshold"]
        self._initial_yaw = cfg.params["initial_yaw_threshold"]
        self._final_yaw = cfg.params["final_yaw_threshold"]

        self._success_threshold = cfg.params.get("success_rate_threshold", 0.7)
        self._ramp_rate = cfg.params.get("ramp_rate", 0.01)
        self._ema_alpha = cfg.params.get("ema_alpha", 0.05)

        self._difficulty = 0.0
        self._success_ema = 0.0

        # apply initial thresholds
        self._cmd_term.cfg.pos_success_threshold = self._initial_pos
        self._cmd_term.cfg.yaw_success_threshold = self._initial_yaw

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        command_name: str,
        initial_pos_threshold: float,
        final_pos_threshold: float,
        initial_yaw_threshold: float,
        final_yaw_threshold: float,
        success_rate_threshold: float = 0.7,
        ramp_rate: float = 0.01,
        ema_alpha: float = 0.05,
    ) -> dict[str, float]:
        if len(env_ids) == 0:
            return {}

        # check which resetting envs terminated via goal_reached
        goal_dones = env.termination_manager.get_term("goal_reached")
        success_rate = goal_dones[env_ids].float().mean().item()

        # update EMA
        self._success_ema = (1 - self._ema_alpha) * self._success_ema + self._ema_alpha * success_rate

        # ramp difficulty when success rate is high
        if self._success_ema > self._success_threshold:
            self._difficulty = min(1.0, self._difficulty + self._ramp_rate)

        # interpolate thresholds
        d = self._difficulty
        self._cmd_term.cfg.pos_success_threshold = self._initial_pos + (self._final_pos - self._initial_pos) * d
        self._cmd_term.cfg.yaw_success_threshold = self._initial_yaw + (self._final_yaw - self._initial_yaw) * d

        return {
            "difficulty": self._difficulty,
            "success_ema": self._success_ema,
            "pos_threshold": self._cmd_term.cfg.pos_success_threshold,
            "yaw_threshold": self._cmd_term.cfg.yaw_success_threshold,
        }

class BoxPushAnnealRewardWeight(ManagerTermBase):
    """Anneal a reward term's weight from its initial value to a target over training."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._reward_name = p["reward_name"]
        self._target_weight = p["target_weight"]
        self._initial_weight = env.reward_manager.get_term_cfg(self._reward_name).weight
        self._success_threshold = p.get("success_rate_threshold", 0.7)
        self._ramp_rate = p.get("ramp_rate", 0.01)
        self._ema_alpha = p.get("ema_alpha", 0.05)
        self._success_ema = 0.0
        self._progress = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        reward_name: str,
        target_weight: float,
        success_rate_threshold: float = 0.7,
        ramp_rate: float = 0.01,
        ema_alpha: float = 0.05,
    ) -> dict[str, float]:
        if len(env_ids) == 0:
            return {}

        goal_dones = env.termination_manager.get_term("goal_reached")
        success_rate = goal_dones[env_ids].float().mean().item()
        self._success_ema = (1 - self._ema_alpha) * self._success_ema + self._ema_alpha * success_rate

        if self._success_ema > self._success_threshold:
            self._progress = min(1.0, self._progress + self._ramp_rate)

        current_weight = self._initial_weight + (self._target_weight - self._initial_weight) * self._progress

        term_cfg = env.reward_manager.get_term_cfg(self._reward_name)
        term_cfg.weight = current_weight
        env.reward_manager.set_term_cfg(self._reward_name, term_cfg)

        return {
            "success_bonus_weight": current_weight,
            "anneal_progress": self._progress,
        }

class BoxManipGoalThresholdCurriculum(ManagerTermBase):
    """Tightens position and orientation thresholds as success rate improves."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._cmd = env.command_manager.get_term(p["command_name"])

        self._initial_pos = p["initial_pos_threshold"]
        self._final_pos = p["final_pos_threshold"]
        self._initial_ori = p["initial_orientation_threshold"]
        self._final_ori = p["final_orientation_threshold"]
        self._success_threshold = p.get("success_rate_threshold", 0.7)
        self._ramp_rate = p.get("ramp_rate", 0.01)
        self._ema_alpha = p.get("ema_alpha", 0.05)

        self._difficulty = 0.0
        self._success_ema = 0.0

        self._cmd.cfg.pos_success_threshold = self._initial_pos
        self._cmd.cfg.orientation_success_threshold = self._initial_ori

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        command_name: str,
        initial_pos_threshold: float,
        final_pos_threshold: float,
        initial_orientation_threshold: float,
        final_orientation_threshold: float,
        success_rate_threshold: float = 0.7,
        ramp_rate: float = 0.01,
        ema_alpha: float = 0.05,
    ) -> dict[str, float]:
        if len(env_ids) == 0:
            return {}

        goal_dones = env.termination_manager.get_term("goal_reached")
        self._success_ema = (
            (1 - self._ema_alpha) * self._success_ema
            + self._ema_alpha * goal_dones[env_ids].float().mean().item()
        )

        if self._success_ema > self._success_threshold:
            self._difficulty = min(1.0, self._difficulty + self._ramp_rate)

        d = self._difficulty
        self._cmd.cfg.pos_success_threshold = self._initial_pos + (self._final_pos - self._initial_pos) * d
        self._cmd.cfg.orientation_success_threshold = self._initial_ori + (self._final_ori - self._initial_ori) * d

        return {
            "difficulty": self._difficulty,
            "success_ema": self._success_ema,
            "pos_threshold": self._cmd.cfg.pos_success_threshold,
            "ori_threshold": self._cmd.cfg.orientation_success_threshold,
        }

class BoxManipAnnealRewardWeight(ManagerTermBase):
    """Anneal a reward term's weight from initial value to target over training."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._reward_name = p["reward_name"]
        self._target_weight = p["target_weight"]
        self._initial_weight = env.reward_manager.get_term_cfg(self._reward_name).weight
        self._success_threshold = p.get("success_rate_threshold", 0.7)
        self._ramp_rate = p.get("ramp_rate", 0.01)
        self._ema_alpha = p.get("ema_alpha", 0.05)
        self._success_ema = 0.0
        self._progress = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        reward_name: str,
        target_weight: float,
        success_rate_threshold: float = 0.7,
        ramp_rate: float = 0.01,
        ema_alpha: float = 0.05,
    ) -> dict[str, float]:
        if len(env_ids) == 0:
            return {}

        goal_dones = env.termination_manager.get_term("goal_reached")
        self._success_ema = (
            (1 - self._ema_alpha) * self._success_ema
            + self._ema_alpha * goal_dones[env_ids].float().mean().item()
        )

        if self._success_ema > self._success_threshold:
            self._progress = min(1.0, self._progress + self._ramp_rate)

        w = self._initial_weight + (self._target_weight - self._initial_weight) * self._progress

        term_cfg = env.reward_manager.get_term_cfg(self._reward_name)
        term_cfg.weight = w
        env.reward_manager.set_term_cfg(self._reward_name, term_cfg)

        return {
            f"{self._reward_name}_weight": w,
            "anneal_progress": self._progress,
        }