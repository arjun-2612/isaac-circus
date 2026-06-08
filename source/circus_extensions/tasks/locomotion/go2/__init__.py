# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Unitree-Go2-Custom-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_env_cfg:UnitreeGo2CustomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Custom-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_env_cfg:UnitreeGo2CustomEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Fast-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_env_cfg:UnitreeGo2FastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FastPPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Fast-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_env_cfg:UnitreeGo2FastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FastPPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-StairsWaves-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_waves_env_cfg:UnitreeGo2StairsWavesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PerceptivePPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-StairsWaves-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_waves_env_cfg:UnitreeGo2StairsWavesEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PerceptivePPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-MotionMimic-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motion_mimic_env_cfg:UnitreeGo2MotionMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MotionMimicPPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-MotionMimic-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motion_mimic_env_cfg:UnitreeGo2MotionMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MotionMimicPPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motion_mimic_env_cfg:UnitreeGo2MotionMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2MotionMimicPPORunnerCfg"
    },
)
