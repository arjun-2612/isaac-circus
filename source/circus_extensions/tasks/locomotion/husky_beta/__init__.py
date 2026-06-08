# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, stairs_waves_env_cfg, dtc_env_cfg, flight_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="HuskyBeta-DTC-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dtc_env_cfg.HuskyBetaDTCEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaDTCPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-DTC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dtc_env_cfg.HuskyBetaDTCEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaDTCPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-Flat-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HuskyBetaFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaRLPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HuskyBetaFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaRLPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-Flight-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flight_env_cfg.HuskyBetaFlightEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaFlightPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-Flight-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flight_env_cfg.HuskyBetaFlightEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaFlightPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-StairsWaves-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stairs_waves_env_cfg.HuskyBetaStairsWavesEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaRLPPORunnerCfg",
    },
)

gym.register(
    id="HuskyBeta-StairsWaves-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stairs_waves_env_cfg.HuskyBetaStairsWavesEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HuskyBetaRLPPORunnerCfg",
    },
)