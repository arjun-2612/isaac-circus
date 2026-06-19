# Edited from IsaacLab, used by Silicon Synapse Lab RL Research Team
# Author: Arjun Viswanathan

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from circus_extensions.simulation.envs.mdp import TerrainBasedPose2dCommand, MotionDatasetSampler, RMPCCommand, HuskyBetaMultimodalJointPosCommand
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from circus_extensions.simulation.terrains import TerrainImporter
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def mode_based_on_obstacle_proximity_reward(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    mm_command: str,
    wall_threshold: float = 0.2,
    pit_threshold: float = -0.2,
    sigmoid_scale: float = 5.0,
    sigmoid_offset: float = 0.3,
) -> torch.Tensor:
    """
    Reward obstacle proximity and mode being correct. Penalize for being wrong.
    
    Obstacle threat (0-1) = fraction of scan points hitting obstacles.
    As robot approaches obstacle, more points hit it, threat increases.
    """   
    height_scanner: RayCaster = env.scene[sensor_cfg.name]
    mm_cmd: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(mm_command)
    
    scans_w = height_scanner.data.ray_hits_w  # [E, N, 3]
    scan_heights = scans_w[:, :, 2]  # [E, N]
    num_points = scan_heights.shape[1]
    
    # Count points hitting obstacles
    wall_hits = (scan_heights > wall_threshold).float().sum(dim=1)
    pit_hits = (scan_heights < pit_threshold).float().sum(dim=1)
    obstacle_threat = (wall_hits + pit_hits) / num_points
    sigmoid_input = sigmoid_scale * (obstacle_threat - sigmoid_offset)
    y_u = torch.sigmoid(sigmoid_input)
    y_l = 1.0 - y_u
    
    in_walk = (mm_cmd.mode == 0).float()
    in_flight = (mm_cmd.mode == 1).float()

    # obstacle_proximity_reward = y_l * in_walk + y_u * in_flight
    obstacle_proximity_reward = (y_l - y_u) * in_walk + (y_u - y_l) * in_flight
    
    return obstacle_proximity_reward

def position_tracking_with_mode(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    mm_command: str,
    std: float, 
    pose_command: str,
    wall_threshold: float = 0.2,
    pit_threshold: float = -0.2,
    sigmoid_scale: float = 5.0,
    sigmoid_offset: float = 0.3,
) -> torch.Tensor:
    """
    Reward position tracking, modulated by obstacle proximity and mode.
    
    Obstacle threat (0-1) = fraction of scan points hitting obstacles.
    As robot approaches obstacle, more points hit it, threat increases.
    """
    command = env.command_manager.get_command(pose_command)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    pos_reward = torch.exp(-distance / std)
    
    height_scanner: RayCaster = env.scene[sensor_cfg.name]
    mm_cmd: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(mm_command)
    
    scans_w = height_scanner.data.ray_hits_w  # [E, N, 3]    
    scan_heights = scans_w[:, :, 2]  # [E, N]
    num_points = scan_heights.shape[1]
    
    # Count points hitting obstacles
    wall_hits = (scan_heights > wall_threshold).float().sum(dim=1)
    pit_hits = (scan_heights < pit_threshold).float().sum(dim=1)
    obstacle_threat = (wall_hits + pit_hits) / num_points
    sigmoid_input = sigmoid_scale * (obstacle_threat - sigmoid_offset)
    y_u = torch.sigmoid(sigmoid_input)
    y_l = 1.0 - y_u
    
    in_walk = (mm_cmd.mode == 0).float()
    in_flight = (mm_cmd.mode == 1).float()

    obstacle_proximity_reward = y_l * in_walk + y_u * in_flight
    reward = pos_reward * obstacle_proximity_reward
    
    return reward

def foot_target_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: RMPCCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [E, 4, 2]
    target_foot_pos_w = command.foot_targets.reshape(env.num_envs, 4, 2)

    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].clone()
    in_contact = (foot_forces[..., 2] > 1.0).bool()  # [E, 4]
    in_swing = command.swing_mask.bool() # [E, 4]
    contact_schedule_followed = (in_contact == (~in_swing)).float()  # [E, 4]

    per_foot_error = torch.norm(foot_pos_w - target_foot_pos_w, dim=-1)  # [E, 4]
    per_foot_reward = torch.exp(-per_foot_error / std) # [E, 4]
    # per_foot_reward = -torch.log(per_foot_error + std)  # [E, 4]
    per_foot_reward *= contact_schedule_followed  # [E, 4]

    reward = per_foot_reward.sum(dim=1) # [E] 
    return reward

def swing_target_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: RMPCCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]  # [E, 4, 3]
    target_foot_pos_w = command.swing_targets.reshape(env.num_envs, 4, 3)

    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].clone()
    in_contact = (foot_forces[..., 2] > 1.0).bool()  # [E, 4]
    in_swing = command.swing_mask.bool() # [E, 4]
    contact_schedule_followed = (in_contact == (~in_swing)).float()  # [E, 4]

    per_foot_error = torch.norm(foot_pos_w - target_foot_pos_w, dim=-1)  # [E, 4]
    per_foot_reward = torch.exp(-per_foot_error / std) # [E, 4]
    # per_foot_reward = -torch.log(per_foot_error + std)  # [E, 4]
    per_foot_reward *= contact_schedule_followed  # [E, 4]

    reward = per_foot_reward.sum(dim=1) # [E] 
    return reward

def foot_force_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    command: RMPCCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    error = torch.linalg.norm((contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] - command.foot_forces.reshape(env.num_envs, 4, 3)), dim=2)
    return torch.exp(-torch.sum(error, dim=1) / std)

def joint_target_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command: RMPCCommand = env.command_manager.get_term(command_name)

    error = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-error / std)

def total_tracking_reward(
    env: ManagerBasedRLEnv, 
    score_relative_weight: float = 1.0,
    reference_name: str = "motion_references",
) -> torch.Tensor:
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    cmd.compute_tracking_metrics()

    total_reward = -sum(value**2 for key, value in cmd.metrics.items() if "error_norm" in key)
    total_reward += score_relative_weight * sum(value for key, value in cmd.metrics.items() if "score" in key)

    return total_reward

def undesired_contacts_in_mode(env: ManagerBasedRLEnv, command_name: str, mode: int, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when the contact force on the sensor exceeds the force threshold in specific mode."""
    # extract the used quantities (to enable type-hinting)
    command_term: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1) * (command_term.mode == mode).float()

def bias(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def flight_mode_cost(
    env: ManagerBasedRLEnv, 
    command_name: str = "multimodal_command", 
) -> torch.Tensor:
    """Constant penalty for being in flight mode."""
    command: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    in_flight = (command.mode == 1).float()
    return in_flight

def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Bonus for actually reaching goal (XYZ)."""
    cmd: TerrainBasedPose2dCommand = env.command_manager.get_term(command_name)
    return (cmd.metrics["settled_steps"] > 0).float()

def unsafe_switch_to_walk(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg,
    height_threshold: float = 0.45,
    ) -> torch.Tensor:
    """1.0 when switching to walk while too close to the ground, else 0."""
    command_term: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    height_scanner: RayCaster = env.scene[sensor_cfg.name]

    mode = command_term.mode          # (num_envs,)
    prev_mode = command_term.prev_mode

    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w
    scans_w = height_scanner.data.ray_hits_w # [E, N, 3]
    scans_b = math_utils.quat_rotate_inverse(
        base_quat.unsqueeze(1),
        scans_w - base_pos.unsqueeze(1),
    )
    avg_z_scans = torch.mean(scans_b[:, :, 2], dim=1)
    height_from_base = torch.where(
        avg_z_scans > 0.0,
        torch.zeros_like(avg_z_scans),
        avg_z_scans
    )
    dist_below_threshold = torch.abs(height_from_base)
    close_to_ground = dist_below_threshold < height_threshold

    switching_to_walk = (prev_mode == 1) & (mode == 0)
    bad_switch = switching_to_walk & (~close_to_ground).squeeze(-1)
    return bad_switch.float()

def mode_persistence_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    mode = command.mode
    prev_mode = command.prev_mode

    changed_mode = (mode != prev_mode).float()
    return changed_mode

def no_vel_at_goal(env: ManagerBasedRLEnv, mm_command: str, pose_command: str, distance_threshold: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(pose_command)
    mm_cmd: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(mm_command)
    is_at_goal = (torch.norm(cmd[:, :3], dim=1) < distance_threshold).float()
    has_velocity = (torch.norm(mm_cmd.desired_velocity[:, :3], dim=1) > 0.1).float()
    return is_at_goal * has_velocity

def vz_command_in_walk_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)

    in_walk_mode = (command.mode == 0).float()

    z_velocity = torch.abs(command.desired_velocity[:, 2])
    penalty = z_velocity * in_walk_mode

    return penalty

def height_difference_flight_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pose_command_name: str,
    multimodal_command_name: str,
    height_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Reward flying when target is at a significantly different height.
    Penalize walking when target height differs.
    """
    mm_cmd: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(multimodal_command_name)
    cmd_b = env.command_manager.get_command(pose_command_name)
    asset = env.scene[asset_cfg.name]

    cmd_w = math_utils.quat_rotate(asset.data.root_quat_w, cmd_b[:, :3] + asset.data.root_pos_w)
    
    env_origin_z = env.scene.env_origins[:, 2]
    height_diff = torch.abs(cmd_w[:, 2] - env_origin_z)
    needs_flight = height_diff > height_threshold
    is_flying = (mm_cmd.mode == 1)
    
    reward = torch.where(
        needs_flight,
        torch.where(is_flying, 1.0, -1.0),
        torch.where(is_flying, -1.0, 1.0), # switch to positive reward instead of 0 later
    )
    return reward

def air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)

    if command_name == "base_velocity":
        cmd = torch.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(dim=1).expand(-1, 4)
        reward = torch.where(
            cmd > velocity_threshold,
            torch.where(t_max < mode_time, t_min, 0),
            stance_cmd_reward,
        )
    elif command_name == "base_vel_and_height":
        cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1).unsqueeze(dim=1).expand(-1, 4)
        reward = torch.where(
            cmd > velocity_threshold,
            torch.where(t_max < mode_time, t_min, 0),
            stance_cmd_reward,
        )
        h_cmd = env.command_manager.get_command(command_name)[:, 3]
        s = torch.clip(
            (0.47 - h_cmd) / (0.47 - 0.25), min=0.0, max=1.0
        )
        reward *= (1-s).unsqueeze(dim=1)

    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    std: float, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg, 
    std: float, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward tracking of base height commands (z) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)[:, 2]
    height_error = torch.linalg.norm((target - asset.data.root_pos_w[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-height_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    std: float, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_name: str = cfg.params.get("command_name", "base_velocity")
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        
        if command_name == "base_velocity":
            cmd = torch.norm(env.command_manager.get_command(command_name), dim=1)
            reward = torch.where(
                cmd > velocity_threshold, sync_reward * async_reward, 0.0
            )
        elif command_name == "base_vel_and_height":
            cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
            reward = torch.where(
                cmd > velocity_threshold, sync_reward * async_reward, 0.0
            )
            h_cmd = env.command_manager.get_command(command_name)[:, 3]
            s = torch.clip(
                (0.47 - h_cmd) / (0.47 - 0.25), min=0.0, max=1.0
            )
            reward *= (1-s)

        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    value = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(value, dim=1) / std)

    if command_name == "base_vel_and_height":
        h_cmd = env.command_manager.get_command(command_name)[:, 3]
        s = torch.clip(
            (0.47 - h_cmd) / (0.47 - 0.25), min=0.0, max=1.0
        )
        reward *= (1-s)

    return reward


##
# Regularization Penalties
##

def base_height_abs(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def base_rp_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    wx = torch.abs(asset.data.root_ang_vel_b[:, 0])
    wy = torch.abs(asset.data.root_ang_vel_b[:, 1])
    penalty = wx + wy
    return penalty


def base_z_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def foot_trip_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalizes vertical contacts to discourage stubbing toes on stairs and tripping

    This reward calculates the magnitude of the horizontal component of the contact
    force for each foot. A large horizontal force when vertical is small indicates a collision with a
    vertical surface (like a stair riser), which should be penalized.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (E, 4, 3)

    horizontal_force_magnitude = torch.linalg.norm(foot_forces[..., :2], dim=2)  # (E, 4, 1)
    vertical_force_magnitude = torch.abs(foot_forces[..., 2])
    is_collision = horizontal_force_magnitude > (4 * vertical_force_magnitude)

    return torch.any(is_collision, dim=1)  # (E, 1)


def foot_near_edge_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Computes a penalty for foot contacts near terrain edges.

    This function implements the r_clearance reward from the paper. It penalizes
    the robot for placing its feet close to edges, which is determined from
    the height scan data.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    foot_pos_xy_discretized = (
        (
            (asset.data.body_pos_w[:, asset_cfg.body_ids, :2] + terrain.cfg.terrain_generator.border_width)
            / terrain.cfg.terrain_generator.horizontal_scale
        )
        .round()
        .long()
    )
    foot_pos_xy_discretized[..., 0] = torch.clip(foot_pos_xy_discretized[..., 0], 0, terrain.edge_mask.shape[0] - 1)
    foot_pos_xy_discretized[..., 1] = torch.clip(foot_pos_xy_discretized[..., 1], 0, terrain.edge_mask.shape[1] - 1)
    foot_at_edge = terrain.edge_mask[foot_pos_xy_discretized[..., 0], foot_pos_xy_discretized[..., 1]]

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > env.step_dt
    is_edge = foot_at_edge & is_contact #& (terrain.terrain_types.unsqueeze(1) > 3)

    return torch.sum(is_edge, dim=-1)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def arm_action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output for just arm joints"""
    return torch.linalg.norm((env.action_manager.action[:, 12:] - env.action_manager.prev_action[:, 12:]), dim=1)


def action_smoothness_penalty_hlp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output for only velocities"""
    return torch.linalg.norm((env.action_manager.action[:, :4] - env.action_manager.prev_action[:, :4]), dim=1)


def air_time_variance_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    penalty = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    
    if command_name == "base_vel_and_height":
        h_cmd = env.command_manager.get_command(command_name)[:, 3]
        s = torch.clip(
            (0.47 - h_cmd) / (0.47 - 0.25), min=0.0, max=1.0
        )
        penalty *= (1-s)
    
    return penalty


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_ang_vel_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def power_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sharpness: float = 0.02, threshold: float = 40.0,
) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), 
        dim=1,
    )
    sigin = sharpness * (power - threshold)
    return torch.sigmoid(sigin)


def joint_position_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    stand_still_scale: float, 
    velocity_threshold: float, 
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(cmd > velocity_threshold, reward, stand_still_scale * reward)


def joint_dev_from_default(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return penalty


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def stomping_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize stomping while traversing the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.body_lin_acc_w[:, asset_cfg.body_ids, 2]), dim=1)


def high_contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 50.0,
) -> torch.Tensor:
    """
    Penalizes high contact forces directly. Simple and effective complement
    to velocity-based penalties. Encourages the robot to land softly.
    
    Args:
        max_force: normalize forces by this value to keep penalty scale reasonable
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_mag = torch.norm(contact_forces, dim=-1) * touchdown # (num_envs, num_feet)
    
    # Penalize forces above a soft threshold using squared penalty
    penalty = torch.sum(torch.square(contact_mag / max_force), dim=-1)
    return penalty