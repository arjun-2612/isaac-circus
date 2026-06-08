# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils import math as math_utils
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

from circus_extensions.simulation.envs.mdp import TerrainBasedPose2dCommand, MotionDatasetSampler, HuskyBetaMultimodalJointPosCommand

def goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the box has been settled at the goal for the required duration."""
    cmd: TerrainBasedPose2dCommand = env.command_manager.get_term(command_name)
    return cmd.metrics["settled_steps"] >= cmd.cfg.required_settled_steps

def illegal_contact_in_flight(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold in flight mode."""
    # extract the used quantities (to enable type-hinting)
    command_term: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    in_contact =  torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    return in_contact * (command_term.mode == 1)  # only terminate if in flight mode

def very_unsafe_sw2_walk(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg,
    height_threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 when switching to walk while too far away from ground, else 0."""
    command_term: HuskyBetaMultimodalJointPosCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    height_scanner: RayCaster = env.scene[sensor_cfg.name]

    mode = command_term.mode          # (E,)
    prev_mode = command_term.prev_mode

    base_pos = asset.data.root_pos_w
    scans_w_from_base = base_pos.unsqueeze(1) - height_scanner.data.ray_hits_w # [E, N, 3]
    avg_z_scans_from_base = torch.mean(scans_w_from_base[:, :, 2], dim=1) # [E,]
    too_high_up = avg_z_scans_from_base > height_threshold

    switching_to_walk = (prev_mode == 1) & (mode == 0)
    bad_switch = switching_to_walk & too_high_up.squeeze(-1)
    return bad_switch

def max_body_pos_dev_wrt_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_deviation: float = 0.2,
    min_task_phase: float = 0.05,
    reference_name: str = "motion_references"
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)

    body_pos_b_ref = cmd.body_pos_w - cmd.root_pos_w.unsqueeze(1)
    target = math_utils.quat_rotate_inverse(cmd.root_quat_w.unsqueeze(1), body_pos_b_ref) # [N, B, 3]

    body_pos_b_meas = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    current = math_utils.quat_rotate_inverse(asset.data.root_quat_w.unsqueeze(1), body_pos_b_meas) # [N, B, 3]

    terminate = torch.max(torch.abs(current - target), dim=-1)[0] > max_deviation

    return torch.any(terminate, dim=1) * (cmd.task_phase > min_task_phase)


def root_pos_z_max_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_deviation: float = 0.2,
    min_task_phase: float = 0.05,
    reference_name: str = "motion_references"
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)
    terminate = torch.abs(cmd.root_pos_w[:, 2] - asset.data.root_pos_w[:, 2]) > max_deviation
    return terminate * (cmd.task_phase > min_task_phase)


def projected_gravity_max_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_deviation: float = 0.5,   # start looser; ~28.6° (curriculum it down later)
    min_task_phase: float = 0.05,
    reference_name: str = "motion_references",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd: MotionDatasetSampler = env.command_manager.get_term(reference_name)

    # world gravity (broadcasts fine if it's [3] or [E,3])
    g_w = asset.data.GRAVITY_VEC_W  # typically [0,0,-9.81] or [E,3]

    # project gravity into each root's body frame: [E,3]
    g_ref_b = math_utils.quat_rotate_inverse(cmd.root_quat_w,        g_w)
    g_cur_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w, g_w)

    # normalize to unit vectors before angle
    g_ref_b = F.normalize(g_ref_b, dim=-1)
    g_cur_b = F.normalize(g_cur_b, dim=-1)

    # angle between vectors
    cos_th = (g_ref_b * g_cur_b).sum(-1).clamp(-1.0, 1.0)  # [E]
    theta  = torch.acos(cos_th)                            # [E], radians

    exceeds = theta > max_deviation                        # [E] bool
    phase_ok = cmd.task_phase > min_task_phase             # [E] bool
    return exceeds & phase_ok


class feet_still_in_flight_at_end_of_reference(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env:  ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.contact_force_threshold = cfg.params["contact_force_threshold"]
        foot_body_name = cfg.params["foot_body_name"]
        self.reference_name = cfg.params["reference_name"]
        self.max_num_steps_in_flight = cfg.params["max_num_steps_in_flight"]
        self.num_steps_in_flight = torch.zeros(env.num_envs, device=self.device)
        self.foot_body_ids, foot_body_names = self.contact_sensor.find_bodies(foot_body_name)

    def reset(self, env_ids: Sequence[int] | None = None):
        self.num_steps_in_flight[env_ids] = 0

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        foot_body_name: str,
        contact_force_threshold: float,
        max_num_steps_in_flight: int,
        reference_name: str = "motion_references",
    ) -> torch.Tensor:
        feet_contact_state = (
            torch.sum(torch.square(self.contact_sensor.data.net_forces_w[:, self.foot_body_ids]), dim=-1)
            > self.contact_force_threshold**2
        )

        feet_contact_state = torch.any(feet_contact_state, dim=-1)
        flight_phase = torch.logical_not(feet_contact_state)
        cmd = env.command_manager.get_term(self.reference_name)

        self.num_steps_in_flight += torch.logical_and(cmd.task_phase >= 1.0, flight_phase)

        return self.num_steps_in_flight > self.max_num_steps_in_flight
    

def bodies_distance_below_threshold(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_distance_threshold: float = 0.05,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    num_envs = env.num_envs
    num_bodies = len(asset_cfg.body_ids)

    distance = torch.cdist(body_pos_w, body_pos_w, p=2)

    self_dist_mask = torch.eye(num_bodies, device=distance.device).unsqueeze(0).expand(num_envs, -1, -1)
    distance = distance + self_dist_mask * 100

    min_distances = torch.amin(distance, dim=(1,2))
    return min_distances < min_distance_threshold


def body_ground_penetration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_ground_penetration: float = 0.05,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    terminate = (body_pos_z < -max_ground_penetration).any(dim=1)

    return terminate


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 1.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size  # e.g., (8.0, 8.0)
        half_w = grid_width * 0.5
        half_l = grid_length * 0.5

        asset: RigidObject = env.scene[asset_cfg.name]
        pos = asset.data.root_pos_w[:, :2]  # (x,y) positions
        origin = env.scene.env_origins[:, :2]

        # position relative to env origin
        rel_pos = pos - origin
        out_x = torch.abs(rel_pos[:, 0]) > (half_l + distance_buffer)
        out_y = torch.abs(rel_pos[:, 1]) > (half_w + distance_buffer)

        return torch.logical_or(out_x, out_y)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
