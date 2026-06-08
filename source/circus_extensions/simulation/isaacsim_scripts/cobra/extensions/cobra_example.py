'''
Cobra snake robot example in IsaacSim 4.5.0 
Date created: 05/29/25
Date last modified: 7/25/25
'''
import os 

from typing import Optional
import numpy as np
import omni
import omni.kit.commands
import carb
import omni.appwindow  # Contains handle to keyboard
from scipy.spatial.transform import Rotation as R
# import pandas as pd

from isaacsim.core.utils.rotations import quat_to_rot_matrix, quat_to_euler_angles
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.examples.interactive.base_sample import BaseSample

class CobraPolicy(PolicyController):
    """Cobra robot"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "cobra",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        run_name = "2025-07-24_22-47-31" 
        self.data_dir = "/home/arjun/Documents/sslab_isaaclab/source/sslab_extensions/simulation/isaacsim_scripts/cobra/data"

        # CSV file paths
        # Buffers to store simulation data before saving to CSV
        self.data_buffers = {
            "ang_vel": [],
            "orientation": [],
            "joint_observed": [],
            "joint_velocities": [],
            "joint_commanded": [],
        }

        self.joint_names = [ # change this
            'bl_hf_joint', 'br_hf_joint', 'fl_hf_joint', 'fr_hf_joint',
            'bl_hs_joint', 'br_hs_joint', 'fl_hs_joint', 'fr_hs_joint',
            'bl_k_joint', 'br_k_joint', 'fl_k_joint', 'fr_k_joint',
        ]

        root_folder = "/home/arjun/Documents/sslab_isaaclab"

        if usd_path == None:
            usd_path = f"{root_folder}/source/sslab_extensions/simulation/models/cobra/cobra/cobra.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            f"{root_folder}/logs/rsl_rl/cobra_rl/{run_name}/exported/policy.pt",
            f"{root_folder}/logs/rsl_rl/cobra_rl/{run_name}/params/env.yaml"
        )
        self._action_scale = 0.25
        self._default_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.698132, 0.698132, 0.698132, 0.698132, 0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4])
        self._previous_action = np.zeros(16)
        self._policy_counter = 0
        self._decimation = 4

        self._noise_std = {
            "lin_vel":        0.05,   # m/s
            "ang_vel":        0.01,   # rad/s
            "gravity":        0.01,  # unit vector noise
            "command_lin":    0.05,   # m/s
            "command_ang":    0.01,   # rad/s
            "joint_pos":      0.005,  # rad
            "joint_vel":      0.01,   # rad/s
            "prev_action":    0.0,   # same units as your actions
        }
        # pre-build a single-array of stds aligned with obs ordering
        self._obs_noise_stds = np.concatenate([
            np.full(3, self._noise_std["lin_vel"]),       # obs[0:3]
            np.full(3, self._noise_std["ang_vel"]),       # obs[3:6]
            np.full(3, self._noise_std["gravity"]),       # obs[6:9]
            np.array([
                self._noise_std["command_lin"],            # obs[ 9]
                self._noise_std["command_lin"],            # obs[10]
                self._noise_std["command_ang"],            # obs[11]
            ]),
            np.full(16, self._noise_std["joint_pos"]),    # obs[12:28]
            np.full(16, self._noise_std["joint_vel"]),    # obs[28:44]
            np.full(12, self._noise_std["prev_action"]),  # obs[44:56]
        ])

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        xyz = quat_to_euler_angles(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(56)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:28] = current_joint_pos - self._default_pos
        obs[28:44] = current_joint_vel
        # Previous Action
        obs[44:56] = self._previous_action[0:12]

        self.data_buffers['ang_vel'].append(ang_vel_b.tolist())
        self.data_buffers['orientation'].append(xyz.tolist())
        self.data_buffers['joint_observed'].append(current_joint_pos.tolist())
        self.data_buffers['joint_velocities'].append(current_joint_vel.tolist())
        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if command[0] == 99.0:  # Save to ZIP file command
            self.save_to_zip()
        else:
            if self._policy_counter % self._decimation == 0:
                obs = self._compute_observation(command)
                self.action = self._compute_action(obs)
                self.action = np.append(self.action, -1 * self.action[8:12])
                self._previous_action = self.action.copy()

            joint_pos = self._default_pos + (self.action * self._action_scale)
            self.data_buffers['joint_commanded'].append(joint_pos)

            action = ArticulationAction(joint_positions=joint_pos)
            self.robot.apply_action(action)

            self._policy_counter += 1

    def save_to_zip(self):
        # Save the current data_buffers to an npz file
        os.makedirs(self.data_dir, exist_ok=True)
        data_to_save = {}
        for key, buffer in self.data_buffers.items():
            data_to_save[key] = np.array(buffer)
        np.savez(os.path.join(self.data_dir, "sim_data.npz"), **data_to_save)

class CobraExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0
        self._world_settings["rendering_dt"] = 10.0 / 200.0
        self._base_command = [0.0, 0.0, 0.0]

        self.magnitude = 1.25
        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [self.magnitude, 0.0, 0.0],
            "UP": [self.magnitude, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-self.magnitude, 0.0, 0.0],
            "DOWN": [-self.magnitude, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -self.magnitude, 0.0],
            "RIGHT": [0.0, -self.magnitude, 0.0],
            # right command
            "NUMPAD_4": [0.0, self.magnitude, 0.0],
            "LEFT": [0.0, self.magnitude, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, self.magnitude],
            "N": [0.0, 0.0, self.magnitude],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -self.magnitude],
            "M": [0.0, 0.0, -self.magnitude],
            # Save to ZIP file
            "Z": [99.0, 0.0, 0.0],
        }

    def setup_scene(self) -> None:
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        self.sim_robot = CobraPolicy(
            prim_path="/World/Cobra",
            name="Cobra",
            position=R.from_euler('xyz', np.array([0.0, 0.0, 0.0])).as_quat(),
        )
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._physics_ready = False
        self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        await self._world.play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.sim_robot.forward(step_size, self._base_command)
        else:
            self._physics_ready = True
            self.sim_robot.initialize()
            self.sim_robot.post_reset()
            self.sim_robot.robot.set_joints_default_state(self.sim_robot.default_pos)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""

        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.sim_robot:
            self._physics_ready = False

    def world_cleanup(self):
        self._event_timer_callback = None
        if self._world.physics_callback_exists("physics_step"):
            self._world.remove_physics_callback("physics_step")
