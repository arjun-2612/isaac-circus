from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import math

from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions
from isaaclab.actuators import IdealPDActuator

if TYPE_CHECKING:
    from .actuator_cfg import PowerLimitedPDActuatorCfg, DelayedPDActuatorCfg, DynamixelActuatorCfg

class DelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        if self.cfg.motor_strength_range is not None:
            self._motor_strength_ranges = self.cfg.motor_strength_range
        else:
            self._motor_strength_ranges = (1.0, 1.0)

        self._motor_strengths = torch.empty((len(self.computed_effort), 1), device=self._device)
        self._current_motor_strengths = self._motor_strengths.uniform_(*self._motor_strength_ranges)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

        self._current_motor_strengths[env_ids] = self._motor_strengths.uniform_(*self._motor_strength_ranges)[env_ids]

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        
        # compute the joint efforts
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.computed_effort *= self._current_motor_strengths
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None 
        control_action.joint_velocities = None

        return control_action
    

class DynamixelActuator(DelayedPDActuator):
    """Dynamixel actuator: adds trapezoidal profile to the DelayedPDActuator."""

    cfg: DynamixelActuatorCfg

    def __init__(self, cfg: DynamixelActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.encoder_resolution = 4096
        self.pwm_max = cfg.goal_pwm

        self.ticks_per_rad = self.encoder_resolution / (2 * math.pi)
        self.back_emf = self.effort_limit / self.velocity_limit  # back EMF constant (N*m/(rad/s))
        self.decimation = 4
        self.interp_step = 0
        self.q_cmd_prev = None
        self.q_cmd_next = None 

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        raw_target = control_action.joint_positions

        # Detect new policy action (target changed) → reset interpolation
        if self.q_cmd_next is None:
            # First call: initialize both to current joint pos
            self.q_cmd_prev = joint_pos.clone()
            self.q_cmd_next = raw_target.clone()
            self._interp_step = 0
        elif not torch.allclose(raw_target, self.q_cmd_next, atol=1e-6):
            # New policy output arrived
            self.q_cmd_prev = self.q_cmd_next.clone()
            self.q_cmd_next = raw_target.clone()
            self._interp_step = 0

        # Interpolate: matches hardware alpha = (interp_step + 1) / decimation
        alpha = (self._interp_step + 1) / self.decimation
        alpha = min(alpha, 1.0)
        interpolated_target = self.q_cmd_prev + alpha * (self.q_cmd_next - self.q_cmd_prev)
        self._interp_step += 1

        target_pos = self.positions_delay_buffer.compute(interpolated_target)
        pos_error_rad = target_pos - joint_pos
        pos_error_ticks = pos_error_rad * self.ticks_per_rad
        vel_ticks = joint_vel * self.ticks_per_rad

        p_term = self.stiffness * pos_error_ticks
        d_term = self.damping * (-vel_ticks)

        pwm = p_term + d_term
        pwm = torch.clip(pwm, -self.pwm_max, self.pwm_max)

        motor_torque = self.effort_limit * (pwm / self.pwm_max) - (self.back_emf * joint_vel)
        torque = self._clip_effort(motor_torque)

        self.computed_effort = torque
        self.applied_effort = torque

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


class PowerLimitedPDActuator(DelayedPDActuator):
    """Power limited PD actuator.

    This class extends the :class:`DelayedPDActuator` class by adding velocity-dependent torque limits to the actuator.
    The torque limits are applied by an equation describing the relationship between the joint angle
    and the maximum output torque. The parameters for this are given in the configration class.

    """

    def __init__(self, cfg: PowerLimitedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)

        max_torque_limit = torch.clip(
            self.cfg.m1*joint_vel + self.cfg.b1,
            max=self.cfg.max_torque,
            min=0.0,
        )
        min_torque_limit = torch.clip(
            self.cfg.m2*joint_vel + self.cfg.b2,
            min=self.cfg.min_torque,
            max=0.0,
        )

        control_action.joint_efforts = torch.clip(
            control_action.joint_efforts,
            min=min_torque_limit,
            max=max_torque_limit,
        )
        self.applied_effort = control_action.joint_efforts

        return control_action