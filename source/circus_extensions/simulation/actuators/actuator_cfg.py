from isaaclab.utils import configclass
from dataclasses import MISSING
from .actuator_pd import PowerLimitedPDActuator, DelayedPDActuator, DynamixelActuator
from isaaclab.actuators import IdealPDActuatorCfg

@configclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedPDActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    motor_strength_range: tuple[float, float] = (1.0, 1.0)


@configclass
class DynamixelActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a dynamixel PD actuator."""
    class_type: type = DynamixelActuator 
    
    goal_pwm: float = MISSING
    """The goal PWM value of the actuator. This is the PWM value that corresponds to the maximum torque that the actuator can produce. This value is used to calculate the maximum torque that the actuator can produce at a given speed."""


@configclass
class PowerLimitedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a power-limited PD actuator."""
    class_type: type = PowerLimitedPDActuator 

    m1: float = MISSING 
    m2: float = MISSING 
    b1: float = MISSING 
    b2: float = MISSING 
    max_torque: float = MISSING 
    min_torque: float = MISSING