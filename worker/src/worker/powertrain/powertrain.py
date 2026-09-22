from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from scipy.optimize import brentq

from .battery import Battery, StateOfCharge
from .motor import Motor
from .propeller import Propeller


@dataclass(frozen=True)
class PowertrainOperatingPoint:
    """Operating state of the electric powertrain."""

    thrust_n: float
    current_a: float
    terminal_voltage_v: float
    rpm: float
    electric_power_w: float


class ThrustProducer(Protocol):
    """Protocol for propulsion systems producing thrust."""

    def evaluate(
        self,
        airspeed_m_s: float,
        throttle: float,
        soc: StateOfCharge,
    ) -> PowertrainOperatingPoint:
        ...


class Powertrain:
    """Coupled battery-motor-propeller propulsion system."""

    def __init__(
        self,
        battery: Battery,
        motor: Motor,
        propeller: Propeller,
    ) -> None:
        self.battery = battery
        self.motor = motor
        self.propeller = propeller

    def evaluate(
        self,
        airspeed_m_s: float,
        throttle: float,
        soc: StateOfCharge,
    ) -> PowertrainOperatingPoint:
        """Find rotational equilibrium and compute operating metrics."""
        effective_throttle = max(0.0, min(1.0, throttle))
        if effective_throttle <= 0.0 or soc.value <= 0.0:
            v_term = self.battery.terminal_voltage(0.0, soc)
            return PowertrainOperatingPoint(
                thrust_n=0.0,
                current_a=0.0,
                terminal_voltage_v=v_term,
                rpm=0.0,
                electric_power_w=0.0,
            )

        rpm_eq = self._find_equilibrium_rpm(airspeed_m_s, effective_throttle, soc)
        return self._build_operating_point(rpm_eq, airspeed_m_s, effective_throttle, soc)

    def _find_equilibrium_rpm(
        self,
        airspeed_m_s: float,
        throttle: float,
        soc: StateOfCharge,
    ) -> float:
        v_ocv = self.battery.open_circuit_voltage(soc)
        rpm_max = throttle * v_ocv * self.motor.kv.value

        def torque_residual(rpm: float) -> float:
            current = self._estimate_current(rpm, throttle, soc)
            q_motor = self.motor.torque(current)
            q_prop = self.propeller.torque(rpm, airspeed_m_s)
            return q_motor - q_prop

        res_0 = torque_residual(1.0)
        res_max = torque_residual(rpm_max)

        if res_0 <= 0.0:
            return 0.0
        if res_max >= 0.0:
            return float(rpm_max)

        return float(brentq(torque_residual, 1.0, rpm_max, xtol=1.0))

    def _estimate_current(
        self,
        rpm: float,
        throttle: float,
        soc: StateOfCharge,
    ) -> float:
        v_ocv = self.battery.open_circuit_voltage(soc)
        v_emf = self.motor.back_emf(rpm)
        total_resistance = self.motor.resistance.ohms + (throttle**2) * self.battery.internal_resistance_ohms
        return max(0.0, (throttle * v_ocv - v_emf) / total_resistance)

    def _build_operating_point(
        self,
        rpm: float,
        airspeed_m_s: float,
        throttle: float,
        soc: StateOfCharge,
    ) -> PowertrainOperatingPoint:
        current = self._estimate_current(rpm, throttle, soc)
        v_term = self.battery.terminal_voltage(current, soc)
        thrust = self.propeller.thrust(rpm, airspeed_m_s)
        power_w = current * v_term

        return PowertrainOperatingPoint(
            thrust_n=thrust,
            current_a=current,
            terminal_voltage_v=v_term,
            rpm=rpm,
            electric_power_w=power_w,
        )
