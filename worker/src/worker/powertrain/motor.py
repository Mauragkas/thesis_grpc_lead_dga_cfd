from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MotorKV:
    """Motor velocity constant in RPM / Volt."""

    value: float

    def __post_init__(self) -> None:
        if self.value <= 0.0:
            raise ValueError("KV must be positive")


@dataclass(frozen=True)
class MotorResistance:
    """Motor internal phase-to-phase resistance in Ohms."""

    ohms: float

    def __post_init__(self) -> None:
        if self.ohms <= 0.0:
            raise ValueError("Resistance must be positive")


class Motor:
    """Brushless DC (BLDC) motor model."""

    def __init__(
        self,
        kv: MotorKV,
        resistance: MotorResistance,
        no_load_current_a: float = 0.8,
        max_current_a: float = 40.0,
    ) -> None:
        self.kv = kv
        self.resistance = resistance
        self.no_load_current_a = max(0.0, no_load_current_a)
        self.max_current_a = max_current_a
        # Torque constant: Kt = 30 / (pi * Kv) in N*m / A
        self._torque_constant = 30.0 / (np.pi * self.kv.value)

    def back_emf(self, rpm: float) -> float:
        """Calculate back-electromotive force voltage at given rotational speed."""
        return max(0.0, float(rpm / self.kv.value))

    def current(self, voltage: float, rpm: float) -> float:
        """Calculate motor current drawn at a given terminal voltage and RPM."""
        emf = self.back_emf(rpm)
        effective_v = voltage - emf
        if effective_v <= 0.0:
            return 0.0
        return float(effective_v / self.resistance.ohms)

    def torque(self, current_amps: float) -> float:
        """Calculate electromagnetic shaft torque in N*m."""
        torque_producing_current = current_amps - self.no_load_current_a
        if torque_producing_current <= 0.0:
            return 0.0
        return float(self._torque_constant * torque_producing_current)

    def is_overcurrent(self, current_amps: float) -> bool:
        """Check if drawn current exceeds maximum allowable continuous limit."""
        return current_amps > self.max_current_a
