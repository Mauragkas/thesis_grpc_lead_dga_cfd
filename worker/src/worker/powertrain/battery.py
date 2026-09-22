from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capacity:
    """Battery capacity value object in Ampere-hours (Ah)."""

    amp_hours: float

    def __post_init__(self) -> None:
        if self.amp_hours <= 0.0:
            raise ValueError("Capacity must be positive")


@dataclass(frozen=True)
class StateOfCharge:
    """Normalized state of charge [0.0, 1.0]."""

    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("State of charge must be between 0 and 1")


class Battery:
    """Equivalent circuit model of a multi-cell Lithium-Polymer (LiPo) battery."""

    # Nominal LiPo cell voltage limits
    V_CELL_MIN: float = 3.3
    V_CELL_MAX: float = 4.2

    def __init__(
        self,
        cell_count: int,
        capacity: Capacity,
        internal_resistance_ohms: float = 0.03,
    ) -> None:
        if cell_count <= 0:
            raise ValueError("cell_count must be positive")
        self.cell_count = cell_count
        self.capacity = capacity
        self.internal_resistance_ohms = max(0.0, internal_resistance_ohms)

    def open_circuit_voltage(self, soc: StateOfCharge) -> float:
        """Calculate open circuit voltage (Voc) based on state of charge."""
        cell_v = self.V_CELL_MIN + soc.value * (self.V_CELL_MAX - self.V_CELL_MIN)
        return float(cell_v * self.cell_count)

    def terminal_voltage(self, current_amps: float, soc: StateOfCharge) -> float:
        """Calculate terminal voltage under load current considering internal resistance sag."""
        voc = self.open_circuit_voltage(soc)
        sag = current_amps * self.internal_resistance_ohms
        return max(0.0, float(voc - sag))

    def discharge(
        self,
        current_amps: float,
        duration_seconds: float,
        current_soc: StateOfCharge,
    ) -> StateOfCharge:
        """Compute the updated state of charge after drawing current for a duration."""
        used_ah = (current_amps * duration_seconds) / 3600.0
        remaining_ah = current_soc.value * self.capacity.amp_hours - used_ah
        new_soc_value = max(0.0, min(1.0, remaining_ah / self.capacity.amp_hours))
        return StateOfCharge(new_soc_value)
