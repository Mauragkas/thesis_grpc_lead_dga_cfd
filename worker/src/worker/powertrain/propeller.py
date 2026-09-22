from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PropellerDimension:
    """Propeller geometric dimensions."""

    diameter_m: float
    pitch_m: float

    def __post_init__(self) -> None:
        if self.diameter_m <= 0.0:
            raise ValueError("Diameter must be positive")
        if self.pitch_m <= 0.0:
            raise ValueError("Pitch must be positive")

    @property
    def pitch_to_diameter(self) -> float:
        return self.pitch_m / self.diameter_m


class Propeller:
    """Analytical propeller model based on standard advance-ratio aerodynamics."""

    def __init__(
        self,
        dimensions: PropellerDimension,
        rho_air: float = 1.225,
    ) -> None:
        self.dimensions = dimensions
        self.rho_air = rho_air
        p_d = dimensions.pitch_to_diameter
        # Coefficients parameterized by pitch-to-diameter ratio
        self._ct0 = 0.11 * p_d
        self._j_max = 1.25 * p_d
        self._cp0 = 0.045 * (p_d**1.2)

    def advance_ratio(self, rpm: float, airspeed_m_s: float) -> float:
        """Calculate non-dimensional advance ratio J = V / (n * D)."""
        revs_per_sec = rpm / 60.0
        if revs_per_sec <= 0.0:
            return 0.0
        return float(airspeed_m_s / (revs_per_sec * self.dimensions.diameter_m))

    def thrust_coefficient(self, j: float) -> float:
        """Thrust coefficient Ct(J)."""
        if j >= self._j_max:
            return 0.0
        return max(0.0, float(self._ct0 * (1.0 - (j / self._j_max) ** 1.5)))

    def power_coefficient(self, j: float) -> float:
        """Power coefficient Cp(J)."""
        if j >= self._j_max:
            return 0.005
        return max(0.005, float(self._cp0 * (1.0 - 0.3 * (j / self._j_max))))

    def thrust(self, rpm: float, airspeed_m_s: float) -> float:
        """Calculate generated thrust in Newtons."""
        revs_per_sec = rpm / 60.0
        if revs_per_sec <= 0.0:
            return 0.0
        j = self.advance_ratio(rpm, airspeed_m_s)
        ct = self.thrust_coefficient(j)
        d = self.dimensions.diameter_m
        return float(ct * self.rho_air * (revs_per_sec**2) * (d**4))

    def torque(self, rpm: float, airspeed_m_s: float) -> float:
        """Calculate required shaft torque in N*m."""
        revs_per_sec = rpm / 60.0
        if revs_per_sec <= 0.0:
            return 0.0
        j = self.advance_ratio(rpm, airspeed_m_s)
        cp = self.power_coefficient(j)
        d = self.dimensions.diameter_m
        power_shaft = cp * self.rho_air * (revs_per_sec**3) * (d**5)
        omega = 2.0 * np.pi * revs_per_sec
        return float(power_shaft / omega)
