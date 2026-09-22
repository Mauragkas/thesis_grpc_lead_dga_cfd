from __future__ import annotations

import numpy as np


class AeroPolar:
    """Pre-computed aerodynamic polar splines for rapid trajectory simulation."""

    def __init__(
        self,
        alpha_deg: np.ndarray,
        cl: np.ndarray,
        cd: np.ndarray,
        cm: np.ndarray,
    ) -> None:
        self.alpha_deg = np.asarray(alpha_deg, dtype=float)
        self.cl = np.asarray(cl, dtype=float)
        self.cd = np.asarray(cd, dtype=float)
        self.cm = np.asarray(cm, dtype=float)

    @property
    def cl_max(self) -> float:
        return float(np.max(self.cl))

    @property
    def cl_min(self) -> float:
        return float(np.min(self.cl))

    def lift_coefficient(self, alpha_deg: float) -> float:
        return float(np.interp(alpha_deg, self.alpha_deg, self.cl))

    def drag_coefficient(self, alpha_deg: float) -> float:
        return float(np.interp(alpha_deg, self.alpha_deg, self.cd))

    def moment_coefficient(self, alpha_deg: float) -> float:
        return float(np.interp(alpha_deg, self.alpha_deg, self.cm))

    def stall_speed(
        self,
        weight_n: float,
        s_ref_m2: float,
        rho_air: float = 1.225,
    ) -> float:
        """Calculate stall velocity at Cl_max."""
        if self.cl_max <= 0.0 or s_ref_m2 <= 0.0:
            return 999.0
        return float(np.sqrt((2.0 * weight_n) / (rho_air * s_ref_m2 * self.cl_max)))

    def trim_alpha_for_cl(self, cl_target: float) -> float | None:
        """Interpolate trim angle of attack to achieve target CL."""
        if cl_target < self.cl_min or cl_target > self.cl_max:
            return None
        return float(np.interp(cl_target, self.cl, self.alpha_deg))
