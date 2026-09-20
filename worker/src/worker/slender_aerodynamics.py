from __future__ import annotations

import numpy as np

# Standard atmospheric constants (sea level)
RHO_AIR: float = 1.225
MU_AIR: float = 1.81e-5


def get_fuselage_wetted_area(
    length: float,
    max_diameter: float,
    nose_ratio: float = 0.25,
    tail_ratio: float = 0.35,
) -> float:
    """Calculate fuselage wetted surface area in m^2.

    Approximates an axisymmetric body with nose cone/ogive, cylindrical midsection,
    and conical tail taper based on the formulation in Atlas-Upat.

    Args:
        length: Total fuselage length in meters.
        max_diameter: Maximum fuselage diameter in meters.
        nose_ratio: Fraction of length dedicated to nose.
        tail_ratio: Fraction of length dedicated to tail.

    Returns:
        Fuselage wetted surface area in m^2.
    """
    if length <= 0.0 or max_diameter <= 0.0:
        return 0.0

    feff = max(length / max_diameter, 1.0)
    ff_nose = max((length * nose_ratio) / max_diameter, 0.5)
    ff_tail = max((length * tail_ratio) / max_diameter, 0.5)

    taper_factor = 1.0 - 0.5 * (nose_ratio + tail_ratio)
    swet_base = np.pi * max_diameter * length * taper_factor
    shape_correction = 1.0 + 3.0 / (feff**1.9) + 0.03443 / ff_nose + 0.03443 / ff_tail
    return float(swet_base * min(shape_correction, 1.5))


def get_fuselage_drag_coefficient(
    length: float,
    max_diameter: float,
    s_ref: float,
    velocity: float,
    nose_ratio: float = 0.25,
    tail_ratio: float = 0.35,
    rho: float = RHO_AIR,
    mu: float = MU_AIR,
) -> tuple[float, float]:
    """Calculate fuselage parasite drag coefficient (referenced to wing S_ref) and wetted area.

    Uses Schlichting turbulent flat-plate skin friction and Torenbeek form factor
    from Atlas-Upat (Atlas/aero/slender_aerodynamics.py).

    Args:
        length: Fuselage length in meters.
        max_diameter: Fuselage maximum diameter in meters.
        s_ref: Wing reference area in m^2.
        velocity: Flight velocity in m/s.
        nose_ratio: Nose section length ratio.
        tail_ratio: Tail section length ratio.
        rho: Air density in kg/m^3.
        mu: Dynamic viscosity of air in Pa*s.

    Returns:
        tuple of (cd_fuselage referenced to s_ref, wetted_area in m^2).
    """
    if length <= 0.0 or max_diameter <= 0.0 or s_ref <= 0.0 or abs(velocity) < 0.1:
        return 0.0, 0.0

    re_fuse = (rho * abs(velocity) * length) / mu
    if re_fuse < 1e3:
        cf = 1.328 / np.sqrt(max(re_fuse, 1.0))
    else:
        cf = 0.455 / ((np.log10(re_fuse)) ** 2.58)

    feff = max(length / max_diameter, 1.0)
    # Torenbeek fuselage form factor: 1 + 60/(feff^3) + feff/400
    form_factor = 1.0 + (60.0 / (feff**3)) + (feff / 400.0)

    swet = get_fuselage_wetted_area(
        length=length,
        max_diameter=max_diameter,
        nose_ratio=nose_ratio,
        tail_ratio=tail_ratio,
    )

    cd_fuse = (cf * form_factor * swet) / s_ref
    return float(cd_fuse), float(swet)
