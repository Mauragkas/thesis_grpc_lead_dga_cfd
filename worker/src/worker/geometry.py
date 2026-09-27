from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

try:
    from .config import BASELINE, GeneBound
    from .slender_aerodynamics import get_fuselage_wetted_area
except (ImportError, ValueError):
    from config import BASELINE, GeneBound
    from slender_aerodynamics import get_fuselage_wetted_area


@dataclass(frozen=True)
class Fuselage:
    """Pure geometry of the parametrized fuselage body."""

    length: float
    nose_ratio: float
    tail_ratio: float
    max_diameter: float

    @property
    def nose_length(self) -> float:
        return self.length * self.nose_ratio

    @property
    def tail_length(self) -> float:
        return self.length * self.tail_ratio

    @property
    def mid_length(self) -> float:
        return self.length - self.nose_length - self.tail_length

    @property
    def r_max(self) -> float:
        return self.max_diameter / 2.0

    @property
    def volume_mm3(self) -> float:
        r_max = self.r_max
        v_nose = (2.0 / 3.0) * np.pi * (r_max**2) * self.nose_length
        v_mid = np.pi * (r_max**2) * self.mid_length
        r_tail_tip = r_max * (1.0 - 0.85)
        v_tail = (
            (1.0 / 3.0)
            * np.pi
            * self.tail_length
            * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)
        )
        return float(v_nose + v_mid + v_tail)


@dataclass(frozen=True)
class WingGeometry:
    """Planform geometry of the main wing."""

    span: float
    root_chord: float
    tip_chord: float

    @property
    def wingspan(self) -> float:
        return 2.0 * self.span

    @property
    def planform_area(self) -> float:
        return 2.0 * self.span * (self.root_chord + self.tip_chord) / 2.0

    @property
    def mac(self) -> float:
        """Mean Aerodynamic Chord for a trapezoidal wing."""
        cr = self.root_chord
        ct = self.tip_chord
        return (2.0 / 3.0) * (cr + ct - (cr * ct) / max(cr + ct, 1e-6))


@dataclass(frozen=True)
class TailVolumeTargets:
    """Target horizontal and vertical tail volume coefficients (Stan Hall / Pazmany method)."""

    vh: float
    vv: float

    def __post_init__(self) -> None:
        if self.vh <= 0:
            raise ValueError("vh must be positive")
        if self.vv <= 0:
            raise ValueError("vv must be positive")


def calculate_wing_geometry(span: float, root_chord: float, tip_chord: float) -> WingGeometry:
    return WingGeometry(span=span, root_chord=root_chord, tip_chord=tip_chord)


def calculate_tail_planform_areas(
    wing: WingGeometry,
    arm_h: float,
    arm_v: float,
    targets: TailVolumeTargets,
) -> tuple[float, float]:
    """Calculate required horizontal and vertical tail planform areas using volume coefficients.

    VH = SH * LH / (SW * mac)  =>  SH = VH * SW * mac / LH
    VV = SV * LV / (SW * b)    =>  SV = VV * SW * b / LV
    """
    lh = max(arm_h, 1e-4)
    lv = max(arm_v, 1e-4)
    s_h = targets.vh * wing.planform_area * wing.mac / lh
    s_v = targets.vv * wing.planform_area * wing.wingspan / lv
    return s_h, s_v


def fuselage_from_params(params: dict[str, float]) -> Fuselage:
    return Fuselage(
        length=params["fuse_length"],
        nose_ratio=params["nose_ratio"],
        tail_ratio=params["tail_ratio"],
        max_diameter=params["fuse_max_diam"],
    )


def fuselage_volume_mm3(params: dict[str, float]) -> float:
    return fuselage_from_params(params).volume_mm3


def wing_panel_volume(span: float, root_c: float, tip_c: float, t_ratio: float) -> float:
    return 0.685 * t_ratio * (span / 3.0) * (root_c**2 + root_c * tip_c + tip_c**2)


def calculate_mass_breakdown(
    params: dict[str, float],
    m_avionics_kg: float = 0.080,
    areal_density_fuse: float = 1.20,
    areal_density_wing: float = 1.00,
) -> dict[str, float]:
    """Compute component mass breakdown based on Atlas-Upat structural sizing.

    Accounts for 3D printed shells, internal infill/spars, and fixed avionics
    (motor, ESC, servos, receiver, battery).
    """
    fuse_len_m = params["fuse_length"] * 1e-3
    fuse_diam_m = params["fuse_max_diam"] * 1e-3
    swet_fuse = get_fuselage_wetted_area(
        length=fuse_len_m,
        max_diameter=fuse_diam_m,
        nose_ratio=params.get("nose_ratio", 0.25),
        tail_ratio=params.get("tail_ratio", 0.35),
    )
    m_fuse = swet_fuse * areal_density_fuse

    # Wing planform area (both sides) and wetted area
    s_wing_m2 = (
        2.0
        * params["wing_span"]
        * (params["wing_root_chord"] + params["wing_tip_chord"])
        / 2.0
        * 1e-6
    )
    swet_wing = 2.05 * s_wing_m2
    m_wing = swet_wing * areal_density_wing

    # Empennage (Horizontal + Vertical Stabilizers)
    s_h_stab = (
        2.0
        * params["h_stab_span"]
        * (params["h_stab_root"] + params["h_stab_tip"])
        / 2.0
        * 1e-6
    )
    s_v_stab = (
        params["v_stab_height"]
        * (params["v_stab_root"] + params["v_stab_tip"])
        / 2.0
        * 1e-6
    )
    swet_tail = 2.05 * (s_h_stab + s_v_stab)
    m_tail = swet_tail * (areal_density_wing * 0.8)

    m_total = m_fuse + m_wing + m_tail + m_avionics_kg
    return {
        "m_fuse": m_fuse,
        "m_wing": m_wing,
        "m_tail": m_tail,
        "m_avionics": m_avionics_kg,
        "m_total": m_total,
    }


def calculate_total_mass_and_weight(
    params: dict[str, float],
    rho_material: float = 250.0,
    g: float = 9.81,
    m_avionics_kg: float = 0.080,
    areal_density_fuse: float = 1.20,
    areal_density_wing: float = 1.00,
) -> tuple[float, float]:
    breakdown = calculate_mass_breakdown(
        params,
        m_avionics_kg=m_avionics_kg,
        areal_density_fuse=areal_density_fuse,
        areal_density_wing=areal_density_wing,
    )
    mass_kg = breakdown["m_total"]
    return mass_kg, mass_kg * g


def calculate_center_of_gravity(
    params: dict[str, float],
    m_avionics_kg: float = 0.080,
    areal_density_fuse: float = 1.20,
    areal_density_wing: float = 1.00,
) -> tuple[float, float, float]:
    """Compute dynamic center of gravity (x, y, z) in meters from component centroids."""
    breakdown = calculate_mass_breakdown(
        params,
        m_avionics_kg=m_avionics_kg,
        areal_density_fuse=areal_density_fuse,
        areal_density_wing=areal_density_wing,
    )
    m_total = breakdown["m_total"]

    # Longitudinal centroids in meters
    x_avionics = (params["fuse_length"] * 0.18) * 1e-3
    x_fuse = (params["fuse_length"] * 0.45) * 1e-3
    # Wing aerodynamic center / centroid approximation
    x_wing = (params["wing_x_pos"] + 0.25 * params["wing_root_chord"]) * 1e-3
    x_tail = (params["tail_x_pos"] + 0.25 * params["h_stab_root"]) * 1e-3

    x_cg = (
        breakdown["m_avionics"] * x_avionics
        + breakdown["m_fuse"] * x_fuse
        + breakdown["m_wing"] * x_wing
        + breakdown["m_tail"] * x_tail
    ) / max(m_total, 1e-6)

    return float(x_cg), 0.0, 0.0


def _apply_tail_volume_scaling(
    params: dict[str, float],
    baseline: dict[str, float],
    targets: TailVolumeTargets,
) -> None:
    """Scale tail dimensions proportionally to maintain target tail volume coefficients."""
    wing = calculate_wing_geometry(
        span=params["wing_span"],
        root_chord=params["wing_root_chord"],
        tip_chord=params["wing_tip_chord"],
    )
    x_wing_ac = params["wing_x_pos"] + 0.25 * params["wing_root_chord"]
    arm_h = (params["tail_x_pos"] + 0.25 * baseline.get("h_stab_root", 28.0)) - x_wing_ac
    arm_v = (params["tail_x_pos"] + 0.25 * baseline.get("v_stab_root", 35.0)) - x_wing_ac

    s_h_req, s_v_req = calculate_tail_planform_areas(wing, arm_h, arm_v, targets)

    # Baseline tail planform areas
    s_h_base = 2.0 * baseline["h_stab_span"] * (baseline["h_stab_root"] + baseline["h_stab_tip"]) / 2.0
    s_v_base = baseline["v_stab_height"] * (baseline["v_stab_root"] + baseline["v_stab_tip"]) / 2.0

    scale_h = float(np.sqrt(max(s_h_req / max(s_h_base, 1e-4), 0.01)))
    scale_v = float(np.sqrt(max(s_v_req / max(s_v_base, 1e-4), 0.01)))

    params["h_stab_span"] = baseline["h_stab_span"] * scale_h
    params["h_stab_root"] = baseline["h_stab_root"] * scale_h
    params["h_stab_tip"] = baseline["h_stab_tip"] * scale_h

    params["v_stab_height"] = baseline["v_stab_height"] * scale_v
    params["v_stab_root"] = baseline["v_stab_root"] * scale_v
    params["v_stab_tip"] = baseline["v_stab_tip"] * scale_v


def decode_genes(
    genes: Sequence[float],
    baseline: dict[str, float] = BASELINE,
    bounds: Sequence[GeneBound] = (),
    tail_targets: TailVolumeTargets | None = None,
) -> dict[str, float]:
    params = baseline.copy()
    for i, bound in enumerate(bounds):
        u = max(0.0, min(1.0, float(genes[i])))
        params[bound.name] = bound.low + (bound.high - bound.low) * u

    # Place tail root at the end of the fuselage (-10mm margin from the trailing tip)
    tail_chord = max(baseline.get("v_stab_root", 35.0), baseline.get("h_stab_root", 28.0))
    params["tail_x_pos"] = params["fuse_length"] - tail_chord - 10.0

    if tail_targets is not None:
        _apply_tail_volume_scaling(params, baseline, tail_targets)

    return params
