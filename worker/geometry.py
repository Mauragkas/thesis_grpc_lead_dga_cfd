from __future__ import annotations

from typing import Sequence
import numpy as np

from config import BASELINE, GeneBound


def wing_panel_volume(span: float, root_c: float, tip_c: float, t_ratio: float) -> float:
    return 0.685 * t_ratio * (span / 3.0) * (root_c**2 + root_c * tip_c + tip_c**2)


def calculate_fuselage_volume_mm3(params: dict[str, float]) -> float:
    l_nose = params["fuse_length"] * params["nose_ratio"]
    l_tail = params["fuse_length"] * params["tail_ratio"]
    l_mid = params["fuse_length"] - l_nose - l_tail
    r_max = params["fuse_max_diam"] / 2.0

    v_nose = (2.0 / 3.0) * np.pi * (r_max**2) * l_nose
    v_mid = np.pi * (r_max**2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0 / 3.0) * np.pi * l_tail * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)

    return float(v_nose + v_mid + v_tail)


def calculate_total_mass_and_weight(
    params: dict[str, float],
    rho_material: float,
    g: float,
) -> tuple[float, float]:
    l_nose = params["fuse_length"] * params["nose_ratio"]
    l_tail = params["fuse_length"] * params["tail_ratio"]
    l_mid = params["fuse_length"] - l_nose - l_tail
    r_max = params["fuse_max_diam"] / 2.0

    v_nose = (2.0 / 3.0) * np.pi * (r_max**2) * l_nose
    v_mid = np.pi * (r_max**2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0 / 3.0) * np.pi * l_tail * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)
    v_fuse = v_nose + v_mid + v_tail

    t_main = params["naca_t"] / 100.0
    v_main = 2.0 * wing_panel_volume(
        params["wing_span"],
        params["wing_root_chord"],
        params["wing_tip_chord"],
        t_main,
    )
    v_h = 2.0 * wing_panel_volume(
        params["h_stab_span"],
        params["h_stab_root"],
        params["h_stab_tip"],
        0.10,
    )
    v_v = wing_panel_volume(
        params["v_stab_height"],
        params["v_stab_root"],
        params["v_stab_tip"],
        0.10,
    )

    v_total_m3 = (v_fuse + v_main + v_h + v_v) * 1e-9
    mass_kg = v_total_m3 * rho_material
    return mass_kg, mass_kg * g


def decode_genes(
    genes: Sequence[float],
    baseline: dict[str, float] = BASELINE,
    bounds: Sequence[GeneBound] = (),
) -> dict[str, float]:
    params = baseline.copy()
    for i, bound in enumerate(bounds):
        u = max(0.0, min(1.0, float(genes[i])))
        params[bound.name] = bound.low + (bound.high - bound.low) * u
    return params
