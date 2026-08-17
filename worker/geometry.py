from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from config import BASELINE, GeneBound


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


def calculate_total_mass_and_weight(
    params: dict[str, float],
    rho_material: float,
    g: float,
) -> tuple[float, float]:
    v_fuse = fuselage_from_params(params).volume_mm3

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