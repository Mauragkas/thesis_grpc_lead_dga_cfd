from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import aerosandbox as asb
import numpy as np

from config import WorkerConfig
from geometry import calculate_total_mass_and_weight


@dataclass(frozen=True)
class AeroResult:
    ld: float
    alpha_trim: float
    cm_alpha: float


class AeroEvaluator(Protocol):
    def evaluate(self, params: dict[str, float]) -> AeroResult | None:
        ...


def solve_trim(
    cl_req: float,
    alpha_sweep: np.ndarray,
    cl_list: list[float],
    cd_list: list[float],
    cm_list: list[float],
) -> AeroResult | None:
    """Interpolate trim condition from a VLM alpha sweep; None when untrimmable."""
    if cl_req < min(cl_list) or cl_req > max(cl_list):
        return None
    cd_trim = float(np.interp(cl_req, cl_list, cd_list))
    alpha_trim = float(np.interp(cl_req, cl_list, alpha_sweep))
    cm_alpha = float(np.gradient(cm_list, np.radians(alpha_sweep))[6])
    return AeroResult(ld=cl_req / cd_trim, alpha_trim=alpha_trim, cm_alpha=cm_alpha)


class AerosandboxAeroEvaluator:
    def __init__(self, config: WorkerConfig, logger: logging.Logger | None = None):
        self._config = config
        self._logger = logger or logging.getLogger("worker.aero")

    def evaluate(self, params: dict[str, float]) -> AeroResult | None:
        log = self._logger
        mass_kg, weight_n = calculate_total_mass_and_weight(
            params,
            rho_material=self._config.rho_material,
            g=self._config.g,
        )
        log.debug("mass=%.3fkg weight=%.3fN", mass_kg, weight_n)

        s_ref = self._reference_area(params)
        airplane = self._build_airplane(params)
        cl_req = self._required_cl(weight_n, s_ref)

        alpha_sweep, cl_list, cd_list, cm_list = self._run_sweep(airplane)
        log.debug(
            "alpha_sweep cl=[%.3f..%.3f] cd=[%.4f..%.4f]",
            min(cl_list), max(cl_list), min(cd_list), max(cd_list),
        )
        if cl_req < min(cl_list) or cl_req > max(cl_list):
            log.debug(
                "trim out of sweep cl_req=%.4f range=[%.4f,%.4f]",
                cl_req, min(cl_list), max(cl_list),
            )
            return None

        result = solve_trim(cl_req, alpha_sweep, cl_list, cd_list, cm_list)
        log.debug(
            "trim alpha=%.3f cd=%.4f ld=%.3f cm_alpha=%.4f",
            result.alpha_trim, cl_req / result.ld, result.ld, result.cm_alpha,
        )
        return result

    @staticmethod
    def _reference_area(params: dict[str, float]) -> float:
        return (
            2.0
            * params["wing_span"]
            * (params["wing_root_chord"] + params["wing_tip_chord"])
            / 2.0
            * 1e-6
        )

    def _required_cl(self, weight_n: float, s_ref: float) -> float:
        return (2.0 * weight_n) / (
            self._config.rho_air * (self._config.v_cruise**2) * s_ref
        )

    def _build_airplane(self, params: dict[str, float]) -> asb.Airplane:
        scale = 1e-3
        airfoil_main, airfoil_tail = self._airfoils(params)

        main_wing = asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=params["wing_root_chord"] * scale,
                    twist=0,
                    airfoil=airfoil_main,
                ),
                asb.WingXSec(
                    xyz_le=[
                        params["wing_span"] * scale * np.tan(np.radians(params["wing_sweep"])),
                        params["wing_span"] * scale,
                        params["wing_span"] * scale * np.tan(np.radians(params["wing_dihedral"])),
                    ],
                    chord=params["wing_tip_chord"] * scale,
                    twist=params["wing_twist"],
                    airfoil=airfoil_main,
                ),
            ],
        ).translate([params["wing_x_pos"] * scale, 0, params["wing_z_pos"] * scale])

        h_stab = asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=params["h_stab_root"] * scale,
                    twist=0,
                    airfoil=airfoil_tail,
                ),
                asb.WingXSec(
                    xyz_le=[
                        (params["h_stab_root"] - params["h_stab_tip"]) * scale,
                        params["h_stab_span"] * scale,
                        0,
                    ],
                    chord=params["h_stab_tip"] * scale,
                    twist=0,
                    airfoil=airfoil_tail,
                ),
            ],
        ).translate([params["tail_x_pos"] * scale, 0, 0])

        v_stab = asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=params["v_stab_root"] * scale,
                    twist=0,
                    airfoil=airfoil_tail,
                ),
                asb.WingXSec(
                    xyz_le=[
                        (params["v_stab_root"] - params["v_stab_tip"]) * scale,
                        0,
                        params["v_stab_height"] * scale,
                    ],
                    chord=params["v_stab_tip"] * scale,
                    twist=0,
                    airfoil=airfoil_tail,
                ),
            ],
        ).translate([params["tail_x_pos"] * scale, 0, 0])

        return asb.Airplane(
            name="Parametric Model",
            xyz_ref=self._config.cg_pos,
            wings=[main_wing, h_stab, v_stab],
        )

    @staticmethod
    def _airfoils(params: dict[str, float]) -> tuple[asb.Airfoil, asb.Airfoil]:
        naca_str = f"naca{int(round(params['naca_m']))}{int(params['naca_p'])}{int(params['naca_t']):02d}"
        return asb.Airfoil(naca_str), asb.Airfoil("naca0010")

    def _run_sweep(
        self, airplane: asb.Airplane
    ) -> tuple[np.ndarray, list[float], list[float], list[float]]:
        alpha_sweep = np.linspace(-2.0, 10.0, 13)
        cl_list, cd_list, cm_list = [], [], []
        for alpha in alpha_sweep:
            op_point = asb.OperatingPoint(velocity=self._config.v_cruise, alpha=alpha)
            vlm = asb.VortexLatticeMethod(
                airplane=airplane,
                op_point=op_point,
                spanwise_resolution=8,
                chordwise_resolution=3,
            )
            res = vlm.run()
            cl_list.append(res["CL"])
            cd_list.append(res["CD"])
            cm_list.append(res["Cm"])
        return alpha_sweep, cl_list, cd_list, cm_list