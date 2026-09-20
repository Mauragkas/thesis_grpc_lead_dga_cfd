from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import aerosandbox as asb
import numpy as np

try:
    from .config import WorkerConfig
    from .geometry import calculate_total_mass_and_weight, calculate_center_of_gravity
    from .slender_aerodynamics import get_fuselage_drag_coefficient
except (ImportError, ValueError):
    from config import WorkerConfig
    from geometry import calculate_total_mass_and_weight, calculate_center_of_gravity
    from slender_aerodynamics import get_fuselage_drag_coefficient


@dataclass(frozen=True)
class AeroResult:
    ld: float
    alpha_trim: float
    cm_alpha: float
    cm_trim: float = 0.0


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
    """Interpolate trim condition from an alpha sweep; None when untrimmable."""
    if cl_req < min(cl_list) or cl_req > max(cl_list):
        return None
    cd_trim = float(np.interp(cl_req, cl_list, cd_list))
    alpha_trim = float(np.interp(cl_req, cl_list, alpha_sweep))
    cm_trim = float(np.interp(alpha_trim, alpha_sweep, cm_list))
    mid_idx = len(alpha_sweep) // 2
    cm_alpha = float(np.gradient(cm_list, np.radians(alpha_sweep))[mid_idx])
    ld = cl_req / max(cd_trim, 1e-4)
    return AeroResult(ld=ld, alpha_trim=alpha_trim, cm_alpha=cm_alpha, cm_trim=cm_trim)


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
            m_avionics_kg=self._config.m_avionics_kg,
            areal_density_fuse=self._config.areal_density_fuse,
            areal_density_wing=self._config.areal_density_wing,
        )
        cg_pos = calculate_center_of_gravity(
            params,
            m_avionics_kg=self._config.m_avionics_kg,
            areal_density_fuse=self._config.areal_density_fuse,
            areal_density_wing=self._config.areal_density_wing,
        )
        log.debug("mass=%.3fkg weight=%.3fN cg=%s", mass_kg, weight_n, cg_pos)

        s_ref = self._reference_area(params)
        airplane = self._build_airplane(params, cg_pos=cg_pos)
        cl_req = self._required_cl(weight_n, s_ref)

        alpha_sweep, cl_list, cd_vlm_list, cm_list = self._run_sweep(airplane)

        # Parasitic drag buildup ported from Atlas-Upat
        cd_fuse, _ = get_fuselage_drag_coefficient(
            length=params["fuse_length"] * 1e-3,
            max_diameter=params["fuse_max_diam"] * 1e-3,
            s_ref=s_ref,
            velocity=self._config.v_cruise,
            nose_ratio=params.get("nose_ratio", 0.25),
            tail_ratio=params.get("tail_ratio", 0.35),
            rho=self._config.rho_air,
        )

        # Wing profile drag
        c_mean = (params["wing_root_chord"] + params["wing_tip_chord"]) / 2.0 * 1e-3
        re_wing = (self._config.rho_air * self._config.v_cruise * c_mean) / 1.81e-5
        cf_wing = 1.328 / np.sqrt(max(re_wing, 1.0))
        tc = params.get("naca_t", 12.0) / 100.0
        ff_wing = 1.0 + 2.0 * tc + 60.0 * (tc**4)
        cd0_wing = 2.05 * cf_wing * ff_wing * 1.25

        # Empennage profile drag
        s_tail = (
            2.0 * params["h_stab_span"] * (params["h_stab_root"] + params["h_stab_tip"]) / 2.0
            + params["v_stab_height"] * (params["v_stab_root"] + params["v_stab_tip"]) / 2.0
        ) * 1e-6
        c_tail_mean = 0.025
        re_tail = (self._config.rho_air * self._config.v_cruise * c_tail_mean) / 1.81e-5
        cf_tail = 1.328 / np.sqrt(max(re_tail, 1.0))
        cd0_tail = (2.05 * s_tail / s_ref) * cf_tail * 1.20 * 1.25

        cd0_total = cd_fuse + cd0_wing + cd0_tail

        # Enforce physical induced drag floor (Prandtl / Munk minimum induced drag from Atlas-Upat):
        # CDi >= CL^2 / (pi * AR * e). VLM near-field panel discretization on coarse panels
        # with sweep and camber can produce spurious negative drag, which must be bounded.
        b_span = 2.0 * params["wing_span"] * 1e-3
        ar = max((b_span**2) / max(s_ref, 1e-6), 1.0)
        e_oswald = 1.0 / (1.0 + 0.38 * cd0_total * np.pi * ar)

        cd_list = []
        for cl, cd_vlm in zip(cl_list, cd_vlm_list):
            cdi_min = (cl**2) / (np.pi * ar * max(e_oswald, 0.1))
            cdi = max(cd_vlm, cdi_min)
            cd_list.append(cdi + cd0_total)

        log.debug(
            "alpha_sweep cl=[%.3f..%.3f] cd=[%.4f..%.4f] cd0=%.4f ar=%.2f",
            min(cl_list), max(cl_list), min(cd_list), max(cd_list), cd0_total, ar,
        )
        if cl_req < min(cl_list) or cl_req > max(cl_list):
            log.debug(
                "trim out of sweep cl_req=%.4f range=[%.4f,%.4f]",
                cl_req, min(cl_list), max(cl_list),
            )
            return None

        result = solve_trim(cl_req, alpha_sweep, cl_list, cd_list, cm_list)
        if result is not None:
            log.debug(
                "trim alpha=%.3f cd=%.4f ld=%.3f cm_alpha=%.4f cm_trim=%.4f",
                result.alpha_trim, cl_req / max(result.ld, 1e-4), result.ld, result.cm_alpha, result.cm_trim,
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

    def _build_airplane(
        self,
        params: dict[str, float],
        cg_pos: tuple[float, float, float] | None = None,
    ) -> asb.Airplane:
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
            xyz_ref=cg_pos if cg_pos is not None else self._config.cg_pos,
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