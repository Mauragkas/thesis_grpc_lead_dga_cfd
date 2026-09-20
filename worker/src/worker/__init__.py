"""Worker package for aerodynamic evaluation service."""

from .aero import AeroResult, AerosandboxAeroEvaluator, solve_trim
from .config import BASELINE, GENE_BOUNDS, WorkerConfig, load_config
from .fitness import REJECT_FITNESS, EvaluationOutcome, FitnessEvaluator
from .geometry import (
    Fuselage,
    calculate_center_of_gravity,
    calculate_mass_breakdown,
    calculate_total_mass_and_weight,
    decode_genes,
    fuselage_from_params,
    fuselage_volume_mm3,
    wing_panel_volume,
)
from .slender_aerodynamics import (
    get_fuselage_drag_coefficient,
    get_fuselage_wetted_area,
)
from .hilbert import (
    BITS,
    NUM_CURVES,
    config_hilbert,
    hilbert_encode,
    parse_hilbert_key,
    probe_keys,
)
from .service import EvaluatorServicer

__all__ = [
    "AeroResult",
    "AerosandboxAeroEvaluator",
    "solve_trim",
    "BASELINE",
    "GENE_BOUNDS",
    "WorkerConfig",
    "load_config",
    "REJECT_FITNESS",
    "EvaluationOutcome",
    "FitnessEvaluator",
    "Fuselage",
    "calculate_center_of_gravity",
    "calculate_mass_breakdown",
    "calculate_total_mass_and_weight",
    "decode_genes",
    "fuselage_from_params",
    "fuselage_volume_mm3",
    "wing_panel_volume",
    "get_fuselage_drag_coefficient",
    "get_fuselage_wetted_area",
    "BITS",
    "NUM_CURVES",
    "config_hilbert",
    "hilbert_encode",
    "parse_hilbert_key",
    "probe_keys",
    "EvaluatorServicer",
]
