import numpy as np

from config import BASELINE, GENE_BOUNDS
from geometry import (
    Fuselage,
    calculate_total_mass_and_weight,
    decode_genes,
    fuselage_volume_mm3,
    fuselage_from_params,
)


def test_fuselage_volume_baseline_matches_current_value():
    expected = 13812.372
    assert round(fuselage_volume_mm3(BASELINE), 3) == expected


def test_fuselage_value_object_reports_volume():
    f = Fuselage(
        length=BASELINE["fuse_length"],
        nose_ratio=BASELINE["nose_ratio"],
        tail_ratio=BASELINE["tail_ratio"],
        max_diameter=BASELINE["fuse_max_diam"],
    )
    assert f.volume_mm3 == fuselage_volume_mm3(BASELINE)
    assert f.mid_length == f.length - f.nose_length - f.tail_length
    assert f.r_max == BASELINE["fuse_max_diam"] / 2.0


def test_fuselage_from_params_parses_geometry():
    f = fuselage_from_params(BASELINE)
    assert isinstance(f, Fuselage)
    assert f.length == BASELINE["fuse_length"]


def test_total_mass_and_weight_matches_current_value():
    mass_kg, weight_n = calculate_total_mass_and_weight(BASELINE, 250.0, 9.81)
    assert round(mass_kg, 6) == 0.014385
    assert round(weight_n, 6) == 0.141116
    assert weight_n == mass_kg * 9.81


def test_decode_genes_scales_midpoint_to_bound_midpoint():
    genes = [0.5] * len(GENE_BOUNDS)
    params = decode_genes(genes, BASELINE, GENE_BOUNDS)
    assert params["wing_span"] == 150.0
    assert params["naca_m"] == 2.5


def test_decode_genes_clamps_out_of_range():
    over = [2.0] * len(GENE_BOUNDS)
    under = [-2.0] * len(GENE_BOUNDS)
    over_params = decode_genes(over, BASELINE, GENE_BOUNDS)
    under_params = decode_genes(under, BASELINE, GENE_BOUNDS)
    for b in GENE_BOUNDS:
        assert over_params[b.name] == b.high
        assert under_params[b.name] == b.low


def test_decode_genes_keeps_baseline_for_unlisted_genes():
    genes = [0.5] * len(GENE_BOUNDS)
    params = decode_genes(genes, BASELINE, GENE_BOUNDS)
    assert params["h_stab_span"] == BASELINE["h_stab_span"]