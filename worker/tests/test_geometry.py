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
    assert round(fuselage_volume_mm3(BASELINE), 3) == 55249.488


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
    mass_kg, weight_n = calculate_total_mass_and_weight(BASELINE)
    assert 0.10 <= mass_kg <= 0.25
    assert weight_n == mass_kg * 9.81


def test_dynamic_cg_tracks_wing_position():
    from geometry import calculate_center_of_gravity

    p_fwd = BASELINE.copy()
    p_fwd["wing_x_pos"] = 55.0
    p_aft = BASELINE.copy()
    p_aft["wing_x_pos"] = 95.0

    cg_fwd, _, _ = calculate_center_of_gravity(p_fwd)
    cg_aft, _, _ = calculate_center_of_gravity(p_aft)
    assert cg_fwd < cg_aft


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


def test_tail_x_pos_placed_at_end_of_fuselage():
    # Test min fuse length (200mm)
    genes_short = [0.0] * len(GENE_BOUNDS)
    params_short = decode_genes(genes_short, BASELINE, GENE_BOUNDS)
    expected_short = 200.0 - max(params_short["v_stab_root"], params_short["h_stab_root"]) - 10.0
    assert params_short["tail_x_pos"] == expected_short

    # Test max fuse length (350mm)
    genes_long = [1.0] * len(GENE_BOUNDS)
    params_long = decode_genes(genes_long, BASELINE, GENE_BOUNDS)
    expected_long = 350.0 - max(params_long["v_stab_root"], params_long["h_stab_root"]) - 10.0
    assert params_long["tail_x_pos"] == expected_long
    assert params_long["tail_x_pos"] > params_short["tail_x_pos"]