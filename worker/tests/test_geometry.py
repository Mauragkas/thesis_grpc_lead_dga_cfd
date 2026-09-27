import numpy as np
import pytest

from config import BASELINE, GENE_BOUNDS
from geometry import (
    Fuselage,
    TailVolumeTargets,
    calculate_tail_planform_areas,
    calculate_total_mass_and_weight,
    calculate_wing_geometry,
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


def test_tail_volume_targets_validates_positive():
    with pytest.raises(ValueError, match="vh must be positive"):
        TailVolumeTargets(vh=0.0, vv=0.04)
    with pytest.raises(ValueError, match="vv must be positive"):
        TailVolumeTargets(vh=0.5, vv=-0.01)


def test_calculate_wing_geometry():
    wing = calculate_wing_geometry(span=140.0, root_chord=55.0, tip_chord=25.0)
    assert wing.planform_area == 11200.0
    assert wing.wingspan == 280.0
    assert round(wing.mac, 2) == 41.88


def test_calculate_tail_planform_areas():
    wing = calculate_wing_geometry(span=140.0, root_chord=55.0, tip_chord=25.0)
    targets = TailVolumeTargets(vh=0.5, vv=0.04)
    sh, sv = calculate_tail_planform_areas(wing, arm_h=120.0, arm_v=120.0, targets=targets)
    # Sh = Vh * Sw * mac / Lh = 0.5 * 11200 * 41.88 / 120
    assert round(sh, 1) == round(0.5 * 11200.0 * wing.mac / 120.0, 1)
    # Sv = Vv * Sw * b / Lv = 0.04 * 11200 * 280 / 120
    assert round(sv, 1) == round(0.04 * 11200.0 * 280.0 / 120.0, 1)


def test_decode_genes_with_tail_volume_scaling_adjusts_tail_with_larger_wing():
    # Baseline wing
    small_wing_genes = [0.0] * len(GENE_BOUNDS)
    large_wing_genes = [0.0] * len(GENE_BOUNDS)
    # index 0 is wing_span (80 to 220)
    large_wing_genes[0] = 1.0

    targets = TailVolumeTargets(vh=0.5, vv=0.04)
    small_params = decode_genes(small_wing_genes, BASELINE, GENE_BOUNDS, tail_targets=targets)
    large_params = decode_genes(large_wing_genes, BASELINE, GENE_BOUNDS, tail_targets=targets)

    # Tail on larger wing plane should be significantly larger to maintain tail volume
    assert large_params["h_stab_span"] > small_params["h_stab_span"]
    assert large_params["v_stab_height"] > small_params["v_stab_height"]


def test_decode_genes_default_keeps_baseline_when_no_targets():
    genes = [0.5] * len(GENE_BOUNDS)
    params = decode_genes(genes, BASELINE, GENE_BOUNDS)
    assert params["h_stab_span"] == BASELINE["h_stab_span"]
    assert params["v_stab_height"] == BASELINE["v_stab_height"]


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
