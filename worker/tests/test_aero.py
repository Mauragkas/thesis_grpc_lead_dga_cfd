import numpy as np
import pytest

from aero import AeroResult, solve_trim


def test_solve_trim_returns_result_when_in_sweep_range():
    alpha = np.linspace(-2.0, 10.0, 13)
    cl = alpha * 0.05 + 0.3
    cd = np.full_like(alpha, 0.02)
    cm = np.linspace(-0.1, 0.1, 13)
    result = solve_trim(0.5, alpha, cl, cd, cm)
    assert isinstance(result, AeroResult)
    assert result.ld == 0.5 / 0.02
    assert result.alpha_trim == pytest.approx((0.5 - 0.3) / 0.05)
    assert hasattr(result, "cm_trim")
    assert result.cm_trim == pytest.approx(np.interp(result.alpha_trim, alpha, cm))


def test_solve_trim_returns_none_when_below_sweep_range():
    alpha = np.linspace(-2.0, 10.0, 13)
    cl = alpha * 0.05 + 0.3
    cd = np.full_like(alpha, 0.02)
    cm = np.linspace(-0.1, 0.1, 13)
    assert solve_trim(-5.0, alpha, cl, cd, cm) is None


def test_solve_trim_returns_none_when_above_sweep_range():
    alpha = np.linspace(-2.0, 10.0, 13)
    cl = alpha * 0.05 + 0.3
    cd = np.full_like(alpha, 0.02)
    cm = np.linspace(-0.1, 0.1, 13)
    assert solve_trim(500.0, alpha, cl, cd, cm) is None


def test_evaluator_baseline_has_realistic_aerodynamics():
    from config import BASELINE, WorkerConfig
    from aero import AerosandboxAeroEvaluator

    cfg = WorkerConfig(worker_id="test", v_min_fuse_mm3=20000.0)
    evaluator = AerosandboxAeroEvaluator(cfg)
    result = evaluator.evaluate(BASELINE)

    assert result is not None
    assert 5.0 <= result.ld <= 20.0
    assert result.cm_alpha < 0.0  # Pitch-stable


def test_evaluator_prevents_unphysical_drag_cancellation():
    from config import BASELINE, GENE_BOUNDS, WorkerConfig
    from geometry import decode_genes
    from aero import AerosandboxAeroEvaluator

    cfg = WorkerConfig(worker_id="test", v_min_fuse_mm3=20000.0)
    evaluator = AerosandboxAeroEvaluator(cfg)

    # Mutant genome that previously triggered VLM near-field cancellation artifact
    mutant_genes = [
        0.10184141013480076, 0.490470497315193, 0.0, 0.3651039759781773,
        1.0, 0.9699307495372821, 0.9874281426430888, 0.9683688239520192,
        1.0, 0.7243182994183703
    ]
    params = decode_genes(mutant_genes, BASELINE, GENE_BOUNDS)
    result = evaluator.evaluate(params)

    # If it trims, L/D must not blow up to unphysical values (> 30)
    if result is not None:
        assert result.ld <= 30.0