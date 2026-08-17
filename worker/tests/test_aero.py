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