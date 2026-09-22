import numpy as np
import pytest
from worker.mission.polar import AeroPolar


def test_aero_polar_interpolation():
    alpha = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    cl = 0.08 * alpha + 0.3
    cd = 0.02 + 0.001 * (alpha**2)
    cm = -0.05 - 0.01 * alpha

    polar = AeroPolar(alpha_deg=alpha, cl=cl, cd=cd, cm=cm)

    assert polar.lift_coefficient(0.0) == pytest.approx(0.3)
    assert polar.lift_coefficient(4.0) == pytest.approx(0.62)
    assert polar.drag_coefficient(0.0) == pytest.approx(0.02)
    assert polar.drag_coefficient(4.0) == pytest.approx(0.036)


def test_aero_polar_max_cl_and_stall_speed():
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.2, 0.7, 1.2])  # CL_max = 1.2
    cd = np.array([0.02, 0.04, 0.08])
    cm = np.array([0.0, 0.0, 0.0])

    polar = AeroPolar(alpha_deg=alpha, cl=cl, cd=cd, cm=cm)

    assert polar.cl_max == pytest.approx(1.2)
    # Weight = 10 N, S_ref = 0.2 m^2, rho = 1.225
    # V_stall = sqrt(2 * 10 / (1.225 * 0.2 * 1.2)) = sqrt(20 / 0.294) ~ 8.246 m/s
    v_stall = polar.stall_speed(weight_n=10.0, s_ref_m2=0.2, rho_air=1.225)
    assert v_stall == pytest.approx(8.246, rel=1e-2)


def test_aero_polar_trim_alpha_for_cl():
    alpha = np.linspace(-2.0, 10.0, 7)
    cl = 0.1 * alpha + 0.2
    cd = np.full_like(alpha, 0.03)
    cm = np.full_like(alpha, -0.02)

    polar = AeroPolar(alpha_deg=alpha, cl=cl, cd=cd, cm=cm)

    # CL = 0.5 -> 0.1 * alpha + 0.2 = 0.5 -> alpha = 3.0
    alpha_trim = polar.trim_alpha_for_cl(0.5)
    assert alpha_trim == pytest.approx(3.0)

    # Untrimmable target outside range
    assert polar.trim_alpha_for_cl(5.0) is None
