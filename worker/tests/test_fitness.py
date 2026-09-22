import pytest
import numpy as np
from config import BASELINE, WorkerConfig
from fitness import REJECT_FITNESS, EvaluationOutcome, FitnessEvaluator
from aero import AeroResult
from mission.polar import AeroPolar
from mission.trajectory import MissionOutcome


class FakeAero:
    def __init__(self, result: AeroResult | None):
        self._result = result

    def evaluate(self, params):
        return self._result


AERO_OK = AeroResult(ld=8.0, alpha_trim=2.0, cm_alpha=0.05)


def make_evaluator(aero_result=AERO_OK, v_min=20000.0, **cfg) -> FitnessEvaluator:
    config = WorkerConfig(worker_id="t", v_min_fuse_mm3=v_min, **cfg)
    return FitnessEvaluator(
        aero_evaluator=FakeAero(aero_result),
        config=config,
    )


def test_rejects_when_fuselage_below_min_volume():
    ev = make_evaluator(v_min=10_000_000.0)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.rejected is True
    assert outcome.fitness == REJECT_FITNESS
    assert outcome.aero is None


def test_rejects_when_aero_untrimmable():
    ev = make_evaluator(aero_result=None, v_min=1.0)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.rejected is True
    assert outcome.fitness == REJECT_FITNESS
    assert outcome.aero is None


def test_scores_using_ld_minus_penalties():
    ev = make_evaluator(aero_result=AERO_OK, v_min=1.0, enable_mission_sim=False)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.rejected is False
    alpha_pen = 0.5 * abs(AERO_OK.alpha_trim)
    stab_pen = 20.0 * max(0.0, AERO_OK.cm_alpha) ** 2
    mom_pen = 30.0 * (AERO_OK.cm_trim**2)
    expected = AERO_OK.ld - alpha_pen - stab_pen - mom_pen
    assert outcome.fitness == expected
    assert outcome.aero == AERO_OK
    assert outcome.fuselage_volume_mm3 > 0.0


def test_scores_penalizes_out_of_trim_moment():
    untrimmed = AeroResult(ld=12.0, alpha_trim=3.0, cm_alpha=-2.0, cm_trim=-0.2)
    ev = make_evaluator(aero_result=untrimmed, v_min=1.0, enable_mission_sim=False)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.rejected is False
    alpha_pen = 0.5 * 3.0
    stab_pen = 0.0  # cm_alpha < 0 is stable
    mom_pen = 30.0 * ((-0.2) ** 2)
    expected = 12.0 - alpha_pen - stab_pen - mom_pen
    assert outcome.fitness == pytest.approx(expected)


def test_rejected_outcome_carries_volume():
    ev = make_evaluator(aero_result=None, v_min=1.0)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.fuselage_volume_mm3 > 0.0


def test_evaluator_integrates_mission_simulation_when_polar_present():
    alpha = np.linspace(-2.0, 10.0, 13)
    cl = 0.08 * alpha + 0.3
    cd = 0.025 + 0.002 * (alpha**2)
    cm = -0.05 - 0.01 * alpha
    polar = AeroPolar(alpha_deg=alpha, cl=cl, cd=cd, cm=cm)

    aero_with_polar = AeroResult(
        ld=10.0,
        alpha_trim=1.5,
        cm_alpha=-0.05,
        cm_trim=0.0,
        polar=polar,
        mass_kg=0.25,
        s_ref=0.04,
    )

    ev = make_evaluator(aero_result=aero_with_polar, v_min=1.0, enable_mission_sim=True)
    outcome = ev.evaluate_genes([0.5] * 10)

    assert outcome.rejected is False
    assert outcome.mission is not None
    assert outcome.mission.completed is True
    assert outcome.mission.takeoff_distance_m > 0.0
    assert outcome.fitness < aero_with_polar.ld  # Penalties applied
