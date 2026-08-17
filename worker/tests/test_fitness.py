from config import BASELINE, WorkerConfig
from fitness import REJECT_FITNESS, EvaluationOutcome, FitnessEvaluator
from aero import AeroResult


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
    ev = make_evaluator(aero_result=AERO_OK, v_min=1.0)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.rejected is False
    alpha_pen = 0.5 * abs(AERO_OK.alpha_trim)
    stab_pen = 20.0 * max(0.0, AERO_OK.cm_alpha) ** 2
    expected = AERO_OK.ld - alpha_pen - stab_pen
    assert outcome.fitness == expected
    assert outcome.aero == AERO_OK
    assert outcome.fuselage_volume_mm3 > 0.0


def test_rejected_outcome_carries_volume():
    ev = make_evaluator(aero_result=None, v_min=1.0)
    outcome = ev.evaluate_genes([0.5] * 10)
    assert outcome.fuselage_volume_mm3 > 0.0