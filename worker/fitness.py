from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aero import AeroEvaluator, AeroResult
from config import BASELINE, GENE_BOUNDS, GeneBound, WorkerConfig
from geometry import calculate_fuselage_volume_mm3, decode_genes


@dataclass(frozen=True)
class EvaluationOutcome:
    fitness: float
    fuselage_volume_mm3: float
    aero: AeroResult | None


class FitnessEvaluator:
    def __init__(
        self,
        aero_evaluator: AeroEvaluator,
        config: WorkerConfig,
        baseline: dict[str, float] = BASELINE,
        bounds: Sequence[GeneBound] = GENE_BOUNDS,
    ):
        self._aero_evaluator = aero_evaluator
        self._config = config
        self._baseline = baseline
        self._bounds = bounds

    def evaluate_genes(self, genes: Sequence[float]) -> EvaluationOutcome:
        params = decode_genes(genes, self._baseline, self._bounds)
        return self.evaluate_params(params)

    def evaluate_params(self, params: dict[str, float]) -> EvaluationOutcome:
        v_fuse = calculate_fuselage_volume_mm3(params)

        if v_fuse < self._config.v_min_fuse_mm3:
            return EvaluationOutcome(fitness=-1e9, fuselage_volume_mm3=v_fuse, aero=None)

        aero = self._aero_evaluator.evaluate(params)
        if aero is None:
            return EvaluationOutcome(fitness=-1e9, fuselage_volume_mm3=v_fuse, aero=None)

        alpha_penalty = self._config.w_alpha * abs(aero.alpha_trim)
        stability_penalty = self._config.w_stability * max(0.0, aero.cm_alpha) ** 2
        fitness = aero.ld - alpha_penalty - stability_penalty

        return EvaluationOutcome(
            fitness=fitness,
            fuselage_volume_mm3=v_fuse,
            aero=aero,
        )
