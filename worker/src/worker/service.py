from __future__ import annotations

import asyncio
import logging
import time

try:
    from . import eval_pb2
    from . import eval_pb2_grpc
    from .fitness import REJECT_FITNESS, EvaluationOutcome, FitnessEvaluator
except (ImportError, ValueError):
    import eval_pb2
    import eval_pb2_grpc
    from fitness import REJECT_FITNESS, EvaluationOutcome, FitnessEvaluator


class EvaluatorServicer(eval_pb2_grpc.EvaluatorServicer):
    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        worker_id: str,
        gene_count: int,
        logger: logging.Logger,
        semaphore_size: int = 1,
    ):
        self._fitness_evaluator = fitness_evaluator
        self._worker_id = worker_id
        self._gene_count = gene_count
        self._logger = logger
        self._semaphore = asyncio.Semaphore(semaphore_size)

    def _rejection(self) -> eval_pb2.EvaluationResult:
        return eval_pb2.EvaluationResult(worker_id=self._worker_id, fitness=REJECT_FITNESS)

    def _valid_genes(self, genes: list[float]) -> bool:
        return len(genes) == self._gene_count

    def _result(self, outcome: EvaluationOutcome) -> eval_pb2.EvaluationResult:
        return eval_pb2.EvaluationResult(
            worker_id=self._worker_id,
            fitness=outcome.fitness,
        )

    async def Evaluate(self, request, context):
        t0 = time.perf_counter()
        async with self._semaphore:
            genes = list(request.genes)
            if not self._valid_genes(genes):
                self._logger.warning(
                    "Evaluate worker=%s got=%d expected=%d",
                    self._worker_id,
                    len(genes),
                    self._gene_count,
                )
                return self._rejection()

            self._logger.debug("Evaluate worker=%s genes=%s", self._worker_id, genes)

            outcome = await self._evaluate(genes)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.info(
                "Evaluate worker=%s fit=%.4f v_fuse=%.0fmm3 alpha=%.2f cm_alpha=%.3f dt=%.1fms",
                self._worker_id,
                outcome.fitness,
                outcome.fuselage_volume_mm3,
                outcome.aero.alpha_trim if outcome.aero else float("nan"),
                outcome.aero.cm_alpha if outcome.aero else float("nan"),
                dt_ms,
            )
            return self._result(outcome)

    async def EvaluateBatch(self, request, context):
        t0 = time.perf_counter()
        n = len(request.individuals)
        self._logger.info("EvaluateBatch worker=%s n=%d", self._worker_id, n)

        async with self._semaphore:
            results = []
            n_fail = 0

            for i, ind in enumerate(request.individuals):
                genes = list(ind.genes)
                if not self._valid_genes(genes):
                    self._logger.warning(
                        "EvaluateBatch worker=%s idx=%d got=%d expected=%d",
                        self._worker_id,
                        i,
                        len(genes),
                        self._gene_count,
                    )
                    results.append(self._rejection())
                    n_fail += 1
                    continue

                self._logger.debug(
                    "EvaluateBatch worker=%s idx=%d genes=%s",
                    self._worker_id,
                    i,
                    genes,
                )

                outcome = await self._evaluate(genes)
                if outcome.rejected:
                    n_fail += 1
                results.append(self._result(outcome))

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._logger.info(
                "EvaluateBatch worker=%s n=%d ok=%d fail=%d dt=%.1fms avg=%.2fms",
                self._worker_id,
                n,
                n - n_fail,
                n_fail,
                dt_ms,
                dt_ms / n if n else 0.0,
            )
            return eval_pb2.BatchResponse(results=results)

    async def _evaluate(self, genes: list[float]) -> EvaluationOutcome:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._fitness_evaluator.evaluate_genes,
            genes,
        )