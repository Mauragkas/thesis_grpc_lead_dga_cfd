from __future__ import annotations

import asyncio
import logging
import time

import eval_pb2
import eval_pb2_grpc

from fitness import FitnessEvaluator


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

    async def Evaluate(self, request, context):
        t0 = time.perf_counter()
        async with self._semaphore:
            genes = list(request.genes)
            if len(genes) != self._gene_count:
                self._logger.warning(
                    "Evaluate worker=%s got=%d expected=%d",
                    self._worker_id,
                    len(genes),
                    self._gene_count,
                )
                return eval_pb2.EvaluationResult(worker_id=self._worker_id, fitness=-1e9)

            self._logger.debug(
                "Evaluate worker=%s genes=%s", self._worker_id, genes
            )

            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(
                None,
                self._fitness_evaluator.evaluate_genes,
                genes,
            )

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
            return eval_pb2.EvaluationResult(
                worker_id=self._worker_id,
                fitness=outcome.fitness,
            )

    async def EvaluateBatch(self, request, context):
        t0 = time.perf_counter()
        n = len(request.individuals)
        self._logger.info("EvaluateBatch worker=%s n=%d", self._worker_id, n)

        async with self._semaphore:
            loop = asyncio.get_running_loop()
            results = []
            n_fail = 0

            for i, ind in enumerate(request.individuals):
                genes = list(ind.genes)
                if len(genes) != self._gene_count:
                    self._logger.warning(
                        "EvaluateBatch worker=%s idx=%d got=%d expected=%d",
                        self._worker_id,
                        i,
                        len(genes),
                        self._gene_count,
                    )
                    results.append(
                        eval_pb2.EvaluationResult(worker_id=self._worker_id, fitness=-1e9)
                    )
                    n_fail += 1
                    continue

                self._logger.debug(
                    "EvaluateBatch worker=%s idx=%d genes=%s",
                    self._worker_id,
                    i,
                    genes,
                )

                outcome = await loop.run_in_executor(
                    None,
                    self._fitness_evaluator.evaluate_genes,
                    genes,
                )
                if outcome.fitness <= -1e8:
                    n_fail += 1
                results.append(
                    eval_pb2.EvaluationResult(
                        worker_id=self._worker_id,
                        fitness=outcome.fitness,
                    )
                )

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
