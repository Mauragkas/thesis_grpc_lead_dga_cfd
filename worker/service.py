from __future__ import annotations

import asyncio
import logging

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
        async with self._semaphore:
            genes = list(request.genes)
            if len(genes) != self._gene_count:
                self._logger.warning(
                    "Worker '%s' got %d genes, expected %d",
                    self._worker_id,
                    len(genes),
                    self._gene_count,
                )
                return eval_pb2.EvaluationResult(worker_id=self._worker_id, fitness=-1e9)

            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(
                None,
                self._fitness_evaluator.evaluate_genes,
                genes,
            )

            self._logger.info(
                "Worker '%s' | V_fuse=%.0f mm^3 | fit=%.3f",
                self._worker_id,
                outcome.fuselage_volume_mm3,
                outcome.fitness,
            )
            return eval_pb2.EvaluationResult(
                worker_id=self._worker_id,
                fitness=outcome.fitness,
            )

    async def EvaluateBatch(self, request, context):
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            results = []

            for ind in request.individuals:
                genes = list(ind.genes)
                if len(genes) != self._gene_count:
                    results.append(
                        eval_pb2.EvaluationResult(worker_id=self._worker_id, fitness=-1e9)
                    )
                    continue

                outcome = await loop.run_in_executor(
                    None,
                    self._fitness_evaluator.evaluate_genes,
                    genes,
                )
                results.append(
                    eval_pb2.EvaluationResult(
                        worker_id=self._worker_id,
                        fitness=outcome.fitness,
                    )
                )

            return eval_pb2.BatchResponse(results=results)
