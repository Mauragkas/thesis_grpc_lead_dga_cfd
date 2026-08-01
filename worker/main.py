from __future__ import annotations

import asyncio
import logging

import grpc

import eval_pb2_grpc
from aero import AerosandboxAeroEvaluator
from config import GENE_BOUNDS, load_config
from fitness import FitnessEvaluator
from service import EvaluatorServicer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("worker")


async def serve() -> None:
    config = load_config()

    aero_evaluator = AerosandboxAeroEvaluator(config)
    fitness_evaluator = FitnessEvaluator(aero_evaluator=aero_evaluator, config=config)

    server = grpc.aio.server()
    eval_pb2_grpc.add_EvaluatorServicer_to_server(
        EvaluatorServicer(
            fitness_evaluator=fitness_evaluator,
            worker_id=config.worker_id,
            gene_count=len(GENE_BOUNDS),
            logger=logger,
            semaphore_size=config.semaphore_size,
        ),
        server,
    )

    server.add_insecure_port("[::]:50051")
    logger.info(
        "Worker '%s' running aero evaluator on :50051 (V_MIN_FUSE_MM3=%.1f)",
        config.worker_id,
        config.v_min_fuse_mm3,
    )

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
