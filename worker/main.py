from __future__ import annotations

import asyncio
import logging
import os

import grpc

import eval_pb2_grpc
from aero import AerosandboxAeroEvaluator
from config import GENE_BOUNDS, load_config
from fitness import FitnessEvaluator
from service import EvaluatorServicer


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )
    # grpc/aio internals are noisy at DEBUG
    logging.getLogger("grpc").setLevel(logging.WARNING)
    return logging.getLogger("worker")


logger = _configure_logging()


async def serve() -> None:
    config = load_config()

    logger.info(
        "starting worker id=%s v_min_fuse_mm3=%.1f concurrency=%d gene_count=%d log_level=%s",
        config.worker_id,
        config.v_min_fuse_mm3,
        config.semaphore_size,
        len(GENE_BOUNDS),
        os.getenv("LOG_LEVEL", "INFO").upper(),
    )

    aero_evaluator = AerosandboxAeroEvaluator(config, logger.getChild("aero"))
    fitness_evaluator = FitnessEvaluator(
        aero_evaluator=aero_evaluator,
        config=config,
        logger=logger.getChild("fitness"),
    )

    server = grpc.aio.server()
    eval_pb2_grpc.add_EvaluatorServicer_to_server(
        EvaluatorServicer(
            fitness_evaluator=fitness_evaluator,
            worker_id=config.worker_id,
            gene_count=len(GENE_BOUNDS),
            logger=logger.getChild("service"),
            semaphore_size=config.semaphore_size,
        ),
        server,
    )

    server.add_insecure_port("[::]:50051")
    logger.info("worker %s listening on :50051", config.worker_id)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
