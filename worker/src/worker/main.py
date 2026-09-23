from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import grpc

from grpc_health.v1 import health, health_pb2, health_pb2_grpc

try:
    from . import eval_pb2_grpc
    from .aero import AerosandboxAeroEvaluator
    from .config import GENE_BOUNDS, load_config
    from .fitness import FitnessEvaluator
    from .service import EvaluatorServicer
except (ImportError, ValueError):
    import eval_pb2_grpc
    from aero import AerosandboxAeroEvaluator
    from config import GENE_BOUNDS, load_config
    from fitness import FitnessEvaluator
    from service import EvaluatorServicer


class JsonFormatter(logging.Formatter):
    """Output JSON log lines compatible with fluent-bit JSON parser and the monitor dashboard."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "level": record.levelname,
            "target": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields if present (e.g. worker_id, generation)
        for key in ("worker_id", "generation", "offset", "elapsed"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        island_id = getattr(record, "island_id", os.getenv("ISLAND_ID"))
        if island_id:
            payload["island_id"] = island_id
        role = getattr(record, "role", os.getenv("ROLE", "worker"))
        if role:
            payload["role"] = role
        worker_pool = getattr(record, "worker_pool", os.getenv("WORKER_POOL"))
        if worker_pool:
            payload["worker_pool"] = worker_pool

        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # 1. Stdout handler: human-readable plain text for `docker logs`
    stdout_handler = logging.StreamHandler()
    stdout_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler.setFormatter(stdout_formatter)
    root.addHandler(stdout_handler)

    # 2. Optional JSON file handler: for Fluent-Bit
    log_file_path = os.getenv("LOG_FILE_PATH")
    if not log_file_path and os.getenv("LOG_DIR"):
        worker_id = os.getenv("WORKER_ID", "worker")
        hostname = os.getenv("HOSTNAME", "unknown")
        log_file_path = os.path.join(os.getenv("LOG_DIR"), f"{worker_id}_{hostname}.log")

    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
        except Exception as e:
            sys.stderr.write(f"Failed to open log file '{log_file_path}': {e}\n")

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
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

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


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
