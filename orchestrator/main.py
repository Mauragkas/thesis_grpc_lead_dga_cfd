import asyncio
import logging
import os
import random
import time
import math

import grpc
import eval_pb2
import eval_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

EVAL_ENDPOINT = os.getenv("EVAL_ENDPOINT", "load-balancer:50051")
POP_SIZE = 60
GENES_LEN = 10         # matches GENE_BOUNDS in worker
GENERATIONS = 10
MUT_SIGMA = 0.08       # in normalized gene space
ELITE_FRAC = 0.5
# BATCH_SIZE = max(1, math.ceil(POP_SIZE / int(os.getenv("N_WORKERS", "1"))))
BATCH_SIZE = 30

# Reproducible GA across runs / worker counts
SEED = int(os.getenv("GA_SEED", "42"))
random.seed(SEED)

GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]


def create_individual():
    # Random genes in [0, 1]
    return [random.random() for _ in range(GENES_LEN)]


async def eval_batch(stub, individuals, max_attempts=30):
    req = eval_pb2.BatchRequest(
        individuals=[eval_pb2.Individual(genes=g) for g in individuals]
    )
    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = await stub.EvaluateBatch(req, timeout=120)
            return [r.fitness for r in resp.results]
        except grpc.aio.AioRpcError as e:
            last_err = e
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.info(f"upstream not ready (attempt {attempt+1}), retrying...")
                await asyncio.sleep(2.0)
                continue
            raise
    raise last_err


async def evaluate_population(stub, population):
    # split into chunks, fire one batch RPC per chunk in parallel
    chunks = [population[i:i+BATCH_SIZE] for i in range(0, len(population), BATCH_SIZE)]
    results = await asyncio.gather(*[eval_batch(stub, c) for c in chunks])
    # flatten
    flat = []
    for r in results:
        flat.extend(r)
    return flat


def clip(x):
    return max(0.0, min(1.0, x))


async def wait_for_channel(channel, timeout=120):
    """Block until the gRPC channel (via Envoy -> worker) is ready."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
            return
        except (asyncio.TimeoutError, grpc.aio.AioRpcError):
            logger.info("Waiting for worker to become ready via Envoy...")
            await asyncio.sleep(2.0)
    raise RuntimeError("Channel did not become ready in time.")


async def run_ga():
    random.seed(SEED)  # ensures identical individuals/mutations every run
    start_time = time.perf_counter()
    population = [create_individual() for _ in range(POP_SIZE)]
    best_ever = float("-inf")
    async with grpc.aio.insecure_channel(EVAL_ENDPOINT, options=GRPC_CHANNEL_OPTIONS) as channel:
        logger.info("Waiting for evaluator endpoint to be ready...")
        await wait_for_channel(channel)
        stub = eval_pb2_grpc.EvaluatorStub(channel)
        for gen in range(GENERATIONS):
            logger.info(f"Evaluating generation {gen+1}/{GENERATIONS}...")
            fitnesses = await evaluate_population(stub, population)

            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_ever = max(best_ever, best_fitness)
            print(f"Gen {gen+1}/{GENERATIONS} | Best: {best_fitness:.4f} | "
                  f"Avg: {avg_fitness:.4f} | BestEver: {best_ever:.4f}")

            sorted_pop = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
            n_survivors = max(2, int(POP_SIZE * ELITE_FRAC))
            survivors = sorted_pop[:n_survivors]
            next_gen = survivors.copy()
            while len(next_gen) < POP_SIZE:
                parent = random.choice(survivors)
                child = [clip(g + random.gauss(0, MUT_SIGMA)) for g in parent]
                next_gen.append(child)
            population = next_gen

    elapsed = time.perf_counter() - start_time
    print(f"Finished GA run in {elapsed:.2f}s. Best fitness: {best_ever:.4f}")


if __name__ == "__main__":
    print("Starting aero GA run (hard volume constraint) via Envoy gRPC proxy...")
    asyncio.run(run_ga())
