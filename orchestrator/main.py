import asyncio
import logging
import os
import random
import time

import grpc
import eval_pb2
import eval_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

EVAL_ENDPOINT = os.getenv("EVAL_ENDPOINT", "load-balancer:50051")
POP_SIZE = 20
GENES_LEN = 5
GENERATIONS = 5

GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]

def create_individual():
    return [random.uniform(-5.0, 5.0) for _ in range(GENES_LEN)]

async def eval_individual(stub: eval_pb2_grpc.EvaluatorStub, genes: list[float]) -> float:
    request = eval_pb2.Individual(genes=genes)
    response = await stub.Evaluate(request)
    return response.fitness

async def evaluate_population(stub, population):
    tasks = [eval_individual(stub, ind) for ind in population]
    return await asyncio.gather(*tasks)

async def run_ga():
    start_time = time.perf_counter()
    population = [create_individual() for _ in range(POP_SIZE)]
    async with grpc.aio.insecure_channel(EVAL_ENDPOINT, options=GRPC_CHANNEL_OPTIONS) as channel:
        stub = eval_pb2_grpc.EvaluatorStub(channel)
        for gen in range(GENERATIONS):
            logger.info(f"Evaluating generation {gen + 1}/{GENERATIONS}...")
            fitnesses = await evaluate_population(stub, population)
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            print(f"Gen {gen + 1}/{GENERATIONS} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

            sorted_pop = [x for _, x in sorted(zip(fitnesses, population))]
            survivors = sorted_pop[: POP_SIZE // 2]
            next_gen = survivors.copy()
            while len(next_gen) < POP_SIZE:
                parent = random.choice(survivors)
                child = [g + random.gauss(0, 0.1) for g in parent]
                next_gen.append(child)
            population = next_gen
    elapsed = time.perf_counter() - start_time
    print(f"Finished GA run in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    print("Starting GA run with Envoy gRPC proxy...")
    asyncio.run(run_ga())
