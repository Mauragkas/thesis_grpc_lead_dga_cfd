import asyncio
import logging
import os

import grpc

import eval_pb2
import eval_pb2_grpc

# Set up stdout logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("worker")

WORKER_ID = os.getenv("WORKER_ID") or os.getenv("HOSTNAME", "unknown")
_semaphore = asyncio.Semaphore(1)

class EvaluatorServicer(eval_pb2_grpc.EvaluatorServicer):
    async def Evaluate(self, request, context):
        async with _semaphore:
            logger.info(f"Worker '{WORKER_ID}' processing evaluation request for {len(request.genes)} genes")

            # Simulate compute overhead
            await asyncio.sleep(0.5)

            # Sphere function calculation
            fitness = sum(x ** 2 for x in request.genes)

            logger.info(f"Worker '{WORKER_ID}' completed evaluation | Fitness: {fitness:.4f}")

            return eval_pb2.EvaluationResult(
                worker_id=WORKER_ID,
                fitness=fitness
            )

async def serve():
    server = grpc.aio.server()
    eval_pb2_grpc.add_EvaluatorServicer_to_server(EvaluatorServicer(), server)
    server.add_insecure_port("[::]:50051")
    logger.info(f"Worker '{WORKER_ID}' running gRPC server on port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
