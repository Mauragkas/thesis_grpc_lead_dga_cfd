# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies using frozen lockfile with uv cache mount
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY eval.proto ./
RUN .venv/bin/python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. eval.proto

FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/eval_pb2*.py ./
COPY *.py ./

CMD ["python", "main.py"]
