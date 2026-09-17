# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install grpc_health_probe for robust gRPC health checking
ADD --chmod=755 https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.37/grpc_health_probe-linux-amd64 /usr/local/bin/grpc_health_probe

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY eval.proto ./worker/eval.proto

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN .venv/bin/python -m grpc_tools.protoc -I. --python_out=src --grpc_python_out=src worker/eval.proto

FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 libgfortran5 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 10001 appgroup \
 && useradd -u 10001 -g appgroup -s /bin/sh -m appuser

COPY --from=builder /usr/local/bin/grpc_health_probe /usr/local/bin/grpc_health_probe
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=appuser:appgroup /app/src ./src
COPY --chown=appuser:appgroup pyproject.toml README.md ./

USER appuser:appgroup

HEALTHCHECK --interval=3s --timeout=3s --retries=20 --start-period=5s \
    CMD ["/usr/local/bin/grpc_health_probe", "-addr=:50051"]

CMD ["python", "-m", "worker.main"]
