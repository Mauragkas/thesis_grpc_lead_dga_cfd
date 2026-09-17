# syntax=docker/dockerfile:1
FROM rust:1.97-slim AS builder
WORKDIR /app

# protoc is required by tonic-build at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

COPY hilbert ./hilbert
COPY orchestrator/Cargo.toml orchestrator/Cargo.lock* orchestrator/build.rs ./orchestrator/
COPY orchestrator/proto ./orchestrator/proto
COPY orchestrator/benches ./orchestrator/benches
COPY orchestrator/src ./orchestrator/src

WORKDIR /app/orchestrator

ENV CARGO_TARGET_DIR=/app/target

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp /app/target/release/orchestrator /app/bin-orchestrator

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 10001 appgroup \
 && useradd -u 10001 -g appgroup -s /bin/sh -m appuser

WORKDIR /app
COPY --from=builder /app/bin-orchestrator /usr/local/bin/orchestrator

USER appuser:appgroup

CMD ["orchestrator"]
