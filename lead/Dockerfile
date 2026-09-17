# syntax=docker/dockerfile:1
FROM rust:1.97-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
COPY build.rs ./
COPY proto ./proto
COPY benches ./benches
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp /app/target/release/lead_node /app/lead_node

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 10001 appgroup \
 && useradd -u 10001 -g appgroup -s /bin/sh -m appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appgroup /app/lead_node /app/lead_node

USER appuser:appgroup

EXPOSE 8080 50051
CMD ["./lead_node"]
