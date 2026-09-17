# syntax=docker/dockerfile:1
FROM rust:1.97-slim AS builder
WORKDIR /app

# protoc is required by tonic-build; g++ and OpenMP are required by cc
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    g++ \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* build.rs ./
COPY proto ./proto
COPY native ./native
COPY benches ./benches
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp /app/target/release/surrogate_node /app/surrogate_node

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    netcat-openbsd \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 10001 appgroup \
 && useradd -u 10001 -g appgroup -s /bin/sh -m appuser

COPY --from=builder /app/surrogate_node /usr/local/bin/surrogate_node

USER appuser:appgroup

ENV RUST_LOG=info
EXPOSE 50054
CMD ["surrogate_node"]
