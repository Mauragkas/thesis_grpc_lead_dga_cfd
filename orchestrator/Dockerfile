# syntax=docker/dockerfile:1
FROM rust:1.97-slim AS builder
WORKDIR /app

# protoc is required by tonic-build at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* build.rs ./
COPY proto ./proto
COPY benches ./benches
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/orchestrator /usr/local/bin/orchestrator
CMD ["orchestrator"]
