# syntax=docker/dockerfile:1
FROM rust:1.80-slim AS builder
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

RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    netcat-openbsd \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/surrogate_node /usr/local/bin/surrogate_node
ENV RUST_LOG=info
EXPOSE 50054
CMD ["surrogate_node"]
