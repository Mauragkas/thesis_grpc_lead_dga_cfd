# Docker Compose & Distributed Deployment

This directory contains the multi-container configuration files to run the distributed system in an isolated multi-island simulated LAN network.

## System Architecture Diagram

```mermaid
flowchart TB
    subgraph LAN["External Docker Network: simulated-lan"]

        subgraph INFRA["1. Observability & Event Streaming (docker-compose.infra.yml)"]
            Kafka["Kafka Broker (KRaft)\napache/kafka:3.9.0\nPort: 9092\nTopics: logs.lead, logs.orchestrator, logs.worker"]
            FluentBit["Fluent-Bit Log Forwarder\nfluent/fluent-bit:3.0\nPort: 24224 (fluentd input)\nRoutes by tag to Kafka topics"]
            Monitor["Monitor Web Dashboard\nNode.js / Express / WebSockets\nPort: 3000 (UI)\nConsumes Kafka logs"]

            FluentBit -->|Structured JSON logs| Kafka
            Kafka -->|Stream log events| Monitor
        end

        subgraph LEAD_CLUSTER["2. LEAD Distributed DHT Ring (docker-compose.lead.yml)"]
            direction LR
            Lead1["lead-node1 (Seed)\nHTTP: 2001 | gRPC: 50051\n20 vnodes (VIDs 0..19)"]
            Lead2["lead-node2\nHTTP: 2002 | gRPC: 50052\n20 vnodes (VIDs 20..39)\nJoins lead-node1"]
            Lead3["lead-node3\nHTTP: 2003 | gRPC: 50053\n20 vnodes (VIDs 40..59)\nJoins lead-node1"]

            Lead1 <==>|Chord Ring &\nFedAvg Sync| Lead2
            Lead2 <==>|Chord Ring &\nFedAvg Sync| Lead3
            Lead3 <==>|Chord Ring &\nFedAvg Sync| Lead1
        end

        subgraph ISLAND1["3. GA Island 1 (docker-compose.worker.yml)"]
            Orch1["Orchestrator 1 (Rust)\nGA Seed: 42 | Ring: 50060\nIsland Model GA Driver"]
            Envoy1["Envoy Load Balancer 1\nROUND_ROBIN gRPC\nPort: 50051"]
            subgraph WORKER_POOL1["Worker Pool 1 (--scale worker=N)"]
                W1_1["worker (replica 1)\nPython + AeroSandbox"]
                W1_2["worker (replica 2)\nPython + AeroSandbox"]
                W1_N["worker (replica N)\nPython + AeroSandbox"]
            end

            Orch1 -->|EvaluateBatch| Envoy1
            Envoy1 -->|gRPC Round-Robin| W1_1
            Envoy1 -->|gRPC Round-Robin| W1_2
            Envoy1 -->|gRPC Round-Robin| W1_N
        end

        subgraph ISLAND2["4. GA Island 2 (docker-compose.worker2.yml)"]
            Orch2["Orchestrator 2 (Rust)\nGA Seed: 43 | Ring: 50060\nJoins Orch 1 Ring"]
            Envoy2["Envoy Load Balancer 2\nROUND_ROBIN gRPC\nPort: 50051 (Host: 50052)"]
            subgraph WORKER_POOL2["Worker Pool 2 (--scale worker2=N)"]
                W2_1["worker2 (replica 1)\nPython + AeroSandbox"]
                W2_2["worker2 (replica 2)\nPython + AeroSandbox"]
                W2_N["worker2 (replica N)\nPython + AeroSandbox"]
            end

            Orch2 -->|EvaluateBatch| Envoy2
            Envoy2 -->|gRPC Round-Robin| W2_1
            Envoy2 -->|gRPC Round-Robin| W2_2
            Envoy2 -->|gRPC Round-Robin| W2_N
        end

        %% Cross-subsystem links
        Orch1 <==>|Island Migration\nChord Ring gRPC: 50060| Orch2

        Orch1 -.->|PutRouted / RangeQuery| Lead1
        Orch2 -.->|PutRouted / RangeQuery| Lead1

        %% Logging streams
        Lead1 -.->|tag: lead.node1| FluentBit
        Lead2 -.->|tag: lead.node2| FluentBit
        Lead3 -.->|tag: lead.node3| FluentBit
        Orch1 -.->|tag: orchestrator.main| FluentBit
        Orch2 -.->|tag: orchestrator2.main| FluentBit
        W1_1 -.->|tag: worker.main| FluentBit
        W2_1 -.->|tag: worker2.main| FluentBit
    end
```

## Compose Services Breakdown

### 1. Infrastructure (`docker-compose.infra.yml`)
- **`kafka`**: Apache Kafka 3.9.0 running in KRaft mode (no Zookeeper). Exposes port `9092` on the internal network and host.
- **`fluent-bit`**: Fluent-Bit 3.0 configured with dual-output ingestion: tails structured JSON log files from the shared volume `app-logs` (`/var/log/app/*.log`) and listens on port `24224` (Docker forward fallback). Routes logs by tag to Kafka topics:
  - `lead.*` $\to$ `logs.lead`
  - `orchestrator.*` $\to$ `logs.orchestrator`
  - `worker.*` $\to$ `logs.worker`
  - `surrogate.*` $\to$ `logs.surrogate`
- **`monitor`**: Node/Express dashboard on port `3000` consuming Kafka topics in real time, serving the live GA candidate metrics, log console, and interactive Three.js 3D plane geometry viewer.

### 2. LEAD DHT Cluster (`docker-compose.lead.yml`)
- 3 peer containers (`lead-node1`, `lead-node2`, `lead-node3`), each hosting 100 virtual nodes (300 vnodes total on the ring) with durable Sled storage.
- Node 1 serves as the bootstrap node (`SELF_URI=http://lead-node1:50051`); Nodes 2 and 3 join via `JOIN_URI=http://lead-node1:50051`.
- Exposes HTTP debugging endpoints on ports `2001`, `2002`, and `2003`.

### 3. GA Island 1 (`docker-compose.worker.yml`)
- **`orchestrator`**: GA driver initialized with `GA_SEED=42`. Binds ring membership at `0.0.0.0:50060`.
- **`surrogate-node`**: Aerodynamic MLP surrogate microservice running on port `50054` for Tier 2 evaluation and active online retraining.
- **`load-balancer`**: Envoy proxy balancing evaluation requests across all `worker` container replicas using DNS service discovery (`STRICT_DNS`).
- **`worker`**: Python gRPC evaluation service with AeroSandbox aerodynamics, powertrain modeling, and mission simulation. Scaled using `--scale worker=N`.

### 4. GA Island 2 (`docker-compose.worker2.yml`)
- **`orchestrator2`**: GA driver with `GA_SEED=43`. Joins Island 1's ring via `RING_BOOTSTRAP=http://orchestrator:50060` and exchanges migrants every 5 generations.
- **`load-balancer2`**: Envoy proxy (mapped to host port `50052` to avoid collision).
- **`worker2`**: Second independent pool of Python evaluation workers.

---

## How to Run

```bash
# 1. Create shared network and shared log volume
docker network create simulated-lan
docker volume create app-logs

# 2. Start infra (wait for Kafka healthcheck)
docker compose -f compose/docker-compose.infra.yml up -d --build

# 3. Start LEAD DHT cluster
docker compose -f compose/docker-compose.lead.yml up -d --build

# 4. Start GA Island 1 and Island 2 with scaled workers
docker compose -f compose/docker-compose.worker.yml up -d --build --scale worker=4
docker compose -f compose/docker-compose.worker2.yml up -d --build --scale worker2=4
```

## How to Stop

```bash
docker compose \
  -f compose/docker-compose.infra.yml \
  -f compose/docker-compose.lead.yml \
  -f compose/docker-compose.worker.yml \
  -f compose/docker-compose.worker2.yml \
  down -v --remove-orphans
```
