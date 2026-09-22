How to run

```bash
# 1. Shared network & log volume (once)
docker network create simulated-lan
docker volume create app-logs

# 2. Infra first (until Kafka is healthy)
docker compose -f compose/docker-compose.infra.yml up -d --build

# 3. Lead cluster
docker compose -f compose/docker-compose.lead.yml up -d --build

# 4. Worker side
docker compose -f compose/docker-compose.worker.yml up -d --build --scale worker=4
docker compose -f compose/docker-compose.worker2.yml up -d --build --scale worker2=4
```

To simulate failure of a “remote” machine, just stop one project:

```bash
docker compose \
  -f compose/docker-compose.infra.yml \
  -f compose/docker-compose.lead.yml \
  -f compose/docker-compose.worker.yml \
  -f compose/docker-compose.worker2.yml \
  down -v --remove-orphans
```

or 

```bash
docker compose -p compose down -v --remove-orphans
```
