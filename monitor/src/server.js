const express = require('express');
const http = require('http');
const path = require('path');

const { createStore } = require('./store');
const { startConsumer } = require('./consumer');
const { buildApi } = require('./api');
const { attachWebSocket } = require('./ws');

const PORT = process.env.PORT || 3000;
const KAFKA_BROKERS = (process.env.KAFKA_BROKERS || 'kafka:9092').split(',');
const TOPICS = (process.env.KAFKA_TOPICS || 'logs.lead,logs.orchestrator,logs.worker').split(',');
const BUFFER_CAP = parseInt(process.env.BUFFER_CAP || '10000', 10);

async function main() {
  const store = createStore({ capPerSource: BUFFER_CAP });
  const app = express();
  const server = http.createServer(app);

  app.use(express.json());
  app.use(express.static(path.join(__dirname, '..', 'public')));

  buildApi(app, store);

  const broadcast = attachWebSocket(server, store);

  // Start HTTP server FIRST so the dashboard is always reachable,
  // even while Kafka is still booting or having issues.
  server.listen(PORT, () => {
    console.log(`monitor listening on :${PORT}`);
    console.log(`kafka brokers: ${KAFKA_BROKERS.join(',')}`);
    console.log(`topics: ${TOPICS.join(', ')}`);
  });

  // Launch the consumer in the background — don't await it.
  // It retries forever internally, so it'll catch up once Kafka is ready.
  startConsumer({
    brokers: KAFKA_BROKERS,
    topics: TOPICS,
    groupId: process.env.KAFKA_GROUP_ID || 'monitor-group',
    onLog: (log) => {
      store.push(log);
      broadcast(log);
    },
  }).catch((err) => console.error('consumer fatal:', err));

  const shutdown = async (sig) => {
    console.log(`\n${sig} received, shutting down...`);
    server.close();
    process.exit(0);
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
