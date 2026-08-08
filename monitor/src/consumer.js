const { Kafka } = require('kafkajs');

async function startConsumer({ brokers, topics, groupId, onLog }) {
  const kafka = new Kafka({
    brokers,
    clientId: 'loadbalancega-monitor',
    retry: {
      initialRetryTime: 1000,
      maxRetryTime: 30000,
      retries: 1000,
    },
  });

  const consumer = kafka.consumer({ groupId });

  for (let attempt = 1; ; attempt++) {
    try {
      await consumer.connect();
      console.log(`kafka connected (attempt ${attempt})`);

      for (const topic of topics) {
        await consumer.subscribe({ topic, fromBeginning: false });
        console.log(`subscribed to ${topic}`);
      }

      await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          let log;
          try {
            log = JSON.parse(message.value.toString());
          } catch {
            log = { message: message.value.toString(), raw: true };
          }
          log.source = topic;
          log.partition = partition;
          log.offset = message.offset;

          // Normalize timestamp: prefer the payload's `timestamp` (ISO string),
          // then `ts` (ms epoch), then Kafka's message timestamp, then Date.now().
          if (!log.ts) {
            if (log.timestamp) {
              // ISO 8601 string → ms epoch
              log.ts = new Date(log.timestamp).getTime();
            } else {
              const t = message.timestamp;
              if (t) log.ts = parseInt(t, 10);
              else log.ts = Date.now();
            }
          }

          if (!log.level && log.levelStr) log.level = log.levelStr;
          if (!log.message && log.msg) log.message = log.msg;
          if (!log.timestamp && log.ts) log.timestamp = new Date(log.ts).toISOString();

          try {
            onLog(log);
          } catch (err) {
            console.error('onLog handler error:', err);
          }
        },
      });

      console.log('consumer running');
      break;
    } catch (err) {
      console.error(`consumer setup failed (attempt ${attempt}): ${err.message}`);
      try { await consumer.disconnect(); } catch {}
      await new Promise((r) => setTimeout(r, Math.min(2000 * attempt, 15000)));
    }
  }

  consumer.on(consumer.events.DISCONNECT, async () => {
    console.warn('kafka consumer disconnected, reconnecting in 3s...');
    await new Promise((r) => setTimeout(r, 3000));
    try {
      await consumer.connect();
    } catch (e) {
      console.error('reconnect failed:', e.message);
    }
  });

  return consumer;
}

module.exports = { startConsumer };
