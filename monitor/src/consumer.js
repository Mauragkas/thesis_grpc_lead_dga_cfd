const { Kafka } = require('kafkajs');
const { decodeGenes } = require('./planeGeometry');

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
          let raw;
          try {
            raw = JSON.parse(message.value.toString());
          } catch {
            raw = { message: message.value.toString(), raw: true };
          }
          const log = normalizeLog(raw, {
            topic,
            partition,
            offset: message.offset,
            timestamp: message.timestamp,
          });

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

/**
 * Normalizes a log object from Kafka / Fluent-bit into a clean, standard structure:
 * {
 *   source: string,
 *   source_tag?: string,
 *   level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG',
 *   message: string,
 *   target: string,
 *   ts: number (epoch ms),
 *   timestamp: string (ISO),
 *   candidate?: object,
 *   ...rest
 * }
 */
function normalizeLog(raw, meta = {}) {
  const log = typeof raw === 'object' && raw !== null ? { ...raw } : { message: String(raw), raw: true };

  log.source = meta.topic || log.source || 'unknown';
  if (meta.partition !== undefined) log.partition = meta.partition;
  if (meta.offset !== undefined) log.offset = meta.offset;

  // 1. Container identity (e.g. "/compose-orchestrator-1" -> "compose-orchestrator-1")
  if (log.container_name && typeof log.container_name === 'string') {
    log.source_tag = log.container_name.replace(/^\//, '');
  }

  // 2. Unpack Docker "log" field if it contains a serialized JSON record
  if (typeof log.log === 'string') {
    const trimmed = log.log.trim();
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (typeof parsed === 'object' && parsed !== null) {
          if (!log.level && (parsed.level || parsed.levelStr)) log.level = parsed.level || parsed.levelStr;
          if (!log.message && (parsed.message || parsed.msg)) log.message = parsed.message || parsed.msg;
          if (!log.target && (parsed.target || parsed.logger || parsed.name)) {
            log.target = parsed.target || parsed.logger || parsed.name;
          }
          if (!log.fields && parsed.fields) log.fields = parsed.fields;
          if (!log.timestamp && parsed.timestamp) log.timestamp = parsed.timestamp;
          if (!log.ts && parsed.ts) log.ts = parsed.ts;
          if (parsed.best_genome && !log.best_genome) log.best_genome = parsed.best_genome;
        }
      } catch {}
    }
  }

  // 3. Normalize message text from any source field
  if (!log.message) {
    if (log.msg) {
      log.message = log.msg;
    } else if (log.fields && typeof log.fields.message === 'string') {
      log.message = log.fields.message;
    } else if (log.fields && typeof log.fields.msg === 'string') {
      log.message = log.fields.msg;
    } else if (log.log) {
      log.message = typeof log.log === 'string' ? log.log.trim() : JSON.stringify(log.log);
    } else {
      log.message = '';
    }
  }
  if (typeof log.message === 'string') {
    log.message = log.message.replace(/\r?\n$/, '');
  }

  // 4. Normalize timestamp
  if (!log.ts) {
    if (log.timestamp) {
      const parsed = new Date(log.timestamp).getTime();
      if (!Number.isNaN(parsed)) log.ts = parsed;
    } else if (log['@timestamp']) {
      if (typeof log['@timestamp'] === 'number') {
        log.ts = log['@timestamp'] < 1e11 ? log['@timestamp'] * 1000 : log['@timestamp'];
      } else {
        const parsed = new Date(log['@timestamp']).getTime();
        if (!Number.isNaN(parsed)) log.ts = parsed;
      }
    } else if (meta.timestamp) {
      const t = parseInt(meta.timestamp, 10);
      if (!Number.isNaN(t)) log.ts = t;
    }
    if (!log.ts) log.ts = Date.now();
  }
  if (!log.timestamp) {
    log.timestamp = new Date(log.ts).toISOString();
  }

  // 5. Normalize level
  if (!log.level) {
    if (log.levelStr) {
      log.level = log.levelStr;
    } else if (log.fields && (log.fields.level || log.fields.levelStr)) {
      log.level = log.fields.level || log.fields.levelStr;
    } else if (log.severity) {
      log.level = log.severity;
    } else if (log.stream === 'stderr') {
      log.level = 'WARN';
    } else if (typeof log.message === 'string') {
      const lvlMatch = log.message.match(/^(?:\[?(ERROR|CRIT|FATAL|WARN|WARNING|INFO|DEBUG|TRACE)\]?[:\s])/i);
      if (lvlMatch) {
        const m = lvlMatch[1].toUpperCase();
        log.level = (m === 'WARNING') ? 'WARN' : (m === 'CRIT' || m === 'FATAL') ? 'ERROR' : m;
      } else {
        log.level = 'INFO';
      }
    } else {
      log.level = 'INFO';
    }
  }
  log.level = String(log.level).toUpperCase();

  // 6. Normalize target
  if (!log.target) {
    if (log.fields && log.fields.target) {
      log.target = log.fields.target;
    } else if (log.name) {
      log.target = log.name;
    } else if (log.logger) {
      log.target = log.logger;
    } else if (log.source_tag) {
      log.target = log.source_tag;
    } else if (log.source) {
      log.target = log.source;
    } else {
      log.target = '';
    }
  }

  // 7. Extract candidate genome if present
  const candidate = extractCandidate(log);
  if (candidate) {
    log.candidate = candidate;
  }

  return log;
}

function extractCandidate(log) {
  let genome = null;
  if (Array.isArray(log.best_genome)) {
    genome = log.best_genome;
  } else if (log.fields && Array.isArray(log.fields.best_genome)) {
    genome = log.fields.best_genome;
  } else if (typeof log.best_genome === 'string') {
    try {
      const parsed = JSON.parse(log.best_genome);
      if (Array.isArray(parsed)) genome = parsed;
    } catch {}
  } else if (log.fields && typeof log.fields.best_genome === 'string') {
    try {
      const parsed = JSON.parse(log.fields.best_genome);
      if (Array.isArray(parsed)) genome = parsed;
    } catch {}
  }

  const textToScan =
    (typeof log.message === 'string' ? log.message : '') ||
    (typeof log.log === 'string' ? log.log : '') ||
    (log.fields && typeof log.fields.message === 'string' ? log.fields.message : '');

  if (!genome && textToScan) {
    const m = textToScan.match(/best[\s_]*(?:candidate[\s_]*)?genome\s*[:=]\s*(\[[^\]]+\])/i);
    if (m) {
      try {
        const parsed = JSON.parse(m[1]);
        if (Array.isArray(parsed)) genome = parsed;
      } catch {}
    }
  }

  if (!genome) return null;

  let gen = log.gen || (log.fields && log.fields.gen);
  if (!gen && textToScan) {
    const mGen = textToScan.match(/Gen\s+(\d+)/i);
    if (mGen) gen = parseInt(mGen[1], 10);
  }

  let fitness =
    log.best_ever !== undefined ? log.best_ever :
    log.fields?.best_ever !== undefined ? log.fields.best_ever :
    log.best !== undefined ? log.best :
    log.fields?.best !== undefined ? log.fields.best : undefined;

  if (fitness === undefined && textToScan) {
    const mFitEver = textToScan.match(/best_ever\s*[:=]\s*([-0-9.eE+]+)/i);
    if (mFitEver) {
      fitness = parseFloat(mFitEver[1]);
    } else {
      const mFit = textToScan.match(/best\s*[:=]\s*([-0-9.eE+]+)/i);
      if (mFit) fitness = parseFloat(mFit[1]);
    }
  }

  return {
    gen: gen || 0,
    fitness: fitness !== undefined ? Number(fitness) : null,
    genome,
    decoded: decodeGenes(genome),
    ts: log.ts || Date.now(),
  };
}

module.exports = { startConsumer, normalizeLog, extractCandidate };
