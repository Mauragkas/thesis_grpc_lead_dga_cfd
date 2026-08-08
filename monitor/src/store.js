/**
 * In-memory ring buffer for logs, partitioned by source topic.
 * Keeps the last `capPerSource` logs per source so the API can serve
 * recent history without hitting Kafka.
 */
function createStore({ capPerSource = 10000 } = {}) {
  const buckets = new Map(); // source -> array of logs (newest at end)

  function bucket(source) {
    if (!buckets.has(source)) buckets.set(source, []);
    return buckets.get(source);
  }

  function push(log) {
    const arr = bucket(log.source);
    arr.push(log);
    if (arr.length > capPerSource) {
      arr.splice(0, arr.length - capPerSource);
    }
  }

  function parseLevel(s) {
    if (typeof s === 'string') return s.toUpperCase();
    if (typeof s === 'number') {
      // syslog-style: 7=DEBUG ... 0=EMERG; tracing uses numbers sometimes
      const names = ['EMERG','ALERT','CRIT','ERROR','WARN','NOTICE','INFO','DEBUG'];
      return names[s] || String(s);
    }
    return String(s ?? '').toUpperCase();
  }

  function matches(log, filter) {
    if (filter.source && log.source !== filter.source) return false;
    if (filter.level) {
      const want = filter.level.toUpperCase();
      const got = parseLevel(log.level);
      if (want === 'ERROR') {
        if (!['ERROR','CRIT','ALERT','EMERG'].includes(got)) return false;
      } else if (want === 'WARN') {
        if (!['WARN','WARNING','ERROR','CRIT','ALERT','EMERG'].includes(got)) return false;
      } else if (got !== want) {
        return false;
      }
    }
    if (filter.since && log.ts && log.ts < filter.since) return false;
    if (filter.until && log.ts && log.ts > filter.until) return false;
    if (filter.search) {
      const hay = JSON.stringify(log).toLowerCase();
      if (!hay.includes(filter.search.toLowerCase())) return false;
    }
    if (filter.target && log.target !== filter.target) return false;
    if (filter.workerId && log.worker_id !== filter.workerId) return false;
    return true;
  }

  function query(filter = {}) {
    const sources = filter.source ? [filter.source] : Array.from(buckets.keys());
    let out = [];
    for (const s of sources) {
      const arr = bucket(s);
      // iterate newest-first
      for (let i = arr.length - 1; i >= 0; i--) {
        if (matches(arr[i], filter)) out.push(arr[i]);
        if (filter.limit && out.length >= filter.limit) return out;
      }
    }
    return out;
  }

  function stats() {
    const bySource = {};
    const byLevel = {};
    let total = 0;
    let newest = 0;
    for (const [source, arr] of buckets.entries()) {
      bySource[source] = arr.length;
      total += arr.length;
      if (arr.length > 0) {
        const last = arr[arr.length - 1];
        if (last.ts && last.ts > newest) newest = last.ts;
        for (const l of arr) {
          const lvl = parseLevel(l.level);
          byLevel[lvl] = (byLevel[lvl] || 0) + 1;
        }
      }
    }
    return { total, bySource, byLevel, newestTs: newest, capPerSource };
  }

  function sources() {
    return Array.from(buckets.keys());
  }

  return { push, query, stats, sources };
}

module.exports = { createStore };
