/**
 * In-memory ring buffer for logs, partitioned by source topic.
 * Keeps the last `capPerSource` logs per source so the API can serve
 * recent history without hitting Kafka.
 *
 * Additions over v1:
 *   - subscribe(fn) / unsubscribe(fn)  — called on every push; used by ws.js
 *                                        to throttle-push stats over WebSocket.
 *   - queryBefore(ts, filter)          — cursor-based history paging: returns
 *                                        up to `limit` logs whose ts < `before`.
 */
function createStore({ capPerSource = 10000 } = {}) {
  const buckets = new Map(); // source -> array of logs (newest at end)

  // Trim threshold: only sweep when at 125% capacity, then prune 20% off the front.
  const TRIGGER = Math.floor(capPerSource * 1.25);
  const KEEP    = Math.floor(capPerSource * 0.80);

  // Subscriber set for push notifications
  const subscribers = new Set();

  // Track latest best candidate across generations
  let latestCandidate = null;
  const candidateSubscribers = new Set();

  // ── Helpers ──────────────────────────────────────────────────────────────

  function bucket(source) {
    if (!buckets.has(source)) buckets.set(source, []);
    return buckets.get(source);
  }

  function parseLevel(s) {
    if (typeof s === 'string') return s.toUpperCase();
    if (typeof s === 'number') {
      // syslog-style: 7=DEBUG … 0=EMERG; tracing uses numbers sometimes
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

  // ── Public API ────────────────────────────────────────────────────────────

  function push(log) {
    const arr = bucket(log.source);
    arr.push(log);
    // Bulk trim: only fire when significantly over capacity
    if (arr.length > TRIGGER) {
      const remove = arr.length - KEEP;
      arr.splice(0, remove); // one O(n) sweep instead of per-push
    }
    // Notify subscribers (e.g. ws.js stats broadcaster)
    for (const fn of subscribers) {
      try { fn(log); } catch { /* ignore subscriber errors */ }
    }
  }

  /** Subscribe to every log push. Returns an unsubscribe function. */
  function subscribe(fn) {
    subscribers.add(fn);
    return () => subscribers.delete(fn);
  }

  /** Remove a previously-added subscriber. */
  function unsubscribe(fn) {
    subscribers.delete(fn);
  }

  /**
   * Standard query — newest-first, up to filter.limit.
   */
  function query(filter = {}) {
    const sources = filter.source ? [filter.source] : Array.from(buckets.keys());
    const out = [];
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

  /**
   * Cursor-based history page: returns up to `limit` logs whose ts < `before`.
   * Results are newest-first (relative to `before`).
   * Used by the "load older" scroll trigger and the /logs/history endpoint.
   */
  function queryBefore(before, filter = {}) {
    const limit = filter.limit || 100;
    const sources = filter.source ? [filter.source] : Array.from(buckets.keys());
    const out = [];
    for (const s of sources) {
      const arr = bucket(s);
      // Walk backwards from newest; skip until ts < before
      for (let i = arr.length - 1; i >= 0; i--) {
        const log = arr[i];
        if (before !== undefined && log.ts !== undefined && log.ts >= before) continue;
        if (matches(log, filter)) out.push(log);
        if (out.length >= limit) break;
      }
    }
    // Sort descending by ts so multi-source results are coherent
    out.sort((a, b) => (b.ts || 0) - (a.ts || 0));
    return out.slice(0, limit);
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

  function setLatestCandidate(cand) {
    latestCandidate = cand;
    for (const fn of candidateSubscribers) {
      try { fn(cand); } catch { /* ignore subscriber errors */ }
    }
  }

  function getLatestCandidate() {
    return latestCandidate;
  }

  function onCandidate(fn) {
    candidateSubscribers.add(fn);
    return () => candidateSubscribers.delete(fn);
  }

  return {
    push,
    subscribe,
    unsubscribe,
    query,
    queryBefore,
    stats,
    sources,
    setLatestCandidate,
    getLatestCandidate,
    onCandidate,
  };
}

module.exports = { createStore };
