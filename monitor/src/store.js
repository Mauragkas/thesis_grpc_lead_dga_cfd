/**
 * In-memory ring buffer for logs, partitioned by source topic.
 * Keeps the last `capPerSource` logs per source so the API can serve
 * recent history without hitting Kafka.
 *
 * Features:
 *   - subscribe(fn) / unsubscribe(fn)  — called on every push; used by ws.js
 *                                        to throttle-push stats over WebSocket.
 *   - queryBefore(ts, filter)          — cursor-based history paging: returns
 *                                        up to `limit` logs whose ts < `before`.
 *   - Island & Role isolation          — partition / filter by island and role;
 *                                        per-island candidate tracking.
 *   - Best Candidate Tracking          — tracks both latest candidate and the
 *                                        best candidate overall (highest fitness).
 */
function createStore({ capPerSource = 10000 } = {}) {
  const buckets = new Map(); // source -> array of logs (newest at end)

  // Trim threshold: only sweep when at 125% capacity, then prune 20% off the front.
  const TRIGGER = Math.floor(capPerSource * 1.25);
  const KEEP    = Math.floor(capPerSource * 0.80);

  // Subscriber set for push notifications
  const subscribers = new Set();

  // Track latest and best candidates across generations and islands
  let latestCandidate = null;
  let bestCandidate = null;
  const candidatesByIsland = new Map();     // island -> latest candidate
  const bestCandidatesByIsland = new Map(); // island -> best candidate
  const candidateSubscribers = new Set();
  const discoveredIslands = new Set();

  // ── Helpers ───────────────────────────────────────────────────────────────

  function bucket(source) {
    if (!buckets.has(source)) buckets.set(source, []);
    return buckets.get(source);
  }

  function parseLevel(s) {
    if (typeof s === 'string') return s.toUpperCase();
    if (typeof s === 'number') {
      const names = ['EMERG','ALERT','CRIT','ERROR','WARN','NOTICE','INFO','DEBUG'];
      return names[s] || String(s);
    }
    return String(s ?? '').toUpperCase();
  }

  function matches(log, filter) {
    if (filter.source && log.source !== filter.source) return false;
    if (filter.island && log.island !== filter.island) return false;
    if (filter.role && log.role !== filter.role) return false;
    if (filter.container && log.source_tag !== filter.container) return false;
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

    if (log.island && log.island !== 'lead' && log.island !== 'default') {
      discoveredIslands.add(log.island);
    }

    if (log.candidate) {
      setLatestCandidate(log.candidate);
    }

    // Bulk trim: only fire when significantly over capacity
    if (arr.length > TRIGGER) {
      const remove = arr.length - KEEP;
      arr.splice(0, remove);
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
      for (let i = arr.length - 1; i >= 0; i--) {
        if (matches(arr[i], filter)) out.push(arr[i]);
        if (filter.limit && out.length >= filter.limit) return out;
      }
    }
    return out;
  }

  /**
   * Cursor-based history page: returns up to `limit` logs whose ts < `before`.
   */
  function queryBefore(before, filter = {}) {
    const limit = filter.limit || 100;
    const sources = filter.source ? [filter.source] : Array.from(buckets.keys());
    const out = [];
    for (const s of sources) {
      const arr = bucket(s);
      for (let i = arr.length - 1; i >= 0; i--) {
        const log = arr[i];
        if (before !== undefined && log.ts !== undefined && log.ts >= before) continue;
        if (matches(log, filter)) out.push(log);
        if (out.length >= limit) break;
      }
    }
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
    return {
      total,
      bySource,
      byLevel,
      newestTs: newest,
      capPerSource,
      islands: Array.from(discoveredIslands).sort(),
      candidates: Object.fromEntries(candidatesByIsland),
      bestCandidates: Object.fromEntries(bestCandidatesByIsland),
      bestOverall: bestCandidate,
    };
  }

  function sources() {
    return Array.from(buckets.keys());
  }

  function parseFit(cand) {
    if (!cand) return -Infinity;
    if (cand.fitness !== null && cand.fitness !== undefined && !Number.isNaN(Number(cand.fitness))) {
      return Number(cand.fitness);
    }
    return -Infinity;
  }

  function setLatestCandidate(cand) {
    if (!cand) return;
    const island = cand.island || 'island-1';
    candidatesByIsland.set(island, cand);
    latestCandidate = cand;

    const candFit = parseFit(cand);

    // Track best per island
    const currentBestIsland = bestCandidatesByIsland.get(island);
    if (!currentBestIsland || candFit >= parseFit(currentBestIsland)) {
      bestCandidatesByIsland.set(island, cand);
    }

    // Track best overall across all islands
    if (!bestCandidate || candFit >= parseFit(bestCandidate)) {
      bestCandidate = cand;
    }

    if (island !== 'lead' && island !== 'default') {
      discoveredIslands.add(island);
    }

    for (const fn of candidateSubscribers) {
      try { fn(cand); } catch { /* ignore subscriber errors */ }
    }
  }

  function getLatestCandidate(island) {
    if (island) return candidatesByIsland.get(island) || null;
    return latestCandidate;
  }

  function getBestCandidate(island) {
    if (island) return bestCandidatesByIsland.get(island) || null;
    return bestCandidate || latestCandidate;
  }

  function getAllCandidates() {
    return Object.fromEntries(candidatesByIsland);
  }

  function getAllBestCandidates() {
    return Object.fromEntries(bestCandidatesByIsland);
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
    getBestCandidate,
    getAllCandidates,
    getAllBestCandidates,
    onCandidate,
  };
}

module.exports = { createStore };
