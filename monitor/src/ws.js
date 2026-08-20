const { WebSocketServer } = require('ws');

/**
 * Attaches WebSocket handling to the HTTP server.
 *
 * Improvements over v1:
 *   - Pushes `{ type: 'stats', … }` to ALL clients whenever new logs land
 *     (throttled: at most once per second) so the browser doesn't need HTTP polling.
 *   - Handles `{ type: 'history', before, source, limit }` from the client and
 *     replies with `{ type: 'history_chunk', logs: […], hasMore }`.
 *   - Per-client filter is unchanged from v1.
 */
function attachWebSocket(server, store) {
  const wss = new WebSocketServer({ server, path: '/stream' });
  const clients = new Set();

  // ── Throttled stats broadcast ─────────────────────────────────────────────
  // Subscribe to every push so we know when to broadcast stats.
  // We coalesce rapid bursts: stats are pushed at most once per second.
  let statsDirty = false;
  let statsTimer = null;

  store.subscribe(() => {
    statsDirty = true;
    if (statsTimer === null) {
      statsTimer = setInterval(() => {
        if (!statsDirty) return;
        statsDirty = false;
        if (clients.size === 0) return;
        const payload = JSON.stringify({ type: 'stats', ...store.stats() });
        for (const ws of clients) {
          if (ws.readyState === ws.OPEN) ws.send(payload);
        }
      }, 1000);
    }
  });

  // ── Filter helpers ────────────────────────────────────────────────────────

  function passesFilter(log, filter) {
    if (!filter) return true;
    if (filter.source && log.source !== filter.source) return false;
    if (filter.container && log.source_tag !== filter.container) return false;
    if (filter.level) {
      const want = String(filter.level).toUpperCase();
      const got = String(log.level || '').toUpperCase();
      if (want === 'ERROR') {
        if (!['ERROR','CRIT','ALERT','EMERG'].includes(got)) return false;
      } else if (got !== want) {
        return false;
      }
    }
    if (filter.search) {
      const hay = JSON.stringify(log).toLowerCase();
      if (!hay.includes(String(filter.search).toLowerCase())) return false;
    }
    if (filter.target && log.target !== filter.target) return false;
    if (filter.workerId && log.worker_id !== filter.workerId) return false;
    return true;
  }

  // ── Connection handler ────────────────────────────────────────────────────

  wss.on('connection', (ws) => {
    ws.filter = null;
    clients.add(ws);

    // Send initial hello + current stats
    ws.send(JSON.stringify({ type: 'hello', ts: Date.now() }));
    ws.send(JSON.stringify({ type: 'stats', ...store.stats() }));

    ws.on('message', (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        ws.send(JSON.stringify({ type: 'error', error: 'invalid json' }));
        return;
      }

      if (msg.type === 'filter') {
        ws.filter = msg.filter || null;
        ws.send(JSON.stringify({ type: 'ack', filter: ws.filter }));

      } else if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));

      } else if (msg.type === 'history') {
        // Cursor-based history fetch from the client.
        // msg.before  — ms timestamp (exclusive upper bound, i.e. "older than this")
        // msg.source  — optional source filter
        // msg.limit   — max rows to return (capped at 200)
        const limit = Math.min(parseInt(msg.limit || '100', 10) || 100, 200);
        const filter = {
          source: msg.source || undefined,
          level:  msg.level  || undefined,
          search: msg.search || undefined,
          limit,
        };
        const logs = store.queryBefore(msg.before, filter);
        ws.send(JSON.stringify({
          type: 'history_chunk',
          before: msg.before,
          logs,
          hasMore: logs.length >= limit,
        }));
      }
    });

    ws.on('close', () => clients.delete(ws));
    ws.on('error', () => clients.delete(ws));
  });

  // ── Broadcast live logs ───────────────────────────────────────────────────

  function broadcast(log) {
    if (clients.size === 0) return;
    const payload = JSON.stringify(log);
    for (const ws of clients) {
      if (ws.readyState !== ws.OPEN) continue;
      if (passesFilter(log, ws.filter)) {
        ws.send(payload);
      }
    }
  }

  return broadcast;
}

module.exports = { attachWebSocket };
