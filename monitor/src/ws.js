const { WebSocketServer } = require('ws');

/**
 * Attaches a WebSocket server at /stream.
 * Clients can send a JSON filter message to narrow what they receive;
 * otherwise they get everything.
 *
 * Returns a broadcast(log) function used by the consumer.
 */
function attachWebSocket(server, _store) {
  const wss = new WebSocketServer({ server, path: '/stream' });

  const clients = new Set();

  wss.on('connection', (ws, req) => {
    ws.filter = null;
    clients.add(ws);

    ws.send(JSON.stringify({ type: 'hello', ts: Date.now() }));

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === 'filter') {
          ws.filter = msg.filter || null;
          ws.send(JSON.stringify({ type: 'ack', filter: ws.filter }));
        } else if (msg.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));
        }
      } catch {
        ws.send(JSON.stringify({ type: 'error', error: 'invalid json' }));
      }
    });

    ws.on('close', () => clients.delete(ws));
    ws.on('error', () => clients.delete(ws));
  });

  function passesFilter(log, filter) {
    if (!filter) return true;
    if (filter.source && log.source !== filter.source) return false;
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

  function broadcast(log) {
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
