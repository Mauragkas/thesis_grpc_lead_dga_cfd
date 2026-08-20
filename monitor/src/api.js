function buildApi(app, store) {
  app.get('/health', (_req, res) => {
    res.json({ ok: true, ts: Date.now() });
  });

  app.get('/stats', (_req, res) => {
    res.json(store.stats());
  });

  app.get('/sources', (_req, res) => {
    res.json(store.sources());
  });

  /**
   * GET /logs
   * Query params:
   *   source   - exact source topic (e.g. logs.worker)
   *   level    - INFO | WARN | ERROR (ERROR includes CRIT/ALERT/EMERG)
   *   since    - ms epoch (inclusive)
   *   until    - ms epoch (inclusive)
   *   search   - case-insensitive substring over the whole JSON
   *   target   - tracing target (e.g. lead_node::lead::routing)
   *   workerId - worker_id field
   *   limit    - max results (default 200, max 1000)
   *   order    - "desc" (default, newest first) | "asc"
   */
  app.get('/logs', (req, res) => {
    const q = req.query;
    const limit = Math.min(parseInt(q.limit || '200', 10) || 200, 1000);
    const filter = {
      source:   q.source,
      level:    q.level,
      since:    q.since  ? parseInt(q.since,  10) : undefined,
      until:    q.until  ? parseInt(q.until,  10) : undefined,
      search:   q.search,
      target:   q.target,
      workerId: q.workerId,
      limit,
    };
    let logs = store.query(filter);
    if (q.order === 'asc') logs = logs.reverse();
    res.json({ count: logs.length, limit, filter, logs });
  });

  /**
   * GET /logs/tail?source=&limit=
   * Convenience: newest N logs from a source (or all sources).
   */
  app.get('/logs/tail', (req, res) => {
    const limit = Math.min(parseInt(req.query.limit || '50', 10) || 50, 500);
    const logs = store.query({ source: req.query.source, limit });
    res.json({ count: logs.length, logs });
  });

  /**
   * GET /logs/history?source=&before=<ms>&limit=&level=&search=
   *
   * Cursor-based pagination for the "load older" feature.
   * Returns up to `limit` logs with ts < `before`, newest-first.
   * The client uses the smallest ts in the response as the next `before`.
   *
   * Query params:
   *   before   - ms epoch (exclusive); logs older than this are returned
   *   source   - optional topic filter
   *   level    - optional level filter
   *   search   - optional substring filter
   *   limit    - max results (default 100, max 500)
   */
  app.get('/logs/history', (req, res) => {
    const q = req.query;
    const before = q.before ? parseInt(q.before, 10) : undefined;
    const limit  = Math.min(parseInt(q.limit || '100', 10) || 100, 500);
    const filter = {
      source: q.source  || undefined,
      level:  q.level   || undefined,
      search: q.search  || undefined,
      limit,
    };
    const logs = store.queryBefore(before, filter);
    res.json({
      count:   logs.length,
      before,
      hasMore: logs.length >= limit,
      logs,
    });
  });

  app.use((err, _req, res, _next) => {
    console.error('api error:', err);
    res.status(500).json({ error: err.message });
  });
}

module.exports = { buildApi };
