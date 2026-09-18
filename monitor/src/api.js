const { decodeGenes, buildPlaneMesh, meshToSTL } = require('./planeGeometry');

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

  app.get('/api/candidate/latest', (_req, res) => {
    const candidate = store.getLatestCandidate();
    if (!candidate) {
      return res.status(404).json({ error: 'No candidate received yet' });
    }
    res.json(candidate);
  });

  app.get('/api/candidate/stl', (_req, res) => {
    const candidate = store.getLatestCandidate();
    let params;
    let filename = 'aircraft.stl';
    if (candidate && candidate.decoded) {
      params = candidate.decoded;
      filename = `candidate_gen_${candidate.gen || 0}.stl`;
    } else {
      params = decodeGenes([]);
    }
    const mesh = buildPlaneMesh(params);
    const stl = meshToSTL(mesh, `aircraft_gen_${candidate?.gen || 0}`);
    res.setHeader('Content-Type', 'model/stl');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.send(stl);
  });

  app.post('/api/candidate', (req, res) => {
    const { genome, gen, fitness } = req.body;
    if (!Array.isArray(genome)) {
      return res.status(400).json({ error: 'genome array required' });
    }
    const candidate = {
      gen: gen || 0,
      fitness: fitness !== undefined ? Number(fitness) : null,
      genome,
      decoded: decodeGenes(genome),
      ts: Date.now(),
    };
    store.setLatestCandidate(candidate);
    res.json({ ok: true, candidate });
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
