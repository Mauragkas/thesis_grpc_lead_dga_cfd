const { decodeGenes, buildPlaneMesh, meshToSTL } = require('./planeGeometry');

/**
 * Registers REST API endpoints on the Express app.
 */
function createApi(app, store) {
  // Healthcheck for Docker compose / orchestrator
  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', ts: Date.now() });
  });

  // Current stats / aggregated metrics
  app.get('/stats', (_req, res) => {
    res.json(store.stats());
  });

  app.get('/sources', (_req, res) => {
    res.json(store.sources());
  });

  app.get('/islands', (_req, res) => {
    res.json({ islands: store.stats().islands });
  });

  app.get('/api/candidates', (_req, res) => {
    res.json(store.getAllCandidates());
  });

  app.get('/api/candidates/best', (_req, res) => {
    res.json(store.getAllBestCandidates());
  });

  app.get('/api/candidate/best', (req, res) => {
    const island = req.query.island;
    const candidate = store.getBestCandidate(island);
    if (!candidate) {
      return res.status(404).json({ error: island ? `No candidate received yet for ${island}` : 'No candidate received yet' });
    }
    res.json(candidate);
  });

  app.get('/api/candidate/latest', (req, res) => {
    const island = req.query.island;
    const candidate = store.getLatestCandidate(island);
    if (!candidate) {
      return res.status(404).json({ error: island ? `No candidate received yet for ${island}` : 'No candidate received yet' });
    }
    res.json(candidate);
  });

  app.get('/api/candidate/stl', (req, res) => {
    const island = req.query.island;
    const mode = req.query.mode || 'best'; // default to best!
    const candidate = mode === 'latest' ? store.getLatestCandidate(island) : store.getBestCandidate(island);
    let params;
    let filename = island ? `aircraft_${island}.stl` : 'aircraft_best_overall.stl';
    if (candidate && candidate.decoded) {
      params = candidate.decoded;
      filename = `candidate_${mode}_${candidate.island || 'overall'}_gen_${candidate.gen || 0}.stl`;
    } else {
      params = decodeGenes([]);
    }
    const mesh = buildPlaneMesh(params);
    const stl = meshToSTL(mesh, `aircraft_${candidate?.island || 'main'}_gen_${candidate?.gen || 0}`);
    res.setHeader('Content-Type', 'model/stl');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.send(stl);
  });

  app.post('/api/candidate', (req, res) => {
    const { genome, gen, fitness, island } = req.body;
    if (!Array.isArray(genome)) {
      return res.status(400).json({ error: 'genome array required' });
    }
    const candidate = {
      island: island || 'island-1',
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
   *   source    - exact source topic (e.g. logs.worker)
   *   island    - island identifier (e.g. island-1, island-2)
   *   role      - orchestrator | worker | surrogate | lead
   *   container - specific container / source_tag
   *   level     - INFO | WARN | ERROR (ERROR includes CRIT/ALERT/EMERG)
   *   since     - ms epoch (inclusive)
   *   until     - ms epoch (inclusive)
   *   search    - case-insensitive substring over the whole JSON
   *   target    - tracing target (e.g. lead_node::lead::routing)
   *   workerId  - worker_id field
   *   limit     - max results (default 200, max 1000)
   *   order     - "desc" (default, newest first) | "asc"
   */
  app.get('/logs', (req, res) => {
    const q = req.query;
    const limit = Math.min(parseInt(q.limit || '200', 10) || 200, 1000);
    const filter = {
      source:    q.source,
      island:    q.island,
      role:      q.role,
      container: q.container,
      level:     q.level,
      since:     q.since  ? parseInt(q.since,  10) : undefined,
      until:     q.until  ? parseInt(q.until,  10) : undefined,
      search:    q.search,
      target:    q.target,
      workerId:  q.workerId,
      limit,
    };
    let logs = store.query(filter);
    if (q.order === 'asc') logs = logs.reverse();
    res.json({ count: logs.length, limit, filter, logs });
  });

  /**
   * GET /logs/tail?source=&limit=&island=&role=
   * Convenience: newest N logs from a source or island.
   */
  app.get('/logs/tail', (req, res) => {
    const limit = Math.min(parseInt(req.query.limit || '50', 10) || 50, 500);
    const logs = store.query({
      source:    req.query.source,
      island:    req.query.island,
      role:      req.query.role,
      container: req.query.container,
      limit,
    });
    res.json({ count: logs.length, logs });
  });

  /**
   * GET /logs/history?before=&limit=&source=&island=&role=
   * Cursor-based history: returns up to `limit` logs whose ts < `before`.
   * Results are sorted newest-first.
   */
  app.get('/logs/history', (req, res) => {
    const before = req.query.before !== undefined ? parseInt(req.query.before, 10) : undefined;
    const limit  = Math.min(parseInt(req.query.limit || '100', 10) || 100, 200);
    const filter = {
      source:    req.query.source    || undefined,
      island:    req.query.island    || undefined,
      role:      req.query.role      || undefined,
      container: req.query.container || undefined,
      level:     req.query.level     || undefined,
      search:    req.query.search    || undefined,
      limit,
    };
    const logs = store.queryBefore(before, filter);
    res.json({
      before,
      limit,
      count: logs.length,
      hasMore: logs.length >= limit,
      logs,
    });
  });

  app.use((err, _req, res, _next) => {
    console.error('api error:', err);
    res.status(500).json({ error: err.message });
  });
}

module.exports = { createApi, buildApi: createApi };
