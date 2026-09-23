const assert = require('assert');
const { normalizeLog, extractCandidate } = require('../src/consumer');
const { createStore } = require('../src/store');

console.log('Running consumer log normalization & multi-island isolation tests...');

// Test 1: Fluentd wrapped unformatted log (User Screenshot Row 6)
const row6 = {
  "@timestamp": 1789738509,
  container_id: "41bedb6582954f268061c24d2e386ab5cb42df9895b83526f15a52416efa598e",
  container_name: "/compose-orchestrator-1",
  source: "logs.orchestrator",
  log: "Gen 9/10 | Best: 713.5261 | Avg: -33332977.3851 | BestEver: 713.5261",
  partition: 0,
  offset: "120",
  source_tag: "compose-orchestrator-1",
  ts: 1789738509984,
  timestamp: "2026-09-18T13:35:09.984Z"
};
const norm6 = normalizeLog(row6);
assert.strictEqual(norm6.message, "Gen 9/10 | Best: 713.5261 | Avg: -33332977.3851 | BestEver: 713.5261");
assert.strictEqual(norm6.level, "INFO");
assert.strictEqual(norm6.target, "compose-orchestrator-1");
assert.strictEqual(norm6.source_tag, "compose-orchestrator-1");
assert.strictEqual(norm6.island, "island-1");
assert.strictEqual(norm6.role, "orchestrator");
console.log('✓ Test 1 passed (Fluentd wrapped unformatted log)');

// Test 2: Fluentd wrapped candidate genome log (User Screenshot Row 7)
const row7 = {
  "@timestamp": 1789738509,
  container_id: "41bedb6582954f268061c24d2e386ab5cb42df9895b83526f15a52416efa598e",
  container_name: "/compose-orchestrator-1",
  source: "logs.orchestrator",
  log: "Gen 9 best candidate genome: [0.9837730258373958, 0.5129863559419924, 1.0, 0.5865154158317756, 0.7700012667996669, 0.46808275247816444, 0.5741249224064591, 0.9210407177839315, 0.6532650776224286, 0.4141034520835827]",
  partition: 0,
  offset: "121",
  source_tag: "compose-orchestrator-1",
  ts: 1789738509984,
  timestamp: "2026-09-18T13:35:09.984Z"
};
const norm7 = normalizeLog(row7);
assert.ok(norm7.message.startsWith("Gen 9 best candidate genome: [0.9837730258373958"));
assert.strictEqual(norm7.level, "INFO");
assert.strictEqual(norm7.target, "compose-orchestrator-1");
assert.ok(norm7.candidate, "Candidate should be extracted");
assert.strictEqual(norm7.candidate.gen, 9);
assert.strictEqual(norm7.candidate.genome.length, 10);
assert.strictEqual(norm7.candidate.island, "island-1");
console.log('✓ Test 2 passed (Fluentd wrapped candidate extraction)');

// Test 3: Standard tracing subscriber JSON event (User Screenshot Row 8)
const row8 = {
  timestamp: "2026-09-18T13:35:09.526Z",
  level: "INFO",
  fields: {
    message: "Gen 9: best=713.5261, avg=-33332977.3851, best_ever=713.5261, best_genome=[0.9837730258373958, 0.5129863559419924, 1.0, 0.5865154158317756, 0.7700012667996669, 0.46808275247816444, 0.5741249224064591, 0.9210407177839315, 0.6532650776224286, 0.4141034520835827]",
    best: 713.5261,
    avg: -33332977.3851,
    best_ever: 713.5261,
    best_genome: [0.9837730258373958, 0.5129863559419924, 1.0, 0.5865154158317756, 0.7700012667996669, 0.46808275247816444, 0.5741249224064591, 0.9210407177839315, 0.6532650776224286, 0.4141034520835827],
    gen: 9
  },
  target: "orchestrator::ga::algorithm"
};
const norm8 = normalizeLog(row8, { topic: "logs.orchestrator" });
assert.strictEqual(norm8.level, "INFO");
assert.strictEqual(norm8.target, "orchestrator::ga::algorithm");
assert.strictEqual(norm8.source, "logs.orchestrator");
assert.strictEqual(norm8.island, "island-1");
assert.strictEqual(norm8.role, "orchestrator");
assert.ok(norm8.message.startsWith("Gen 9: best=713.5261"));
assert.ok(norm8.candidate);
assert.strictEqual(norm8.candidate.gen, 9);
assert.strictEqual(norm8.candidate.fitness, 713.5261);
assert.strictEqual(norm8.candidate.island, "island-1");
console.log('✓ Test 3 passed (Structured tracing subscriber log)');

// Test 4: Docker stdout containing nested JSON string
const nestedJson = {
  container_name: "/lead_node2",
  log: JSON.stringify({
    timestamp: "2026-09-18T13:35:10.000Z",
    level: "WARN",
    fields: { message: "Finger table update delayed" },
    target: "lead_node::routing"
  }) + "\n"
};
const normNested = normalizeLog(nestedJson, { topic: "logs.lead" });
assert.strictEqual(normNested.level, "WARN");
assert.strictEqual(normNested.message, "Finger table update delayed");
assert.strictEqual(normNested.target, "lead_node::routing");
assert.strictEqual(normNested.source_tag, "lead_node2");
assert.strictEqual(normNested.island, "lead");
assert.strictEqual(normNested.role, "lead");
console.log('✓ Test 4 passed (Nested JSON string in Docker log)');

// Test 5: stderr raw log without level field
const stderrLog = {
  stream: "stderr",
  log: "Warning: Connection timed out, retrying..."
};
const normStderr = normalizeLog(stderrLog);
assert.strictEqual(normStderr.level, "WARN");
assert.strictEqual(normStderr.message, "Warning: Connection timed out, retrying...");
console.log('✓ Test 5 passed (stderr fallback level)');

// Test 6: Multi-orchestrator and multi-worker isolation (Island 2)
const orc2Log = {
  file_path: "/var/log/app/orchestrator2.log",
  source: "logs.orchestrator",
  log: JSON.stringify({
    timestamp: "2026-09-18T13:35:12.000Z",
    level: "INFO",
    fields: {
      message: "Gen 5: best=680.12, avg=-1000.0, best_ever=680.12, best_genome=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]",
      best: 680.12,
      gen: 5,
      best_genome: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    },
    target: "orchestrator::ga::algorithm"
  })
};
const normOrc2 = normalizeLog(orc2Log);
assert.strictEqual(normOrc2.source_tag, "orchestrator2");
assert.strictEqual(normOrc2.island, "island-2");
assert.strictEqual(normOrc2.role, "orchestrator");
assert.ok(normOrc2.candidate);
assert.strictEqual(normOrc2.candidate.island, "island-2");
assert.strictEqual(normOrc2.candidate.gen, 5);
assert.strictEqual(normOrc2.candidate.fitness, 680.12);
console.log('✓ Test 6 passed (Orchestrator 2 Island differentiation)');

// Test 7: Worker Pool 2 Python JSON log with explicit metadata
const worker2Log = {
  file_path: "/var/log/app/worker2_f38a901.log",
  source: "logs.worker",
  log: JSON.stringify({
    timestamp: "2026-09-18T13:35:13.000Z",
    level: "INFO",
    target: "worker.fitness",
    message: "evaluation finished elapsed=0.045",
    worker_id: "worker2",
    island_id: "island-2",
    role: "worker",
    worker_pool: "pool-2"
  })
};
const normWorker2 = normalizeLog(worker2Log);
assert.strictEqual(normWorker2.source_tag, "worker2_f38a901");
assert.strictEqual(normWorker2.island, "island-2");
assert.strictEqual(normWorker2.role, "worker");
assert.strictEqual(normWorker2.worker_pool, "pool-2");
console.log('✓ Test 7 passed (Worker Pool 2 metadata parsing)');

// Test 8: Arbitrary Nth Island dynamic inference (e.g. Island 5, Pool 5)
const orcNLog = {
  container_name: "/compose-orchestrator5-1",
  source: "logs.orchestrator",
  log: "Gen 1: best=100.0, best_genome=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]"
};
const normOrcN = normalizeLog(orcNLog);
assert.strictEqual(normOrcN.island, "island-5");
assert.strictEqual(normOrcN.role, "orchestrator");
assert.ok(normOrcN.candidate);
assert.strictEqual(normOrcN.candidate.island, "island-5");
console.log('✓ Test 8 passed (Dynamic N-th Island inference)');

// Test 9: In-memory store candidate isolation across islands
const store = createStore();
store.push(norm8);    // Island 1 candidate (Gen 9, Fit 713.5261)
store.push(normOrc2); // Island 2 candidate (Gen 5, Fit 680.12)
store.push(normWorker2);

// Check per-island candidate retrieval
const cand1 = store.getLatestCandidate("island-1");
const cand2 = store.getLatestCandidate("island-2");
assert.ok(cand1, "Island 1 candidate must exist");
assert.ok(cand2, "Island 2 candidate must exist");
assert.strictEqual(cand1.island, "island-1");
assert.strictEqual(cand1.gen, 9);
assert.strictEqual(cand2.island, "island-2");
assert.strictEqual(cand2.gen, 5);

const allCandidates = store.getAllCandidates();
assert.ok(allCandidates["island-1"]);
assert.ok(allCandidates["island-2"]);

// Check filtering by island in store.query
const isl1Logs = store.query({ island: "island-1" });
const isl2Logs = store.query({ island: "island-2" });
assert.strictEqual(isl1Logs.length, 1);
assert.strictEqual(isl2Logs.length, 2); // orc2 + worker2

// Check filtering by role
const workerLogs = store.query({ role: "worker" });
assert.strictEqual(workerLogs.length, 1);
assert.strictEqual(workerLogs[0].role, "worker");

// Check discovered islands list in stats
const stats = store.stats();
assert.deepStrictEqual(stats.islands, ["island-1", "island-2"]);
console.log('✓ Test 9 passed (Store candidate isolation & island filtering)');

// Test 10: Best candidate overall tracking vs latest candidate
// normOrc2 (Island 2, Fit 680.12) arrived AFTER norm8 (Island 1, Fit 713.5261).
// Latest candidate is Island 2, BUT Best Overall must still be Island 1 (Fit 713.5261).
const latestOverall = store.getLatestCandidate();
const bestOverall = store.getBestCandidate();
assert.strictEqual(latestOverall.island, "island-2", "Latest should be Island 2");
assert.strictEqual(bestOverall.island, "island-1", "Best overall must remain Island 1 due to higher fitness");
assert.strictEqual(bestOverall.fitness, 713.5261);

// Now introduce a new breakthrough candidate from Island 2 with higher fitness (820.5)
const breakthroughLog = {
  container_name: "/compose-orchestrator2-1",
  source: "logs.orchestrator",
  log: "Gen 15: best=820.5, best_genome=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]"
};
const normBreakthrough = normalizeLog(breakthroughLog);
store.push(normBreakthrough);

const newBestOverall = store.getBestCandidate();
assert.strictEqual(newBestOverall.island, "island-2");
assert.strictEqual(newBestOverall.gen, 15);
assert.strictEqual(newBestOverall.fitness, 820.5, "Best overall must update when higher fitness is found");

// Check per-island bests
assert.strictEqual(store.getBestCandidate("island-1").fitness, 713.5261);
assert.strictEqual(store.getBestCandidate("island-2").fitness, 820.5);
console.log('✓ Test 10 passed (Best candidate overall tracking across islands)');

console.log('\nALL CONSUMER & MULTI-ISLAND ISOLATION TESTS PASSED SUCCESSFULLY!');
