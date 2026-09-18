const assert = require('assert');
const { normalizeLog, extractCandidate } = require('../src/consumer');

console.log('Running consumer log normalization tests...');

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
assert.ok(norm8.message.startsWith("Gen 9: best=713.5261"));
assert.ok(norm8.candidate);
assert.strictEqual(norm8.candidate.gen, 9);
assert.strictEqual(norm8.candidate.fitness, 713.5261);
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

console.log('\nALL CONSUMER TESTS PASSED SUCCESSFULLY!');
