# Use Case: Load Configuration from Environment

## Description

Reads environment variables to configure network interfaces, cluster bootstrap endpoints, public advertised addresses, virtual node counts, FRM learning parameters, PID anchor tuning, pruning thresholds, range query overscan multiplier, and background maintenance intervals.

## Primary actor

- Deployment environment / Node startup

## Environment variables

### Storage Subsystem
- `LEAD_STORAGE_BACKEND` / `STORAGE_BACKEND`: Storage engine backend selection (`"memory"` or `"sled"`, default: `"memory"`).
- `LEAD_STORAGE_PATH` / `STORAGE_PATH` / `LEAD_DATA_DIR`: Disk directory path for sled database storage (default: `"./data/lead"` when sled backend is selected).

### Network & Topology
- `HTTP_BIND`: Local bind socket address for the axum HTTP REST API (default: `"0.0.0.0:8080"`).
- `GRPC_BIND`: Local bind socket address for the tonic gRPC server (default: `"0.0.0.0:50051"`).
- `SELF_URI`: Advertised gRPC endpoint URI used by peer nodes to reach this node (default: `"http://127.0.0.1:50051"`).
- `JOIN_URI`: Optional gRPC endpoint URI of an existing cluster member to join.
- `VIRTUAL_NODE_COUNT` / `LEAD_VIRTUAL_NODE_COUNT`: Number of virtual nodes ($k$) hosted by this physical instance (default: `10`).
- `LEAD_SUCCESSOR_LIST_LEN` / `SUCCESSOR_LIST_LEN`: Length of successor list $R$ (default: `4`).
- `LEAD_JOIN_RETRY_COUNT` / `JOIN_RETRY_COUNT`: Maximum bootstrap join attempts before timeout (default: `300`).
- `LEAD_JOIN_RETRY_DELAY_SECS` / `JOIN_RETRY_DELAY_SECS`: Delay between bootstrap join attempts in seconds (default: `1`).

### Range Queries
- `LEAD_RANGE_OVERSCAN_MULTIPLIER` / `RANGE_OVERSCAN_MULTIPLIER`: Local range scan overscan multiplier (default: `3`).

### Federated Learning (FRM)
- `LEAD_FRM_QUORUM_THRESHOLD` / `FRM_QUORUM_THRESHOLD`: Neighbor quorum ratio needed to trigger transient coordinator round (default: `0.90`).
- `LEAD_DRIFT_THRESHOLD` / `DRIFT_THRESHOLD`: Drift threshold triggering model retraining (default: `0.40`).
- `LEAD_MIN_KEYS_FOR_DRIFT` / `MIN_KEYS_FOR_DRIFT`: Minimum total keys before drift detection activates (default: `50`).
- `LEAD_FRM_GRACE_PERIOD_SECS` / `FRM_GRACE_PERIOD_SECS`: Startup grace period suppressing early drift updates (default: `10`).

### PID Controller & Pruning
- `LEAD_PID_ADJUST_INTERVAL` / `PID_ADJUST_INTERVAL`: Key insertion interval for PID anchor adjustments (default: `100`).
- `LEAD_PID_TARGET_RATIO` / `PID_TARGET_RATIO`: Target fraction of keys inside VID window (default: `0.95`).
- `LEAD_PID_SCALE_STEP` / `PID_SCALE_STEP`: Proportional scaling increment for anchor adjustment (default: `0.05`).
- `LEAD_PID_CENTERING_STEP` / `PID_CENTERING_STEP`: Centering offset increment (default: `0.01`).
- `LEAD_PID_UPPER_THRESHOLD` / `PID_UPPER_THRESHOLD`: PID error tolerance bound (default: `0.05`).
- `LEAD_PID_MID_THRESHOLD` / `PID_MID_THRESHOLD`: PID deadband bound (default: `0.02`).
- `LEAD_PID_MIN_SAMPLES` / `PID_MIN_SAMPLES`: Minimum sample count before adjusting anchor (default: `20`).
- `LEAD_PRUNE_ERROR_RATE` / `PRUNE_ERROR_RATE`: Error rate threshold above which vnodes are pruned (default: `0.30`).
- `LEAD_PRUNE_INACTIVE_SECS` / `PRUNE_INACTIVE_SECS`: Inactivity period after which vnodes are pruned (default: `120`).

### Maintenance Intervals
- `LEAD_STABILIZE_INTERVAL_SECS` / `STABILIZE_INTERVAL_SECS`: Periodic ring stabilization interval (default: `2`).
- `LEAD_FIX_FINGERS_INTERVAL_SECS` / `FIX_FINGERS_INTERVAL_SECS`: Periodic finger table maintenance interval (default: `5`).
- `LEAD_CHECK_PREDECESSOR_INTERVAL_SECS` / `CHECK_PREDECESSOR_INTERVAL_SECS`: Periodic predecessor ping check interval (default: `10`).
- `LEAD_HEARTBEAT_INTERVAL_SECS` / `HEARTBEAT_INTERVAL_SECS`: FRM peer heartbeat interval (default: `8`).
- `LEAD_RETRAIN_INTERVAL_SECS` / `RETRAIN_INTERVAL_SECS`: FRM local model retrain check interval (default: `5`).

## Main steps

1. `Config::from_env()` is called during startup.
2. Reads network bindings, topology tunables, FRM parameters, PID constants, and loop intervals from environment variables or applies defaults.
3. Return initialized `Config` struct.

## Postconditions

- A valid `Config` object is provided to `LeadNode` and server builders.

