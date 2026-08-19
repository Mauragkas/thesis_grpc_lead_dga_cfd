# Use Case: Federated Learning and Model Synchronization

## Description

Coordinates distributed model retraining and synchronization across the LEAD cluster. Detects global data distribution drift, uses neighbor heartbeat consensus to elect a Transient Coordinator, aggregates models using Federated Averaging (`FedAvg`), broadcasts the updated global model version, and migrates keys to their newly responsible nodes.

## Primary actor

- `LeadNode`

## Supporting systems

- `LearnedIndex` (drift detection, pending/active models)
- `RemoteNode` (`ModelClient` gRPC interface: `Heartbeat`, `RequestModel`, `PushModel`)
- `FederatedAveraging` (`fed_avg`)
- `KeyStore` (for snapshotting and key migration)

## Main steps: Drift Detection and Local Retraining

1. As keys are inserted, `LearnedIndex::record_insert()` checks if new keys exceed `DRIFT_THRESHOLD` ($40\%$) with at least `MIN_KEYS_FOR_DRIFT` ($50$) keys.
2. If drift threshold is crossed, the index sets `update_ready = true`.
3. `maybe_retrain()` (every 5s) detects `update_ready`, captures a snapshot of local keys, and trains a new local candidate `RmiModel` (`train_auto`).
4. The candidate model is stored as a pending model update.

## Main steps: Heartbeat Consensus and Coordinator Election

1. `heartbeat_round()` (every 8s) queries all immediate predecessor and successor neighbors via `heartbeat(ready, version)`.
2. Each neighbor responds with its readiness status.
3. If $\ge 90\%$ of responding neighbors report `update_ready` and this node has a pending model:
   - This node assumes the role of **Transient Coordinator**.

## Main steps: Federated Averaging and Broadcast

1. The coordinator pulls models from all neighbors via `request_model()`.
2. `fed_avg(models, new_version)` combines the collected models:
   - Weights leaf parameters by sample size: $w_i = n_i / \sum n$.
   - Computes weighted linear weights, biases, and anchor offsets across all bins.
3. The coordinator broadcasts the new model to all known cluster peers via `push_model(version, payload)`.
4. The coordinator activates the model locally.

## Main steps: Key Re-Migration

1. Following model activation (either as coordinator or upon receiving `push_model`), each node invokes `migrate_keys_for_new_model()`.
2. For each key in local storage:
   - Check if the key is still owned by any local vnode under the new model.
   - If ownership moved to a remote node: send `put_local` to the new owner and delete from local storage upon successful transfer.
3. Reset drift counters with `reset_drift_state()`.

## Postconditions

- All nodes converge to an updated, globally consistent LearnedHASH model.
- Keys are migrated so that every key resides on the node responsible for its new hash position.
