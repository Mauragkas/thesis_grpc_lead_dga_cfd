# Use Case: Manage Ring Topology and Chord Stabilization

## Description

Maintains the Chord-like ring topology among distributed orchestrator instances. Orchestrator nodes join the ring, periodically stabilize successor/predecessor pointers, notify successors of their presence, and maintain a successor list for fault tolerance.

## Primary actor

- `Stabilizer` / `LocalRingMember` / `RingServer`

## Supporting systems

- `RingState` (local guarded ring state)
- `GrpcRingClient` (tonic client talking to peer orchestrators)
- `RingServer` (gRPC service implementing `/orchestrator_ring.Ring`)

## Preconditions

- `RingConfig` is loaded with `bind_address` and `self_address`.
- The local Ring gRPC server is listening.

## Main steps: Bootstrap Join

1. An orchestrator starts with a configured `bootstrap_address`.
2. The stabilizer issues `find_successor(bootstrap_addr, self_node.id)` via `RingClient`.
3. The bootstrap node returns the node responsible for `self_node.id`.
4. The joining node sets its successor to the returned node.
5. If joining fails (e.g. bootstrap node starting up), it retries with a backoff delay.

## Main steps: Periodic Stabilization Loop

1. Every `stabilize_interval`, the stabilizer executes:
   1. Ask current successor for its predecessor `x` via `get_predecessor(succ.address)`.
   2. If `x` exists and lies in the open interval `(self.id, succ.id)`, update successor to `x`.
   3. Call `notify(succ.address, self_node)` to inform the successor that `self` is its predecessor.
2. In the same tick, refresh the successor list:
   1. Fetch `get_successor_list(succ.address)`.
   2. Prepend `succ` and keep up to `successor_list_size` entries.
   3. Update local `successor_list`.

## Main steps: Handling Inbound Ring RPCs

1. **`FindSuccessor(id)`**: If `id` is in `(self.id, succ.id]`, return `succ`; otherwise forward to next hop.
2. **`GetPredecessor()`**: Return current `predecessor`.
3. **`Notify(other)`**: Accept `other` as new predecessor if current predecessor is `None` or `other.id` is in `(current_pred.id, self.id)`.
4. **`GetSuccessorList()`**: Return current local successor list.
5. **`Ping()`**: Return health status (`ok: true`).

## Postconditions

- Ring topology self-heals and converges to correct successor/predecessor order across all orchestrators.
- Ring successors are available for island-model migration routing.

## Failure cases

- Bootstrap node unreachable.
- Successor crashes (handled by falling back to successor list / next stabilization cycle).
- Network partitions or temporary disconnects.
