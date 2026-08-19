# Use Case: Perform Periodic Ring Maintenance

## Description

Executes continuous background maintenance tasks across all hosted virtual nodes to maintain ring topology, self-heal failed nodes, refresh base-10 finger tables, and reconcile successor/predecessor pointers.

## Primary actor

- `LeadNode` maintenance scheduler

## Supporting systems

- `VirtualNode`
- `RemoteNode` (`GrpcRemote`)
- `RingMath` (`finger_start`, `in_range`)

## Main steps: Stabilization (`stabilize_all()`, every 2s)

For each active virtual node $v$:
1. **Successor Repair**: Ping the current immediate successor. If unreachable, pop dead entries from `successor_list` until a live node is reached or fallback to self.
2. **Self Repair**: If $v$ is alone (successor is self) but has a live predecessor, adopt that predecessor as successor.
3. **Predecessor Reconciliation**: Query the successor's current predecessor $x$ via `get_predecessor(succ.address, succ.id)`. If $x$ is live and $x \in (v.vid, succ.id)$, prepend $x$ as the new immediate successor.
4. **Notification and List Refresh**: Send `notify` to the successor announcing $v$, then fetch the successor's updated successor list.

## Main steps: Fix Fingers (`fix_fingers_all()`, every 5s)

For each active virtual node $v$:
1. Select the next finger index $i \in [1..17]$ round-robin.
2. Calculate target key using base-10 exponential jump: $\text{target} = v.vid + 10^{i-1}$.
3. Resolve `find_successor(v.vid, target)` across the ring (with 5-second timeout).
4. Update `v.fingers[i]` with the resulting `NodeAddr`.

## Main steps: Check Predecessor (`check_predecessor_all()`, every 10s)

For each active virtual node $v$:
1. If `predecessor` is `Some(p)`, send a health ping to $p.address$.
2. If the ping fails or times out, set `v.predecessor = None`.

## Postconditions

- The distributed ring topology automatically self-heals after node joins, graceful leaves, and node failures.
- Routing tables (finger tables) remain accurate with $O(\log_{10} N)$ hops.
