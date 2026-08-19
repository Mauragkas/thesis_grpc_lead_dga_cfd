# Use Case: Join Ring via Bootstrap Node

## Description

Integrates all local virtual nodes into an existing LEAD Chord DHT cluster by querying a known bootstrap node to discover initial successors and successor lists.

## Primary actor

- `LeadNode`

## Supporting systems

- `RemoteNode` (`GrpcRemote`)
- Bootstrap LEAD node
- Local `VirtualNode` instances

## Preconditions

- `JOIN_URI` is configured and points to a reachable cluster node.
- Local `VirtualNode` instances have been assigned unique `PeerHASH` VIDs.

## Main steps

1. `LeadNode::join(known_uri)` is invoked.
2. For each local `VirtualNode` $v$:
   1. The node sends `find_successor(known_uri, v.vid, v.vid)` via gRPC.
   2. The bootstrap node routes the lookup and returns the `NodeAddr` of the responsible successor `succ`.
   3. The node queries `get_successor_list(succ.address, succ.id)` to obtain the successor's backup list.
   4. The local successor list for $v$ is populated with `succ` prepended to the remote list (up to $R$ entries).
   5. Immediate finger table entry `fingers[0]` is updated to `succ`.
   6. $v$ is marked active.
3. The main startup task checks whether any vnodes are still alone (pointing to themselves) and retries every second until all vnodes have joined.

## Postconditions

- All local virtual nodes have established valid successor connections to existing ring nodes.
- Subsequent Chord stabilization will establish bidirectional predecessor links and populate higher finger entries.

## Failure cases

- Bootstrap node unreachable: join loop logs a warning and retries up to 300 times.
