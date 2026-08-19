# LEAD Sequence Diagrams

Visual sequence diagrams capturing message flows across LEAD node components and remote peers.

## Diagrams

- **`startup/`**: Node startup, join wait loop, background maintenance task scheduling, and server bindings.
- **`config/`**: Environment variable parsing and defaults fallback.
- **`join/`**: Multi-vnode bootstrap join and successor list resolution.
- **`stabilization/`**: Periodic Chord stabilization loop, successor health repair, and predecessor reconciliation.
- **`kv-routed/`**: Routed key-value operations using LearnedHASH prediction and Chord lookup.
- **`range-query/`**: Multi-node range query scanning with local overscanning and recursive successor forwarding.
- **`learning-fedavg/`**: Distribution drift detection, neighbor heartbeat consensus, Transient Coordinator election, FedAvg model aggregation, and key re-migration.
- **`pid-tuning/`**: Online 2-bit PID controller tuning of leaf model scale and centering offset.
