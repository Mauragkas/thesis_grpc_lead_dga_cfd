# Orchestrator Documentation

Comprehensive functional and architectural documentation for the `orchestrator` crate.

## Documentation Structure

- [`use-cases/`](use-cases/README.md): Step-by-step operational workflows and failure handling across all 14 use cases.
- [`sequence-diagrams/`](sequence-diagrams/README.md): Mermaid sequence diagrams modeling execution flows between orchestrator modules and remote microservices.
- [`domain-model.mmd`](domain-model.mmd): Domain object model illustrating relationships between entities, value objects, ports, and adapters.

## Key Subsystems

1. **Evolutionary Optimization**: Island-model Genetic Algorithm with (μ + λ) survivor selection, Gaussian mutation, and migration across a Chord-like ring.
2. **Multi-Tier Evaluation ($\epsilon$-Bypass)**: Hierarchical evaluation strategy combining 0-FLOP exact cache hits, fast external MLP surrogate predictions, and true AeroSandbox aerodynamic simulations.
3. **Distributed Spatial Persistence**: Order-preserving multi-probe Hilbert space-filling curve embedding indexing individuals into the LEAD DHT.
4. **Surrogate Coordination**: Asynchronous integration with the `surrogate_node` microservice for online sliding window sample training.
