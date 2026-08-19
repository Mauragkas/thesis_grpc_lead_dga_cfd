# Orchestrator Documentation

Architecture documentation, domain model class diagram, sequence diagrams, and use-case specifications for the `orchestrator` crate.

## Contents

- **`domain-model.mmd` / `domain-model.png`**: Complete class diagram showing all orchestrator subsystems, interfaces, and relationships.
- **`use-cases/`**: 12 detailed use-case documents specifying the runtime behaviors of the orchestrator.
- **`sequence-diagrams/`**: 12 sequence diagrams illustrating message flows between orchestrator components and remote services.
- **`render-mermaid.sh`**: Helper script to render Mermaid `.mmd` diagrams to dark-mode `.png` images using `mmdc`.
- **`puppeteer-config.json`**: Headless Chromium configuration for Mermaid diagram rendering.
