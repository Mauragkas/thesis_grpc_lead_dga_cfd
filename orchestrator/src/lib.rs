//! Library facade so integration tests (in `tests/`) can exercise the
//! same modules the binary uses. Keeps the crate testable in isolation.
#![allow(clippy::all)]

pub mod config;
pub mod evaluator;
pub mod ga;
pub mod gene_store;
pub mod hilbert;
pub mod lead_store;
pub mod migration;
pub mod neighbor_store;
pub mod proto;
pub mod ring;
pub mod transport;
