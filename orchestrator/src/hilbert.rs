//! Multi-probe Hilbert curve embeddings for order-preserving DHT keys.
//!
//! Re-exports from the unified `hilbert` crate (`hilbert_rs`) to eliminate
//! cross-language code duplication with Python (PyO3/maturin bindings).
//! Maps a multi-dimensional gene vector to a 1D scalar such that nearby points
//! in feature space map to nearby scalars.
//!
//! Multi-probe: each gene vector is indexed under `NUM_CURVES` Hilbert
//! curves built on rotated coordinate axes (deterministic dimension
//! permutations). Queries fan out to all curves and merge candidates,
//! recovering neighbors that sit far from the target in one curve's 1D
//! order but close in another's.
//!
//! Key format: `{curve_hex}{hex_hilbert}|{canonical_json}`.

pub use hilbert_rs::*;
