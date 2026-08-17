//! Multi-probe Hilbert curve embeddings for order-preserving DHT keys.
//!
//! Rust port of `worker/hilbert.py` (the embedding used by
//! `tests/test_sim.py`). Maps a multi-dimensional gene vector to a 1D
//! scalar such that nearby points in feature space map to nearby scalars.
//! The scalar is used as a hex prefix on DHT keys so that lexicographic
//! range queries approximate Euclidean nearest-neighbor search.
//!
//! Multi-probe: each gene vector is indexed under `NUM_CURVES` Hilbert
//! curves built on rotated coordinate axes (deterministic dimension
//! permutations). Queries fan out to all curves and merge candidates,
//! recovering neighbors that sit far from the target in one curve's 1D
//! order but close in another's.
//!
//! Key format: `{curve_hex}{hex_hilbert}|{canonical_json}`. The curve id
//! is the first hex digit, so each curve's keys form a contiguous
//! lexicographic band. The Rust DHT parses the whole hex prefix, so it
//! never needs to know about curves.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// Bits per dimension. 10 dims x 16 bits = 160-bit index (40 hex digits).
pub const BITS: usize = 16;

/// Number of Hilbert curves (differently rotated coordinate axes).
pub const NUM_CURVES: usize = 3;

/// Deterministic seed for the coordinate permutations (same seed value as
/// the Python reference's `random.Random(20240807)`).
const PERM_SEED: u64 = 20240807;

// ---------------------------------------------------------------------------
// SRP: pure Hilbert encoding. Knows nothing about keys, storage, or
// transport — it only maps a normalized point to a Hilbert index.
// ---------------------------------------------------------------------------
pub struct HilbertEncoder {
    ndims: usize,
    bits: usize,
    max_val: u64,
    perms: Vec<Vec<usize>>,
}

impl HilbertEncoder {
    pub fn new(ndims: usize, bits: usize) -> Self {
        Self {
            ndims,
            bits,
            max_val: (1u64 << bits) - 1,
            perms: permutations(ndims),
        }
    }

    /// Encode a normalized `[0,1]^ndims` point into a zero-padded hex
    /// Hilbert index for the given probe curve.
    pub fn encode_hex(&self, point: &[f64], curve: usize, hex_width: usize) -> String {
        debug_assert_eq!(point.len(), self.ndims);
        let perm = &self.perms[curve % self.perms.len()];
        let mut scaled = vec![0u64; self.ndims];
        for (out_idx, &dim) in perm.iter().enumerate() {
            let t = point[dim].clamp(0.0, 1.0);
            scaled[out_idx] = py_round(t * self.max_val as f64);
        }
        hilbert_encode_hex(&scaled, self.bits, self.ndims, hex_width)
    }
}

/// Skilling (2004) "Programming the Hilbert curve" formulation, emitting
/// the interleaved bit sequence directly as a zero-padded hex string.
/// This avoids big-integer arithmetic for >128-bit indices.
fn hilbert_encode_hex(point: &[u64], bits: usize, ndims: usize, hex_width: usize) -> String {
    let mut x = point.to_vec();
    let m: u64 = 1 << (bits - 1);

    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..ndims {
            if x[i] & q != 0 {
                x[0] ^= p; // invert
            } else {
                let t = (x[0] ^ x[i]) & p; // exchange
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }

    for i in 1..ndims {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u64;
    q = m;
    while q > 1 {
        if x[ndims - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for item in x.iter_mut().take(ndims) {
        *item ^= t;
    }

    // Accumulate the interleaved bit sequence into hex digits.
    let mut hex = String::with_capacity(hex_width);
    let mut nibble: u8 = 0;
    let mut nibble_bits = 0;
    for i in 0..bits {
        for item in x.iter().take(ndims) {
            let bit = ((item >> (bits - 1 - i)) & 1) as u8;
            nibble = (nibble << 1) | bit;
            nibble_bits += 1;
            if nibble_bits == 4 {
                hex.push(char::from_digit(nibble as u32, 16).unwrap());
                nibble = 0;
                nibble_bits = 0;
            }
        }
    }
    if nibble_bits > 0 {
        nibble <<= 4 - nibble_bits;
        hex.push(char::from_digit(nibble as u32, 16).unwrap());
    }
    while hex.len() < hex_width {
        hex.insert(0, '0');
    }
    hex
}

/// Python-compatible round (round-half-to-even), matching the reference
/// `hilbert.py`'s `int(round(...))`.
fn py_round(x: f64) -> u64 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        (f + 1.0) as u64
    } else if diff < 0.5 {
        f as u64
    } else {
        let fi = f as u64;
        if fi.is_multiple_of(2) {
            fi
        } else {
            fi + 1
        }
    }
}

/// Deterministic coordinate permutations, one per probe curve.
fn permutations(ndims: usize) -> Vec<Vec<usize>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Vec<Vec<usize>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(perms) = guard.get(&ndims) {
        return perms.clone();
    }
    let mut rng = StdRng::seed_from_u64(PERM_SEED);
    let mut perms: Vec<Vec<usize>> = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    while perms.len() < NUM_CURVES {
        let mut perm: Vec<usize> = (0..ndims).collect();
        perm.shuffle(&mut rng);
        if seen.insert(perm.clone()) {
            perms.push(perm);
        }
    }
    guard.insert(ndims, perms.clone());
    perms
}

// ---------------------------------------------------------------------------
// SRP/OCP: turns a gene vector into the set of order-preserving DHT keys.
// New embeddings (Z-order, LSH, single-curve) implement this trait.
// ---------------------------------------------------------------------------
pub trait GeneKeyGenerator: Send + Sync {
    fn keys_for(&self, genes: &[f64]) -> Vec<String>;
    fn parse_key(&self, key: &str) -> Option<Vec<f64>>;
}

/// Hilbert multi-probe key generator. Produces `NUM_CURVES` keys per gene,
/// one per rotated coordinate axis.
pub struct HilbertKeyGenerator {
    encoder: HilbertEncoder,
    hex_width: usize,
}

impl HilbertKeyGenerator {
    pub fn new(ndims: usize) -> Self {
        Self {
            encoder: HilbertEncoder::new(ndims, BITS),
            hex_width: (ndims * BITS).div_ceil(4),
        }
    }
}

impl GeneKeyGenerator for HilbertKeyGenerator {
    fn keys_for(&self, genes: &[f64]) -> Vec<String> {
        let json = serde_json::to_string(genes).unwrap_or_default();
        (0..NUM_CURVES)
            .map(|c| {
                let h = self.encoder.encode_hex(genes, c, self.hex_width);
                format!("{c:x}{h}|{json}")
            })
            .collect()
    }

    fn parse_key(&self, key: &str) -> Option<Vec<f64>> {
        let (_, json) = key.split_once('|')?;
        serde_json::from_str(json).ok()
    }
}
