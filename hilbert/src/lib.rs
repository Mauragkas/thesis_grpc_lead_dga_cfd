//! Multi-probe Hilbert curve embeddings for order-preserving DHT keys.
//!
//! Maps a multi-dimensional configuration or gene vector to a 1D scalar
//! such that nearby points in feature space map to nearby scalars. The scalar
//! is used as a hex prefix on DHT keys so that lexicographic range queries
//! approximate Euclidean nearest-neighbor search.
//!
//! Multi-probe: each point is indexed under `NUM_CURVES` Hilbert curves built on
//! rotated coordinate axes (deterministic dimension permutations). Queries fan
//! out to all curves and merge candidates, recovering neighbors that sit far from
//! the target in one curve's 1D order but close in another's.
//!
//! Key format: `{curve_hex}{hex_hilbert}|{canonical_json}`.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// Bits per dimension. 10 dims x 16 bits = 160-bit index (40 hex digits).
pub const BITS: usize = 16;

/// Number of Hilbert curves (differently rotated coordinate axes).
pub const NUM_CURVES: usize = 3;

/// Deterministic seed for coordinate permutations.
pub const PERM_SEED: u64 = 20240807;

// ---------------------------------------------------------------------------
// SRP: pure Hilbert encoding. Maps a normalized point to a Hilbert index.
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
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

    pub fn ndims(&self) -> usize {
        self.ndims
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn max_val(&self) -> u64 {
        self.max_val
    }

    pub fn permutations(&self) -> &[Vec<usize>] {
        &self.perms
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

    /// Encode an integer point directly into a zero-padded hex Hilbert index.
    pub fn encode_integer_point(&self, point: &[u64], hex_width: usize) -> String {
        debug_assert_eq!(point.len(), self.ndims);
        hilbert_encode_hex(point, self.bits, self.ndims, hex_width)
    }
}

/// Skilling (2004) "Programming the Hilbert curve" coordinate transformation.
pub fn hilbert_transform(point: &[u64], bits: usize, ndims: usize) -> Vec<u64> {
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
    x
}

/// Skilling (2004) formulation, interleaving bits to produce the exact hex scalar representation.
pub fn hilbert_encode_to_hex_scalar(point: &[u64], bits: usize, ndims: usize) -> String {
    let x = hilbert_transform(point, bits, ndims);
    let total_bits = bits * ndims;
    let pad = (4 - (total_bits % 4)) % 4;

    let mut hex = String::with_capacity((total_bits + pad) / 4);
    let mut nibble: u8 = 0;
    let mut nibble_bits = pad; // Pre-load with pad zero bits so bits align to LSB

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
    if hex.is_empty() {
        hex.push('0');
    }
    hex
}

/// Skilling (2004) formulation emitting a zero-padded hex string of specified width.
pub fn hilbert_encode_hex(point: &[u64], bits: usize, ndims: usize, hex_width: usize) -> String {
    let scalar_hex = hilbert_encode_to_hex_scalar(point, bits, ndims);
    if scalar_hex.len() < hex_width {
        format!("{:0>width$}", scalar_hex, width = hex_width)
    } else {
        scalar_hex
    }
}

/// Python-compatible round (round-half-to-even), matching the reference
/// `hilbert.py`'s `int(round(...))`.
pub fn py_round(x: f64) -> u64 {
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
pub fn permutations(ndims: usize) -> Vec<Vec<usize>> {
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
// ---------------------------------------------------------------------------
pub trait GeneKeyGenerator: Send + Sync {
    fn keys_for(&self, genes: &[f64]) -> Vec<String>;
    fn parse_key(&self, key: &str) -> Option<Vec<f64>>;
}

/// Hilbert multi-probe key generator. Produces `NUM_CURVES` keys per gene,
/// one per rotated coordinate axis.
#[derive(Clone, Debug)]
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

    pub fn hex_width(&self) -> usize {
        self.hex_width
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

// ---------------------------------------------------------------------------
// PyO3 Python bindings (activated with feature "python")
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict};

    /// Convert a hex string into a Python int of arbitrary precision.
    fn hex_to_py_int<'py>(py: Python<'py>, hex_str: &str) -> PyResult<Bound<'py, PyAny>> {
        let builtins = py.import_bound("builtins")?;
        let int_fn = builtins.getattr("int")?;
        int_fn.call1((hex_str, 16))
    }

    /// Encode an n-dimensional integer point into a 1D Hilbert integer index.
    #[pyfunction]
    #[pyo3(signature = (point, bits, ndims))]
    fn hilbert_encode<'py>(
        py: Python<'py>,
        point: Vec<u64>,
        bits: usize,
        ndims: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if point.len() != ndims {
            return Err(PyValueError::new_err(format!(
                "Point length {} != ndims {}",
                point.len(),
                ndims
            )));
        }
        let hex = hilbert_encode_to_hex_scalar(&point, bits, ndims);
        hex_to_py_int(py, &hex)
    }

    /// Compute the Hilbert index for a config dict using the curve's dimension order.
    #[pyfunction]
    #[pyo3(signature = (cfg, dims, bits=BITS, curve=0))]
    fn config_hilbert<'py>(
        py: Python<'py>,
        cfg: &Bound<'py, PyDict>,
        dims: Vec<(String, f64, f64)>,
        bits: usize,
        curve: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ndims = dims.len();
        if ndims == 0 {
            return Err(PyValueError::new_err("dims cannot be empty"));
        }
        let perms = permutations(ndims);
        let perm = &perms[curve % perms.len()];
        let max_val = (1u64 << bits) - 1;

        let mut point = Vec::with_capacity(ndims);
        for &idx in perm {
            let (ref name, lo, hi) = dims[idx];
            let py_val = cfg
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("Missing key '{name}' in config")))?;
            let val: f64 = py_val.extract()?;
            let denom = hi - lo;
            let t = if denom.abs() < 1e-12 {
                0.0
            } else {
                ((val - lo) / denom).clamp(0.0, 1.0)
            };
            point.push(py_round(t * max_val as f64));
        }

        let hex = hilbert_encode_to_hex_scalar(&point, bits, ndims);
        hex_to_py_int(py, &hex)
    }

    /// Generate one order-preserving key per probe curve.
    ///
    /// Format: ``{curve_hex_digit}{hex_hilbert_index}|{canonical_json}``
    #[pyfunction]
    #[pyo3(signature = (cfg, dims, bits=BITS))]
    fn probe_keys<'py>(
        py: Python<'py>,
        cfg: &Bound<'py, PyDict>,
        dims: Vec<(String, f64, f64)>,
        bits: usize,
    ) -> PyResult<Vec<String>> {
        let ndims = dims.len();
        if ndims == 0 {
            return Err(PyValueError::new_err("dims cannot be empty"));
        }
        let hex_width = (ndims * bits).div_ceil(4);

        // Serialize canonical JSON using Python's json module to guarantee 1:1 format
        let json_mod = py.import_bound("json")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("sort_keys", true)?;
        kwargs.set_item("separators", (",", ":"))?;
        let json_str: String = json_mod
            .call_method("dumps", (cfg,), Some(&kwargs))?
            .extract()?;

        let perms = permutations(ndims);
        let max_val = (1u64 << bits) - 1;

        let mut keys = Vec::with_capacity(NUM_CURVES);
        for c in 0..NUM_CURVES {
            let perm = &perms[c % perms.len()];
            let mut point = Vec::with_capacity(ndims);
            for &idx in perm {
                let (ref name, lo, hi) = dims[idx];
                let py_val = cfg.get_item(name)?.ok_or_else(|| {
                    PyValueError::new_err(format!("Missing key '{name}' in config"))
                })?;
                let val: f64 = py_val.extract()?;
                let denom = hi - lo;
                let t = if denom.abs() < 1e-12 {
                    0.0
                } else {
                    ((val - lo) / denom).clamp(0.0, 1.0)
                };
                point.push(py_round(t * max_val as f64));
            }
            let hex = hilbert_encode_hex(&point, bits, ndims, hex_width);
            keys.push(format!("{c:x}{hex}|{json_str}"));
        }
        Ok(keys)
    }

    /// Extract the config dict from a Hilbert-prefixed key.
    #[pyfunction]
    #[pyo3(signature = (key))]
    fn parse_hilbert_key<'py>(py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let (_, json_str) = key
            .split_once('|')
            .ok_or_else(|| PyValueError::new_err(format!("Invalid Hilbert key format: '{key}'")))?;
        let json_mod = py.import_bound("json")?;
        json_mod.call_method1("loads", (json_str,))
    }

    /// Python wrapper for HilbertKeyGenerator (genes list[float] interface).
    #[pyclass(name = "HilbertKeyGenerator")]
    struct PyHilbertKeyGenerator {
        inner: HilbertKeyGenerator,
    }

    #[pymethods]
    impl PyHilbertKeyGenerator {
        #[new]
        fn new(ndims: usize) -> Self {
            Self {
                inner: HilbertKeyGenerator::new(ndims),
            }
        }

        fn keys_for(&self, genes: Vec<f64>) -> Vec<String> {
            self.inner.keys_for(&genes)
        }

        fn parse_key(&self, key: &str) -> Option<Vec<f64>> {
            self.inner.parse_key(key)
        }
    }

    #[pymodule]
    fn hilbert_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("BITS", BITS)?;
        m.add("NUM_CURVES", NUM_CURVES)?;
        m.add("PERM_SEED", PERM_SEED)?;
        m.add_function(wrap_pyfunction!(hilbert_encode, m)?)?;
        m.add_function(wrap_pyfunction!(config_hilbert, m)?)?;
        m.add_function(wrap_pyfunction!(probe_keys, m)?)?;
        m.add_function(wrap_pyfunction!(parse_hilbert_key, m)?)?;
        m.add_class::<PyHilbertKeyGenerator>()?;
        Ok(())
    }
}
