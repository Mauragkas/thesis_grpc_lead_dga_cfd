"""Multi-probe Hilbert curve embeddings for order-preserving DHT keys.

Backed by the compiled Rust implementation (`hilbert_rs`) via PyO3/maturin
to ensure identical spatial discretization, bit depth, coordinate permutations,
and rounding between Python and Rust components.

Key format: "{curve_hex}{hex_hilbert}|{canonical_json}"
The curve id is the first hex digit, so each curve's keys form a contiguous
lexicographic band. The Rust DHT parses the whole hex prefix, so it never
needs to know about curves.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

_logger = logging.getLogger(__name__)

try:
    import hilbert_rs

    _HAS_RUST_EXTENSION = True
except ImportError:
    hilbert_rs = None  # type: ignore[assignment]
    _HAS_RUST_EXTENSION = False
    _logger.warning(
        "hilbert_rs compiled extension not found; falling back to pure Python implementation. "
        "Run 'uv run maturin develop -m ../hilbert/Cargo.toml' to compile the Rust Hilbert engine."
    )

# Bits per dimension. 10 dims x 16 bits = 160-bit index (40 hex digits).
BITS: int = hilbert_rs.BITS if _HAS_RUST_EXTENSION else 16

# Number of Hilbert curves (differently rotated coordinate axes).
NUM_CURVES: int = hilbert_rs.NUM_CURVES if _HAS_RUST_EXTENSION else 3

_PERM_CACHE: dict[int, list[list[int]]] = {}


def hilbert_encode(point: list[int], bits: int, ndims: int) -> int:
    """Encode an n-dimensional integer point into a 1D Hilbert index.

    Skilling (2004) "Programming the Hilbert curve" formulation. For 2 dims
    with 1 bit, this implementation visits (0,0)->0, (0,1)->1, (1,1)->2,
    (1,0)->3 — a valid 2x2 Hilbert orientation (up, right, down).
    """
    if _HAS_RUST_EXTENSION:
        return hilbert_rs.hilbert_encode(point, bits, ndims)

    X = list(point)
    M = 1 << (bits - 1)

    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(ndims):
            if X[i] & Q:
                X[0] ^= P          # invert
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t           # exchange
                X[i] ^= t
            Q >>= 1

    for i in range(1, ndims):
        X[i] ^= X[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if X[ndims - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(ndims):
        X[i] ^= t

    h = 0
    for i in range(bits):
        for j in range(ndims):
            bit = (X[j] >> (bits - 1 - i)) & 1
            h = (h << 1) | bit
    return h


def _permutations(ndims: int) -> list[list[int]]:
    """Deterministic coordinate permutations, one per probe curve."""
    if ndims not in _PERM_CACHE:
        rng = random.Random(20240807)
        perms: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(perms) < NUM_CURVES:
            perm = list(range(ndims))
            rng.shuffle(perm)
            key = tuple(perm)
            if key not in seen:
                seen.add(key)
                perms.append(perm)
        _PERM_CACHE[ndims] = perms
    return _PERM_CACHE[ndims]


def config_hilbert(
    cfg: dict[str, Any],
    dims: list[tuple[str, float, float]],
    bits: int = BITS,
    curve: int = 0,
) -> int:
    """Compute the Hilbert index for a config using the curve's dimension order."""
    if _HAS_RUST_EXTENSION:
        return hilbert_rs.config_hilbert(cfg, dims, bits=bits, curve=curve)

    perm = _permutations(len(dims))[curve]
    ndims = len(dims)
    max_val = (1 << bits) - 1
    point: list[int] = []
    for idx in perm:
        name, lo, hi = dims[idx]
        val = cfg[name]
        t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        point.append(int(round(t * max_val)))
    return hilbert_encode(point, bits, ndims)


def probe_keys(
    cfg: dict[str, Any],
    dims: list[tuple[str, float, float]],
    bits: int = BITS,
) -> list[str]:
    """Generate one order-preserving key per probe curve.

    Format: ``{curve_hex_digit}{hex_hilbert_index}|{canonical_json}``
    """
    if _HAS_RUST_EXTENSION:
        return hilbert_rs.probe_keys(cfg, dims, bits=bits)

    ndims = len(dims)
    hex_width = (ndims * bits + 3) // 4
    json_str = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    keys: list[str] = []
    for c in range(NUM_CURVES):
        h = config_hilbert(cfg, dims, bits, curve=c)
        keys.append(f"{c:x}{h:0{hex_width}x}|{json_str}")
    return keys


def parse_hilbert_key(key: str) -> dict[str, Any]:
    """Extract the config dict from a Hilbert-prefixed key."""
    if _HAS_RUST_EXTENSION:
        return hilbert_rs.parse_hilbert_key(key)

    _, _, json_str = key.partition("|")
    return json.loads(json_str)
