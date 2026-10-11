"""Tests for Hilbert space-filling curve embedding logic.

Verifies:
1. Compiled Rust extension (hilbert_rs) is loaded and operational.
2. Order-preserving multi-probe key generation and parsing roundtrip.
3. Near neighbor spatial proximity preservation in Hilbert keys.
"""

from __future__ import annotations

import json
from config import BASELINE, GENE_BOUNDS
import hilbert
import hilbert_rs


def test_hilbert_extension_loaded():
    assert hilbert._HAS_RUST_EXTENSION is True
    assert hilbert.BITS == 16
    assert hilbert.NUM_CURVES == 3


def test_probe_keys_count_and_format():
    dims = [(b.name, b.low, b.high) for b in GENE_BOUNDS]
    keys = hilbert.probe_keys(BASELINE, dims)
    assert len(keys) == hilbert.NUM_CURVES

    # Each key begins with curve index (0, 1, 2)
    curve_prefixes = [k[0] for k in keys]
    assert sorted(curve_prefixes) == ["0", "1", "2"]

    for k in keys:
        assert "|" in k
        parsed = hilbert.parse_hilbert_key(k)
        assert parsed == BASELINE


def test_hilbert_encode_consistency():
    # 2D point (0,0)->0, (0,1)->1, (1,1)->2, (1,0)->3
    assert hilbert.hilbert_encode([0, 0], 1, 2) == 0
    assert hilbert.hilbert_encode([0, 1], 1, 2) == 1
    assert hilbert.hilbert_encode([1, 1], 1, 2) == 2
    assert hilbert.hilbert_encode([1, 0], 1, 2) == 3


def test_rust_key_generator_class():
    kg = hilbert_rs.HilbertKeyGenerator(10)
    genes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25]
    keys = kg.keys_for(genes)
    assert len(keys) == 3

    for k in keys:
        parsed = kg.parse_key(k)
        assert parsed is not None
        assert len(parsed) == 10
        for g_orig, g_parsed in zip(genes, parsed):
            assert abs(g_orig - g_parsed) < 1e-9


def test_spatial_proximity_preserved():
    dims = [(b.name, b.low, b.high) for b in GENE_BOUNDS]
    cfg_base = dict(BASELINE)
    cfg_near = dict(BASELINE)
    cfg_near["fuse_length"] = BASELINE["fuse_length"] + 0.1  # very close

    cfg_far = dict(BASELINE)
    cfg_far["fuse_length"] = GENE_BOUNDS[0].high  # far boundary

    keys_base = hilbert.probe_keys(cfg_base, dims)
    keys_near = hilbert.probe_keys(cfg_near, dims)
    keys_far = hilbert.probe_keys(cfg_far, dims)

    def leading_prefix(key: str) -> int:
        hex_part = key.split("|")[0]
        return int(hex_part[:16], 16)

    min_near_dist = min(
        abs(leading_prefix(kb) - leading_prefix(kn))
        for kb, kn in zip(keys_base, keys_near)
    )
    min_far_dist = min(
        abs(leading_prefix(kb) - leading_prefix(kf))
        for kb, kf in zip(keys_base, keys_far)
    )

    assert min_near_dist < min_far_dist


def test_arbitrary_num_curves_probe_keys_and_config_hilbert():
    dims = [(b.name, b.low, b.high) for b in GENE_BOUNDS]
    keys_3 = hilbert.probe_keys(BASELINE, dims, num_curves=3)
    keys_8 = hilbert.probe_keys(BASELINE, dims, num_curves=8)

    assert len(keys_3) == 3
    assert len(keys_8) == 8
    assert keys_8[:3] == keys_3

    # Ensure curves 0..7 all yield distinct Hilbert indices (no modulo-3 wraparound)
    indices = [hilbert.config_hilbert(BASELINE, dims, curve=c) for c in range(8)]
    assert len(set(indices)) == 8

