#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).parent / "../worker/src/worker"))
sys.path.insert(0, str(Path(__file__).parent / "../worker/src"))
sys.path.insert(0, str(Path(__file__).parent / "../worker"))
from config import BASELINE

LEAD_URL = "http://localhost:2001"
N_PLANES = 200


def make_plane(i: int) -> dict[str, Any]:
    rng = random.Random(1000 + i)  # deterministic

    cfg = dict(BASELINE)

    # Small variations around the baseline
    cfg["fuse_length"] = round(BASELINE["fuse_length"] + rng.uniform(-30, 60), 2)
    cfg["fuse_max_diam"] = round(BASELINE["fuse_max_diam"] + rng.uniform(-2, 5), 2)
    cfg["wing_span"] = round(BASELINE["wing_span"] + rng.uniform(-25, 50), 2)
    cfg["wing_root_chord"] = round(BASELINE["wing_root_chord"] + rng.uniform(-10, 15), 2)
    cfg["wing_tip_chord"] = round(BASELINE["wing_tip_chord"] + rng.uniform(-8, 12), 2)
    cfg["wing_sweep"] = round(BASELINE["wing_sweep"] + rng.uniform(-8, 10), 2)
    cfg["wing_dihedral"] = round(BASELINE["wing_dihedral"] + rng.uniform(-2, 4), 2)
    cfg["wing_twist"] = round(BASELINE["wing_twist"] + rng.uniform(-3, 3), 2)
    cfg["wing_x_pos"] = round(BASELINE["wing_x_pos"] + rng.uniform(-10, 12), 2)

    # Keep integers as integers
    cfg["naca_m"] = int(BASELINE["naca_m"])
    cfg["naca_p"] = int(BASELINE["naca_p"])
    cfg["naca_t"] = int(BASELINE["naca_t"])

    return cfg


def config_key(cfg: dict[str, Any]) -> str:
    """
    Canonical string representation of the config.

    This makes the config itself the key, so exact same config => exact same key.
    """
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def dummy_fitness(cfg: dict[str, Any], i: int) -> float:
    """
    Dummy score: no VLM, no aero solve.
    Just a reproducible placeholder score.
    """
    score = 1000.0
    score -= abs(cfg["wing_span"] - BASELINE["wing_span"]) * 0.8
    score -= abs(cfg["fuse_length"] - BASELINE["fuse_length"]) * 0.15
    score -= abs(cfg["wing_sweep"] - BASELINE["wing_sweep"]) * 0.5
    score += (i % 13) * 0.25
    return round(score, 4)


def put_local(key: str, value: str) -> bool:
    # Keys are JSON strings, so they must be URL-encoded for the path.
    encoded_key = quote(key, safe="")
    r = requests.post(f"{LEAD_URL}/kv/{encoded_key}", data=value, timeout=5)
    return r.status_code in (200, 201)


def main() -> None:
    ok = 0
    failed = 0

    for i in range(N_PLANES):
        cfg = make_plane(i)
        fit = dummy_fitness(cfg, i)

        # Config is the key; fitness is the value.
        key = config_key(cfg)
        value = json.dumps(
            {
                "fitness_score": fit,
                "fitness_type": "dummy",
            }
        )

        if put_local(key, value):
            ok += 1
            print(f"stored fitness={fit} key={key}")
        else:
            failed += 1
            print(f"FAILED key={key}")

    print(f"\nDone. ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
