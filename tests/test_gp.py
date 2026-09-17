#!/usr/bin/env python3
"""
Multi-parameter aircraft configuration evaluator & Gaussian Process surrogate model.
Supports incremental dataset caching, permutation feature importance ranking,
and GP-based aerodynamic sensitivity analysis.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import time

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration & Physical Constants
# -----------------------------------------------------------------------------

BASELINE: dict[str, float] = {
    "fuse_length": 250.0,
    "fuse_max_diam": 10.0,
    "nose_ratio": 0.25,
    "tail_ratio": 0.35,
    "wing_span": 140.0,
    "wing_root_chord": 55.0,
    "wing_tip_chord": 25.0,
    "wing_sweep": 12.0,
    "wing_dihedral": 4.0,
    "wing_twist": -3.0,
    "wing_x_pos": 75.0,
    "wing_z_pos": -2.0,
    "naca_m": 2.0,
    "naca_p": 4.0,
    "naca_t": 12.0,
    "tail_x_pos": 210.0,
    "v_stab_height": 45.0,
    "v_stab_root": 35.0,
    "v_stab_tip": 18.0,
    "h_stab_span": 45.0,
    "h_stab_root": 28.0,
    "h_stab_tip": 15.0,
}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "wing_span": (80.0, 220.0),
    "wing_root_chord": (35.0, 75.0),
    "wing_tip_chord": (10.0, 45.0),
    "wing_sweep": (0.0, 25.0),
    "wing_dihedral": (0.0, 10.0),
    "wing_twist": (-6.0, 2.0),
    "wing_x_pos": (55.0, 95.0),
    "fuse_length": (180.0, 320.0),
    "fuse_max_diam": (7.0, 16.0),
    "naca_m": (0.0, 5.0),
    "naca_t": (8.0, 16.0),
}

RHO_MATERIAL = 250.0  # kg/m^3
RHO_AIR = 1.225       # kg/m^3
G = 9.81              # m/s^2
V_CRUISE = 15.0       # m/s
CG_POS = [0.08, 0.0, 0.0]
TARGET_FUSE_VOLUME_MM3 = 20000.0


# -----------------------------------------------------------------------------
# Geometry & Aerodynamic Solver
# -----------------------------------------------------------------------------

def wing_panel_volume(span: float, root_c: float, tip_c: float, t_ratio: float) -> float:
    return 0.685 * t_ratio * (span / 3.0) * (root_c**2 + root_c * tip_c + tip_c**2)


def calculate_fuselage_volume_mm3(p: dict[str, float]) -> float:
    l_nose = p["fuse_length"] * p["nose_ratio"]
    l_tail = p["fuse_length"] * p["tail_ratio"]
    l_mid = p["fuse_length"] - l_nose - l_tail
    r_max = p["fuse_max_diam"] / 2.0

    v_nose = (2.0 / 3.0) * np.pi * (r_max**2) * l_nose
    v_mid = np.pi * (r_max**2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0 / 3.0) * np.pi * l_tail * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)
    return float(v_nose + v_mid + v_tail)


def calculate_total_mass_and_weight(p: dict[str, float]) -> tuple[float, float]:
    v_fuse = calculate_fuselage_volume_mm3(p)
    t_main = p["naca_t"] / 100.0
    v_main = 2.0 * wing_panel_volume(p["wing_span"], p["wing_root_chord"], p["wing_tip_chord"], t_main)
    v_h = 2.0 * wing_panel_volume(p["h_stab_span"], p["h_stab_root"], p["h_stab_tip"], 0.10)
    v_v = wing_panel_volume(p["v_stab_height"], p["v_stab_root"], p["v_stab_tip"], 0.10)

    v_total_m3 = (v_fuse + v_main + v_h + v_v) * 1e-9
    mass_kg = v_total_m3 * RHO_MATERIAL
    return mass_kg, mass_kg * G


def build_naca_string(m: float, p: float, t: float) -> str:
    m_int = int(np.clip(round(m), 0, 9))
    p_int = int(np.clip(round(p), 0, 9)) if m_int > 0 else 0
    t_int = int(np.clip(round(t), 1, 40))
    return f"naca{m_int}{p_int}{t_int:02d}"


def evaluate_configuration(p: dict[str, float]) -> dict[str, float]:
    mass_kg, weight_n = calculate_total_mass_and_weight(p)
    s_ref = (2.0 * p["wing_span"] * (p["wing_root_chord"] + p["wing_tip_chord"]) / 2.0) * 1e-6
    scale = 1e-3

    naca_str = build_naca_string(p["naca_m"], p.get("naca_p", 4.0), p["naca_t"])
    airfoil_main = asb.Airfoil(naca_str)
    airfoil_tail = asb.Airfoil("naca0010")

    main_wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=p["wing_root_chord"] * scale, twist=0, airfoil=airfoil_main),
            asb.WingXSec(
                xyz_le=[
                    p["wing_span"] * scale * np.tan(np.radians(p["wing_sweep"])),
                    p["wing_span"] * scale,
                    p["wing_span"] * scale * np.tan(np.radians(p["wing_dihedral"])),
                ],
                chord=p["wing_tip_chord"] * scale,
                twist=p["wing_twist"],
                airfoil=airfoil_main,
            ),
        ],
    ).translate([p["wing_x_pos"] * scale, 0, p["wing_z_pos"] * scale])

    h_stab = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=p["h_stab_root"] * scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["h_stab_root"] - p["h_stab_tip"]) * scale, p["h_stab_span"] * scale, 0],
                chord=p["h_stab_tip"] * scale,
                twist=0,
                airfoil=airfoil_tail,
            ),
        ],
    ).translate([p["tail_x_pos"] * scale, 0, 0])

    v_stab = asb.Wing(
        name="Vertical Stabilizer",
        symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=p["v_stab_root"] * scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["v_stab_root"] - p["v_stab_tip"]) * scale, 0, p["v_stab_height"] * scale],
                chord=p["v_stab_tip"] * scale,
                twist=0,
                airfoil=airfoil_tail,
            ),
        ],
    ).translate([p["tail_x_pos"] * scale, 0, 0])

    airplane = asb.Airplane(
        name="Parametric Model",
        xyz_ref=CG_POS,
        wings=[main_wing, h_stab, v_stab],
    )

    cl_req = (2.0 * weight_n) / (RHO_AIR * (V_CRUISE**2) * s_ref)
    alpha_sweep = np.linspace(-4.0, 12.0, 17)
    cl_list, cd_list, cm_list = [], [], []

    cd_profile_base = 0.015 + 0.02 * (p["naca_t"] / 100.0)

    for alpha in alpha_sweep:
        op_point = asb.OperatingPoint(velocity=V_CRUISE, alpha=alpha)
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            spanwise_resolution=8,
            chordwise_resolution=4,
        )
        res = vlm.run()
        cl_list.append(float(res["CL"]))
        cd_list.append(float(res["CD"]) + cd_profile_base)
        cm_list.append(float(res["Cm"]))

    if cl_req < min(cl_list) or cl_req > max(cl_list):
        return {"ld": np.nan, "alpha_trim": np.nan, "cm_alpha": np.nan, "cl_req": cl_req}

    cd_trim = float(np.interp(cl_req, cl_list, cd_list))
    alpha_trim = float(np.interp(cl_req, cl_list, alpha_sweep))
    cm_alpha = float(np.gradient(cm_list, np.radians(alpha_sweep))[8])

    return {
        "ld": cl_req / cd_trim if cd_trim > 1e-4 else np.nan,
        "alpha_trim": alpha_trim,
        "cm_alpha": cm_alpha,
        "cl_req": cl_req,
        "mass_g": mass_kg * 1000.0,
    }


def compute_fitness(p: dict[str, float], aero_res: dict[str, float]) -> float:
    if np.isnan(aero_res["ld"]):
        return -20.0

    v_fuse = calculate_fuselage_volume_mm3(p)
    v_ratio = (v_fuse - TARGET_FUSE_VOLUME_MM3) / TARGET_FUSE_VOLUME_MM3
    v_pen = 10.0 * np.tanh(v_ratio**2)

    alpha_pen = 0.5 * abs(aero_res["alpha_trim"])

    margin = aero_res["cm_alpha"] + 0.05
    stab_pen = 15.0 * (np.log1p(np.exp(5.0 * margin)) / 5.0) ** 2

    raw_fitness = aero_res["ld"] - v_pen - alpha_pen - stab_pen
    return float(np.clip(raw_fitness, -20.0, 40.0))


# -----------------------------------------------------------------------------
# Sampling & Dataset Persistence
# -----------------------------------------------------------------------------

def sample_configurations(n_samples: int, seed: int = 42) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    param_keys = list(PARAM_BOUNDS.keys())
    d = len(param_keys)

    intervals = np.linspace(0, 1, n_samples + 1)
    samples_unit = np.empty((n_samples, d))
    for j in range(d):
        pts = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(pts)
        samples_unit[:, j] = pts

    configs: list[dict[str, float]] = []
    for i in range(n_samples):
        cfg = BASELINE.copy()
        for j, k in enumerate(param_keys):
            lo, hi = PARAM_BOUNDS[k]
            cfg[k] = float(lo + samples_unit[i, j] * (hi - lo))
        configs.append(cfg)

    return configs


def evaluate_batch(configs: list[dict[str, float]], max_workers: int = 8) -> list[dict[str, Any]]:
    def _eval(cfg: dict[str, float]) -> dict[str, Any]:
        aero = evaluate_configuration(cfg)
        fit = compute_fitness(cfg, aero)
        return {"config": cfg, "aero": aero, "fitness": fit}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_eval, configs))


def load_dataset_json(filepath: str | Path) -> list[dict[str, Any]]:
    path = Path(filepath)
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [
                    item for item in data
                    if isinstance(item, dict) and "config" in item and "fitness" in item
                ]
            return []
    except (json.JSONDecodeError, OSError):
        # Fallback in case of line-delimited JSON
        results: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    try:
                        obj = json.loads(line_str)
                        if isinstance(obj, dict) and "config" in obj and "fitness" in obj:
                            results.append(obj)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return results


def save_dataset_json(results: list[dict[str, Any]], filepath: str | Path) -> None:
    path = Path(filepath)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def append_dataset_json(results: list[dict[str, Any]], filepath: str | Path) -> None:
    existing = load_dataset_json(filepath)
    combined = existing + results
    save_dataset_json(combined, filepath)


# -----------------------------------------------------------------------------
# Gaussian Process Surrogate Pipeline
# -----------------------------------------------------------------------------

class GaussianProcessSurrogate:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        d = len(feature_names)

        kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(d),
            length_scale_bounds=(0.1, 100.0),
            nu=2.5,
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gp", GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-3,
                n_restarts_optimizer=10,
                normalize_y=True,
                random_state=42,
            )),
        ])

    def extract_features(self, configs: list[dict[str, float]]) -> np.ndarray:
        return np.array([[c[k] for k in self.feature_names] for c in configs], dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        gp: GaussianProcessRegressor = self.pipeline.named_steps["gp"]
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        return gp.predict(X_scaled, return_std=return_std)


# -----------------------------------------------------------------------------
# Feature Importance & Sensitivity Analysis
# -----------------------------------------------------------------------------

def analyze_feature_importance(
    surrogate: GaussianProcessSurrogate,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Computes permutation importance for the surrogate model on holdout test set.
    Returns sorted importances (mean, std) and corresponding feature names.
    """
    perm_result = permutation_importance(
        surrogate.pipeline,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=42,
    )
    sorted_idx = np.argsort(perm_result.importances_mean)[::-1]
    sorted_means = perm_result.importances_mean[sorted_idx]
    sorted_stds = perm_result.importances_std[sorted_idx]
    sorted_features = [surrogate.feature_names[i] for i in sorted_idx]

    return sorted_means, sorted_stds, sorted_features


# -----------------------------------------------------------------------------
# Visualization & Validation
# -----------------------------------------------------------------------------

def plot_gp_evaluation(
    surrogate: GaussianProcessSurrogate,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    cv_scores: np.ndarray,
    feat_means: np.ndarray,
    feat_stds: np.ndarray,
    feat_names: list[str],
    save_path: str = "gp_evaluation_results.png",
) -> None:
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    # 1. Parity Plot
    ax_parity = fig.add_subplot(gs[0, 0])
    ax_parity.errorbar(
        y_test,
        y_pred,
        yerr=1.96 * y_std,
        fmt="o",
        color="#2563eb",
        ecolor="#93c5fd",
        elinewidth=1.2,
        capsize=2,
        alpha=0.8,
        label=r"$\pm 1.96\sigma$",
    )
    min_val = min(float(np.min(y_test)), float(np.min(y_pred))) - 1.0
    max_val = max(float(np.max(y_test)), float(np.max(y_pred))) + 1.0
    ax_parity.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="1:1 Parity")
    ax_parity.set_xlim(min_val, max_val)
    ax_parity.set_ylim(min_val, max_val)
    ax_parity.set_xlabel("Ground Truth VLM Fitness")
    ax_parity.set_ylabel("GP Predicted Fitness")
    ax_parity.set_title(f"Surrogate Parity ($R^2 = {r2_score(y_test, y_pred):.3f}$)")
    ax_parity.grid(True, linestyle=":", alpha=0.6)
    ax_parity.legend()

    # 2. Residual Distribution
    residuals = y_test - y_pred
    ax_res = fig.add_subplot(gs[0, 1])
    ax_res.hist(residuals, bins=15, color="#059669", edgecolor="#064e3b", alpha=0.7)
    ax_res.axvline(0.0, color="red", linestyle="--", lw=1.2)
    ax_res.set_xlabel("Residual (True - Predicted)")
    ax_res.set_ylabel("Count")
    ax_res.set_title(f"Residuals (MAE = {mean_absolute_error(y_test, y_pred):.3f})")
    ax_res.grid(True, linestyle=":", alpha=0.6)

    # 3. K-Fold Cross-Validation Scores
    ax_cv = fig.add_subplot(gs[0, 2])
    ax_cv.bar(range(1, len(cv_scores) + 1), cv_scores, color="#7c3aed", alpha=0.7, edgecolor="#4c1d95")
    ax_cv.axhline(float(np.mean(cv_scores)), color="black", linestyle="--", label=f"Mean: {np.mean(cv_scores):.3f}")
    ax_cv.set_xlabel("Fold Index")
    ax_cv.set_ylabel(r"$R^2$ Score")
    ax_cv.set_ylim(0.0, 1.05)
    ax_cv.set_title("5-Fold Cross Validation")
    ax_cv.grid(True, linestyle=":", alpha=0.6)
    ax_cv.legend()

    # 4. Feature Importance Plot (Permutation Drop in R^2)
    ax_feat = fig.add_subplot(gs[0, 3])
    y_pos = np.arange(len(feat_names))
    ax_feat.barh(y_pos, feat_means[::-1], xerr=feat_stds[::-1], align="center", color="#d97706", edgecolor="#78350f", alpha=0.8, capsize=3)
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(feat_names[::-1])
    ax_feat.set_xlabel(r"$\Delta R^2$ (Permutation Importance)")
    ax_feat.set_title("Feature Sensitivity Ranking")
    ax_feat.grid(True, linestyle=":", alpha=0.6)

    # 5-8. 1D Sensitivity Slices for the Top 4 Ranked Features
    top_4_params = feat_names[:4]
    for idx, p_name in enumerate(top_4_params):
        ax_slice = fig.add_subplot(gs[1, idx])
        lo, hi = PARAM_BOUNDS[p_name]
        grid_vals = np.linspace(lo, hi, 25)

        sweep_configs = []
        for v in grid_vals:
            c = BASELINE.copy()
            c[p_name] = float(v)
            sweep_configs.append(c)

        X_sweep = surrogate.extract_features(sweep_configs)
        mu, sigma = surrogate.predict(X_sweep)

        vlm_results = evaluate_batch(sweep_configs, max_workers=4)
        true_y = [r["fitness"] for r in vlm_results]

        ax_slice.plot(grid_vals, mu, "b-", lw=2, label="GP Mean")
        ax_slice.fill_between(grid_vals, mu - 1.96 * sigma, mu + 1.96 * sigma, color="blue", alpha=0.15, label=r"$\pm 1.96\sigma$")
        ax_slice.scatter(grid_vals, true_y, color="red", s=18, zorder=5, label="VLM Ground Truth")

        ax_slice.set_xlabel(f"{p_name}")
        ax_slice.set_ylabel("Fitness Score")
        ax_slice.set_title(f"Rank {idx+1}: {p_name}")
        ax_slice.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_slice.legend()

    plt.savefig(save_path, dpi=200)
    print(f"Validation and feature selection plot generated -> {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------

def main() -> None:
    TARGET_SAMPLES = 1200
    dataset_file = Path(__file__).parent / "configs_and_scores.json"
    JSON_PATH = dataset_file if dataset_file.exists() else Path("configs_and_scores.json")

    print(f"=== 1. Dataset Verification ({JSON_PATH}) ===")
    existing_results = load_dataset_json(JSON_PATH)
    n_existing = len(existing_results)
    print(f"Existing samples on disk: {n_existing}/{TARGET_SAMPLES}")

    if n_existing >= TARGET_SAMPLES:
        print(f"Target count satisfied. Using existing {TARGET_SAMPLES} evaluations.")
        all_results = existing_results[:TARGET_SAMPLES]
    else:
        n_needed = TARGET_SAMPLES - n_existing
        print(f"Evaluating {n_needed} missing configuration(s) via VLM...")
        new_configs = sample_configurations(n_needed, seed=42 + n_existing)
        new_results = evaluate_batch(new_configs, max_workers=8)
        all_results = existing_results + new_results
        save_dataset_json(all_results, JSON_PATH)
        print(f"Dataset updated. Total samples: {len(all_results)}")

    valid_results = [r for r in all_results if r["fitness"] > -15.0]
    print(f"Usable converged configurations: {len(valid_results)}/{len(all_results)}")

    feature_names = list(PARAM_BOUNDS.keys())
    valid_configs = [r["config"] for r in valid_results]
    y_raw = np.array([r["fitness"] for r in valid_results], dtype=np.float64)

    surrogate = GaussianProcessSurrogate(feature_names=feature_names)
    X_raw = surrogate.extract_features(valid_configs)

    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

    print("\n=== 2. Fitting Gaussian Process Model ===")
    t_start = time.perf_counter()

    X_train_arr = np.asarray(X_train, dtype=np.float64)
    y_train_arr = np.asarray(y_train, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64)

    surrogate.fit(X_train_arr, y_train_arr)
    gp_model = surrogate.pipeline.named_steps["gp"]
    print(f"Optimized Kernel: {gp_model.kernel_}")

    y_pred, y_std = surrogate.predict(X_test_arr, return_std=True)

    r2 = float(r2_score(y_test_arr, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred)))
    mae = float(mean_absolute_error(y_test_arr, y_pred))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = np.asarray(cross_val_score(surrogate.pipeline, X_train_arr, y_train_arr, cv=kf, scoring="r2"), dtype=np.float64)

    elapsed_s = time.perf_counter() - t_start
    elapsed_msg = f"{elapsed_s / 60.0:.2f} min" if elapsed_s >= 60.0 else f"{elapsed_s:.2f} sec"

    print("\n=== 3. Computing Permutation Feature Importance ===")
    feat_means, feat_stds, feat_names = analyze_feature_importance(surrogate, X_test_arr, y_test_arr)

    print("\n" + "=" * 55)
    print("GAUSSIAN PROCESS EVALUATION & FEATURE RANKING")
    print("=" * 55)
    print(f"Holdout Test R^2 Score:        {r2:.4f}")
    print(f"Holdout Root Mean Squared Err: {rmse:.4f}")
    print(f"Holdout Mean Absolute Err:     {mae:.4f}")
    print(f"5-Fold Cross-Validation R^2:   {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"Elapsed Time:                  {elapsed_msg}")
    print("-" * 55)
    print(f"{'Rank':<5} {'Parameter':<20} {'Delta R^2':<12} {'Std Dev':<10}")
    print("-" * 55)
    for rank, (name, mean_imp, std_imp) in enumerate(zip(feat_names, feat_means, feat_stds), start=1):
        print(f"{rank:<5} {name:<20} {mean_imp:<12.4f} {std_imp:<10.4f}")
    print("=" * 55)

    print("\n=== 4. Generating Visualization Plots ===")
    plot_gp_evaluation(
        surrogate=surrogate,
        X_test=X_test_arr,
        y_test=y_test_arr,
        y_pred=y_pred,
        y_std=y_std,
        cv_scores=cv_scores,
        feat_means=feat_means,
        feat_stds=feat_stds,
        feat_names=feat_names,
        save_path=str(Path(__file__).parent / "gp_evaluation_results.png"),
    )


if __name__ == "__main__":
    main()
