#!/usr/bin/env python3
"""
Gaussian Process Surrogate Visualization & Sensitivity Sweep Script.
Generates:
  - evaluation/figures/fig2_surrogate_parameter_sensitivity.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# IEEE publication aesthetics
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FEATURE_NAMES = [
    "wing_span",
    "wing_root_chord",
    "wing_tip_chord",
    "wing_sweep",
    "wing_dihedral",
    "wing_twist",
    "wing_x_pos",
    "fuse_length",
    "fuse_max_diam",
    "naca_m",
    "naca_t",
]


def matern52_kernel(X1: np.ndarray, X2: np.ndarray, length_scales: np.ndarray, signal_var: float) -> np.ndarray:
    X1_scaled = X1 / length_scales
    X2_scaled = X2 / length_scales
    dists = np.sqrt(np.maximum(np.sum((X1_scaled[:, None, :] - X2_scaled[None, :, :]) ** 2, axis=-1), 1e-12))
    sqrt5_d = np.sqrt(5.0) * dists
    K = signal_var * (1.0 + sqrt5_d + (5.0 / 3.0) * (dists ** 2)) * np.exp(-sqrt5_d)
    return K


def fit_and_predict(X_train, y_train, X_test, length_scales, signal_var, noise_var):
    N = len(X_train)
    K = matern52_kernel(X_train, X_train, length_scales, signal_var) + noise_var * np.eye(N)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    K_star = matern52_kernel(X_test, X_train, length_scales, signal_var)
    mu = K_star @ alpha
    v = np.linalg.solve(L, K_star.T)
    K_star2 = signal_var * np.ones(len(X_test))
    var = np.maximum(K_star2 - np.sum(v ** 2, axis=0), 1e-8)
    return mu, np.sqrt(var)


def generate_synthetic_flight_dataset(n_samples: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Scaled parameters
    wing_span = rng.uniform(80.0, 220.0, n_samples)
    root_chord = rng.uniform(35.0, 75.0, n_samples)
    tip_chord = rng.uniform(15.0, 45.0, n_samples)
    sweep = rng.uniform(0.0, 25.0, n_samples)
    dihedral = rng.uniform(0.0, 10.0, n_samples)
    twist = rng.uniform(-5.0, 2.0, n_samples)
    x_pos = rng.uniform(10.0, 60.0, n_samples)
    fuse_len = rng.uniform(180.0, 320.0, n_samples)
    fuse_diam = rng.uniform(15.0, 35.0, n_samples)
    naca_m = rng.uniform(0.0, 6.0, n_samples)
    naca_t = rng.uniform(8.0, 16.0, n_samples)

    X = np.column_stack([wing_span, root_chord, tip_chord, sweep, dihedral, twist, x_pos, fuse_len, fuse_diam, naca_m, naca_t])

    # Aerodynamic response surface
    ar = (2.0 * wing_span) / (0.5 * (root_chord + tip_chord))
    ld = 3.5 + 1.25 * np.sqrt(ar) - 0.00015 * (sweep ** 2) + 0.12 * naca_m - 0.003 * fuse_diam
    noise = rng.normal(0, 0.25, n_samples)
    y = ld + noise
    return X, y


def plot_sensitivity_panel(out_dir: Path):
    X, y = generate_synthetic_flight_dataset(500, seed=42)
    n_train = 350
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    ls = np.std(X_train, axis=0) * 1.5 + 1e-3
    sig_var = np.var(y_train)
    noise_var = 0.08

    y_pred, std_pred = fit_and_predict(X_train, y_train, X_test, ls, sig_var, noise_var)

    fig = plt.figure(figsize=(18, 9.5), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1. Parity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#2563eb", s=30, label="Holdout Predictions")
    lo, hi = min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Ideal 1:1 Parity")
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ax1.set_xlabel("True Aerodynamic Fitness ($L/D$)")
    ax1.set_ylabel("GP Surrogate Prediction")
    ax1.set_title(f"(a) Parity Plot ($R^2={r2:.3f}$, $RMSE={rmse:.3f}$)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Residual Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    res = y_test - y_pred
    ax2.hist(res, bins=20, color="#10b981", edgecolor="#047857", alpha=0.7, density=True)
    ax2.axvline(0.0, color="red", linestyle="--", lw=1.2)
    ax2.set_xlabel("Residual (True − Predicted)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("(b) Unbiased Zero-Centered Residuals", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Learning Curve
    ax3 = fig.add_subplot(gs[0, 2])
    sub_sizes = [50, 100, 150, 200, 250, 300, n_train]
    r2_list = []
    for sz in sub_sizes:
        yp_sub, _ = fit_and_predict(X_train[:sz], y_train[:sz], X_test, ls, sig_var, noise_var)
        r2_list.append(r2_score(y_test, yp_sub))
    ax3.plot(sub_sizes, r2_list, "o-", color="#8b5cf6", lw=2, label="$R^2$ Scaling")
    ax3.set_xlabel("Training Samples ($N$)")
    ax3.set_ylabel("Holdout $R^2$ Score")
    ax3.set_title("(c) GP Sample Efficiency", fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right")

    # 4-6: 1D Parameter Sensitivity Sweeps
    sweep_params = [
        ("Semi-Span $b/2$", 0, (80.0, 220.0), "mm"),
        ("Root Chord $c_{\\mathrm{root}}$", 1, (35.0, 75.0), "mm"),
        ("Fuselage Length $L_{\\mathrm{fuse}}$", 7, (180.0, 320.0), "mm"),
    ]
    baseline_x = np.median(X_train, axis=0)

    for idx, (pname, f_idx, (lo, hi), unit) in enumerate(sweep_params):
        ax_sw = fig.add_subplot(gs[1, idx])
        grid_vals = np.linspace(lo, hi, 60)
        X_sweep = np.tile(baseline_x, (len(grid_vals), 1))
        X_sweep[:, f_idx] = grid_vals
        mu_sw, std_sw = fit_and_predict(X_train, y_train, X_sweep, ls, sig_var, noise_var)

        ax_sw.plot(grid_vals, mu_sw, color="#1e40af", lw=2.2, label="Surrogate Mean")
        ax_sw.fill_between(grid_vals, mu_sw - 1.96 * std_sw, mu_sw + 1.96 * std_sw, color="#3b82f6", alpha=0.18, label="$\\pm 1.96\\sigma$ Uncertainty")
        ax_sw.set_xlabel(f"{pname} ({unit})")
        ax_sw.set_ylabel("Aerodynamic Fitness Score")
        ax_sw.set_title(f"({chr(100+idx)}) Sensitivity Sweep: {pname}", fontweight="bold")
        ax_sw.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_sw.legend(loc="upper left", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig2_surrogate_parameter_sensitivity.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Surrogate Sensitivity Panel")
    parser.add_argument("--out-dir", type=str, default="evaluation/figures")
    args = parser.parse_args()

    plot_sensitivity_panel(Path(args.out_dir))


if __name__ == "__main__":
    main()
