#!/usr/bin/env python3
"""
Gaussian Process Surrogate visualization script.
Generates parity plots, residual histograms, and 1D parameter sensitivity curves.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def load_data(filepath: Path):
    with open(filepath, "r") as f:
        data = json.load(f)
    valid = [r for r in data if r.get("fitness", -99) > -15.0]
    X = np.array([[r["config"][k] for k in FEATURE_NAMES] for r in valid])
    y = np.array([r["fitness"] for r in valid])
    return X, y

def matern52_kernel(x1, x2, length_scales, signal_var):
    diff = (x1[:, None, :] - x2[None, :, :]) / length_scales
    d2 = np.sum(diff**2, axis=-1)
    dist = np.sqrt(np.maximum(d2, 0.0))
    sqrt5_r = np.sqrt(5.0) * dist
    return signal_var * (1.0 + sqrt5_r + (5.0 / 3.0) * d2) * np.exp(-sqrt5_r)

def fit_and_predict(X_train, y_train, X_test, ls, sig_var, noise_var):
    # Scale X and y
    mean_X, std_X = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std_X[std_X < 1e-8] = 1.0
    mean_y, std_y = np.mean(y_train), np.std(y_train)
    
    X_tr_s = (X_train - mean_X) / std_X
    y_tr_s = (y_train - mean_y) / std_y
    X_te_s = (X_test - mean_X) / std_X
    
    K = matern52_kernel(X_tr_s, X_tr_s, ls, sig_var) + noise_var * np.eye(len(X_tr_s))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr_s))
    
    K_star = matern52_kernel(X_te_s, X_tr_s, ls, sig_var)
    mu_s = K_star @ alpha
    
    v = np.linalg.solve(L, K_star.T)
    var_s = np.maximum(sig_var - np.sum(v**2, axis=0), 0.0)
    
    mu_orig = mu_s * std_y + mean_y
    std_orig = np.sqrt(var_s) * std_y
    return mu_orig, std_orig

def main():
    root = Path(__file__).parent.parent.parent
    data_file = root / "tests" / "configs_and_scores.json"
    if not data_file.exists():
        data_file = Path("tests/configs_and_scores.json")
    
    if not data_file.exists():
        print(f"Dataset not found at {data_file}")
        return

    X, y = load_data(data_file)
    n = len(X)
    
    # 60-20-20 partition
    np.random.seed(42)
    indices = np.random.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    
    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]
    
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    # Optimized hyperparameters from gp_node
    ls = np.array([2.5, 1.8, 1.2, 3.0, 2.8, 1.5, 2.2, 3.5, 2.0, 1.4, 2.1])
    sig_var = 2.60
    noise_var = 0.0031

    y_pred, y_std = fit_and_predict(X_train, y_train, X_test, ls, sig_var, noise_var)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1. Parity Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(y_test, y_pred, yerr=1.96 * y_std, fmt="o", color="#2563eb", ecolor="#93c5fd",
                 alpha=0.75, capsize=2, label=r"GP Prediction $\pm 1.96\sigma$")
    min_v = min(np.min(y_test), np.min(y_pred)) - 0.5
    max_v = max(np.max(y_test), np.max(y_pred)) + 0.5
    ax1.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, label="1:1 Parity")
    ax1.set_xlabel("Ground Truth VLM Fitness", fontsize=11)
    ax1.set_ylabel("GP Predicted Mean", fontsize=11)
    ax1.set_title(f"Holdout Test Parity ($R^2 = {r2:.4f}$)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # 2. Residual Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=16, color="#059669", edgecolor="#064e3b", alpha=0.7, density=True)
    ax2.axvline(0.0, color="red", linestyle="--", lw=1.2)
    ax2.set_xlabel("Residual Error (True - Pred)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"Residual Distribution (RMSE = {rmse:.4f}, MAE = {mae:.4f})", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Learning / Scaling Curve
    ax3 = fig.add_subplot(gs[0, 2])
    train_sizes = [50, 100, 150, 200, 250, 300, len(X_train)]
    r2_curve, rmse_curve = [], []
    for sz in train_sizes:
        yp_sub, _ = fit_and_predict(X_train[:sz], y_train[:sz], X_test, ls, sig_var, noise_var)
        r2_curve.append(r2_score(y_test, yp_sub))
        rmse_curve.append(np.sqrt(mean_squared_error(y_test, yp_sub)))
    
    ax3.plot(train_sizes, r2_curve, "o-", color="#7c3aed", lw=2, label=r"$R^2$ Score")
    ax3.set_xlabel("Training Set Size (N)", fontsize=11)
    ax3.set_ylabel(r"Holdout $R^2$ Score", fontsize=11)
    ax3.set_title("GP Surrogate Learning Curve", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=10)

    # 4-6. 1D Sensitivity Parameter Sweeps (wing_span, wing_root_chord, fuse_length)
    sweep_params = [
        ("wing_span", 0, (80.0, 220.0)),
        ("wing_root_chord", 1, (35.0, 75.0)),
        ("fuse_length", 7, (180.0, 320.0)),
    ]
    baseline_x = np.median(X_train, axis=0)

    for idx, (pname, f_idx, (lo, hi)) in enumerate(sweep_params):
        ax_sw = fig.add_subplot(gs[1, idx])
        grid_vals = np.linspace(lo, hi, 50)
        X_sweep = np.tile(baseline_x, (len(grid_vals), 1))
        X_sweep[:, f_idx] = grid_vals
        
        mu_sw, std_sw = fit_and_predict(X_train, y_train, X_sweep, ls, sig_var, noise_var)
        
        ax_sw.plot(grid_vals, mu_sw, "b-", lw=2, label="GP Mean Prediction")
        ax_sw.fill_between(grid_vals, mu_sw - 1.96 * std_sw, mu_sw + 1.96 * std_sw, color="blue", alpha=0.15, label=r"$\pm 1.96\sigma$ Confidence")
        ax_sw.set_xlabel(f"{pname} (mm)", fontsize=11)
        ax_sw.set_ylabel("Predicted Fitness Score", fontsize=11)
        ax_sw.set_title(f"Sensitivity: {pname}", fontsize=12, fontweight="bold")
        ax_sw.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_sw.legend(fontsize=9)

    out_path = Path(__file__).parent / "gp_surrogate_evaluation.png"
    plt.savefig(out_path, dpi=200)
    print(f"GP visualization panel successfully generated at: {out_path}")

if __name__ == "__main__":
    main()
