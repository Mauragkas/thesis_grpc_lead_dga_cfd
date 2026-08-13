from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import aerosandbox as asb

# -----------------------------------------------------------------------------
# Baseline Geometry & Flight Parameters (OpenSCAD Defaults)
# -----------------------------------------------------------------------------
BASELINE = {
    "fuse_length": 250.0,
    "fuse_max_diam": 10.0,
    "nose_ratio": 0.25,
    "tail_ratio": 0.35,
    "wing_span": 140.0,
    "wing_root_lead": 55.0,
    "wing_tip_lead": 25.0,
    "wing_sweep": 12.0,
    "wing_dihedral": 4.0,
    "wing_twist": -3.0,
    "wing_x_pos": 75.0,
    "wing_z_pos": -2.0,
    "naca_m": 2,  # 2% camber
    "naca_p": 4,  # 40% camber pos
    "naca_t": 12, # 12% thickness
    "tail_x_pos": 210.0,
    "v_stab_height": 45.0,
    "v_stab_root": 35.0,
    "v_stab_tip": 18.0,
    "h_stab_span": 45.0,
    "h_stab_root": 28.0,
    "h_stab_tip": 15.0,
}

RHO_MATERIAL = 250.0  # kg/m^3 (Effective density for 20% infill print)
RHO_AIR = 1.225       # kg/m^3
G = 9.81              # m/s^2
V_CRUISE = 15.0       # m/s
CG_POS = [0.08, 0, 0] # Estimated Center of Gravity (m)

# -----------------------------------------------------------------------------
# Mass & Geometric Helper Functions
# -----------------------------------------------------------------------------
def wing_panel_volume(span, root_c, tip_c, t_ratio):
    return 0.685 * t_ratio * (span / 3.0) * (root_c**2 + root_c * tip_c + tip_c**2)

def calculate_total_mass_and_weight(p):
    l_nose = p["fuse_length"] * p["nose_ratio"]
    l_tail = p["fuse_length"] * p["tail_ratio"]
    l_mid = p["fuse_length"] - l_nose - l_tail
    r_max = p["fuse_max_diam"] / 2.0

    v_nose = (2.0 / 3.0) * np.pi * (r_max ** 2) * l_nose
    v_mid = np.pi * (r_max ** 2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0 / 3.0) * np.pi * l_tail * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)
    v_fuse = v_nose + v_mid + v_tail

    t_main = p["naca_t"] / 100.0
    v_main = 2.0 * wing_panel_volume(p["wing_span"], p["wing_root_lead"], p["wing_tip_lead"], t_main)
    v_h = 2.0 * wing_panel_volume(p["h_stab_span"], p["h_stab_root"], p["h_stab_tip"], 0.10)
    v_v = wing_panel_volume(p["v_stab_height"], p["v_stab_root"], p["v_stab_tip"], 0.10)

    v_total_m3 = (v_fuse + v_main + v_h + v_v) * 1e-9
    mass_kg = v_total_m3 * RHO_MATERIAL
    return mass_kg, mass_kg * G

# -----------------------------------------------------------------------------
# Aerodynamic Evaluation Kernel
# -----------------------------------------------------------------------------
def evaluate_configuration(p):
    mass_kg, weight_n = calculate_total_mass_and_weight(p)
    s_ref = (2.0 * p["wing_span"] * (p["wing_root_lead"] + p["wing_tip_lead"]) / 2.0) * 1e-6
    scale = 1e-3

    naca_str = f"naca{int(p['naca_m'])}{int(p['naca_p'])}{int(p['naca_t']):02d}"
    airfoil_main = asb.Airfoil(naca_str)
    airfoil_tail = asb.Airfoil("naca0010")

    main_wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], lead=p["wing_root_lead"] * scale, twist=0, airfoil=airfoil_main),
            asb.WingXSec(
                xyz_le=[
                    p["wing_span"] * scale * np.tan(np.radians(p["wing_sweep"])),
                    p["wing_span"] * scale,
                    p["wing_span"] * scale * np.tan(np.radians(p["wing_dihedral"])),
                ],
                lead=p["wing_tip_lead"] * scale,
                twist=p["wing_twist"],
                airfoil=airfoil_main,
            ),
        ],
    ).translate([p["wing_x_pos"] * scale, 0, p["wing_z_pos"] * scale])

    h_stab = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], lead=p["h_stab_root"] * scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["h_stab_root"] - p["h_stab_tip"]) * scale, p["h_stab_span"] * scale, 0],
                lead=p["h_stab_tip"] * scale,
                twist=0,
                airfoil=airfoil_tail,
            ),
        ],
    ).translate([p["tail_x_pos"] * scale, 0, 0])

    v_stab = asb.Wing(
        name="Vertical Stabilizer",
        symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], lead=p["v_stab_root"] * scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["v_stab_root"] - p["v_stab_tip"]) * scale, 0, p["v_stab_height"] * scale],
                lead=p["v_stab_tip"] * scale,
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

    alpha_sweep = np.linspace(-2.0, 10.0, 13)
    cl_list, cd_list, cm_list = [], [], []

    for alpha in alpha_sweep:
        op_point = asb.OperatingPoint(velocity=V_CRUISE, alpha=alpha)
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            spanwise_resolution=8,
            leadwise_resolution=3,
        )
        res = vlm.run()
        cl_list.append(res["CL"])
        cd_list.append(res["CD"])
        cm_list.append(res["Cm"])

    if cl_req < min(cl_list) or cl_req > max(cl_list):
        return {"ld": np.nan, "alpha_trim": np.nan, "cm_alpha": np.nan, "cl_req": cl_req}

    cd_trim = float(np.interp(cl_req, cl_list, cd_list))
    alpha_trim = float(np.interp(cl_req, cl_list, alpha_sweep))
    cm_trim = float(np.interp(cl_req, cl_list, cm_list))

    # Calculate static pitching stability slope dCm/dAlpha
    cm_alpha = float(np.gradient(cm_list, np.radians(alpha_sweep))[6])

    return {
        "ld": cl_req / cd_trim,
        "alpha_trim": alpha_trim,
        "cm_alpha": cm_alpha,
        "cl_req": cl_req,
        "mass_g": mass_kg * 1000,
    }

# -----------------------------------------------------------------------------
# Multi-Dimensional Parameter Sweep Pipeline (8 Parameters)
# -----------------------------------------------------------------------------
def run_extended_analysis():
    sweep_definitions = {
        "wing_span": ("Wing Semi-Span (mm)", np.linspace(80.0, 220.0, 12)),
        "wing_root_lead": ("Root Lead (mm)", np.linspace(35.0, 75.0, 12)),
        "wing_tip_lead": ("Tip Lead (mm)", np.linspace(10.0, 45.0, 12)),
        "wing_sweep": ("Sweep Angle (deg)", np.linspace(0.0, 25.0, 12)),
        "wing_dihedral": ("Dihedral Angle (deg)", np.linspace(0.0, 10.0, 12)),
        "wing_twist": ("Washout Twist (deg)", np.linspace(-6.0, 2.0, 12)),
        "wing_x_pos": ("Wing X-Position (mm)", np.linspace(55.0, 95.0, 12)),
        "naca_m": ("Airfoil Camber (%)", np.linspace(0.0, 5.0, 6)),
    }

    all_jobs = []
    job_indices = {}
    current_idx = 0

    for param_name, (_, val_range) in sweep_definitions.items():
        start = current_idx
        for val in val_range:
            p_config = BASELINE.copy()
            p_config[param_name] = val
            all_jobs.append(p_config)
            current_idx += 1
        job_indices[param_name] = (start, current_idx)

    print(f"Executing {len(all_jobs)} aerodynamic evaluations across 8 dimensions...")

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_configuration, all_jobs))

    # Plot 2x4 Grid of Parameter Studies
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    baseline_eval = evaluate_configuration(BASELINE)
    base_ld = baseline_eval["ld"]

    print("\n" + "=" * 65)
    print(f"{'PARAMETER':<20} | {'BASE VAL':<10} | {'SENSITIVITY (d(L/D)/dP)':<25}")
    print("=" * 65)

    for idx, (param_name, (label, val_range)) in enumerate(sweep_definitions.items()):
        start, end = job_indices[param_name]
        param_results = results[start:end]

        ld_vals = [r["ld"] for r in param_results]
        alpha_vals = [r["alpha_trim"] for r in param_results]

        ax = axes[idx]
        ax.plot(val_range, ld_vals, "b-o", linewidth=1.8, label="L/D Ratio")
        ax.set_xlabel(label)
        ax.set_ylabel("Trim L/D", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Plot Trim Alpha on secondary axis
        ax2 = ax.twinx()
        ax2.plot(val_range, alpha_vals, "r--", alpha=0.7, label=r"$\alpha_{trim}$")
        ax2.set_ylabel(r"Trim $\alpha$ (°)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        # Compute numerical sensitivity near baseline
        valid_ld = [v for v in ld_vals if not np.isnan(v)]
        if len(valid_ld) > 1:
            grad = np.gradient(ld_vals, val_range)
            mid_grad = grad[len(grad) // 2]
            print(f"{param_name:<20} | {BASELINE[param_name]:<10.1f} | {mid_grad:<+25.4f}")

    print("=" * 65)
    plt.suptitle(f"Extended 8-Parameter Sensitivity Study (Baseline L/D = {base_ld:.2f})", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_extended_analysis()
