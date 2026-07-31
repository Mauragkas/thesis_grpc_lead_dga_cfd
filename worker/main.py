import asyncio
import logging
import os
import math
import numpy as np
import aerosandbox as asb

import grpc

import eval_pb2
import eval_pb2_grpc

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("worker")

WORKER_ID = os.getenv("WORKER_ID") or os.getenv("HOSTNAME", "unknown")
_semaphore = asyncio.Semaphore(1)

# ---------------------------------------------------------------------------
# Physical constants (from grid search)
# ---------------------------------------------------------------------------
RHO_MATERIAL = 250.0
RHO_AIR = 1.225
G = 9.81
V_CRUISE = 15.0
CG_POS = [0.08, 0, 0]

# Hard constraint: minimum fuselage volume (mm^3). Set via env, default ~baseline.
V_MIN_FUSE_MM3 = float(os.getenv("V_MIN_FUSE_MM3", "20000.0"))

BASELINE = {
    "fuse_length": 250.0, "fuse_max_diam": 10.0,
    "nose_ratio": 0.25, "tail_ratio": 0.35,
    "wing_span": 140.0, "wing_root_chord": 55.0, "wing_tip_chord": 25.0,
    "wing_sweep": 12.0, "wing_dihedral": 4.0, "wing_twist": -3.0,
    "wing_x_pos": 75.0, "wing_z_pos": -2.0,
    "naca_m": 2, "naca_p": 4, "naca_t": 12,
    "tail_x_pos": 210.0, "v_stab_height": 45.0,
    "v_stab_root": 35.0, "v_stab_tip": 18.0,
    "h_stab_span": 45.0, "h_stab_root": 28.0, "h_stab_tip": 15.0,
}

# (name, low, high) — order defines gene index
GENE_BOUNDS = [
    ("wing_span",       80.0, 220.0),
    ("wing_root_chord", 35.0,  75.0),
    ("wing_tip_chord",  10.0,  45.0),
    ("wing_sweep",       0.0,  25.0),
    ("wing_dihedral",    0.0,  10.0),
    ("wing_twist",      -6.0,   2.0),
    ("wing_x_pos",      55.0,  95.0),
    ("naca_m",           0.0,   5.0),
    ("fuse_length",    200.0, 350.0),
    ("fuse_max_diam",    8.0,  16.0),
]
GENES_LEN = len(GENE_BOUNDS)

# Fitness weights
W_ALPHA = 0.5
W_STABILITY = 20.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def wing_panel_volume(span, root_c, tip_c, t_ratio):
    return 0.685 * t_ratio * (span / 3.0) * (root_c**2 + root_c*tip_c + tip_c**2)


def calculate_fuselage_volume_mm3(p):
    l_nose = p["fuse_length"] * p["nose_ratio"]
    l_tail = p["fuse_length"] * p["tail_ratio"]
    l_mid = p["fuse_length"] - l_nose - l_tail
    r_max = p["fuse_max_diam"] / 2.0
    v_nose = (2.0/3.0) * np.pi * (r_max**2) * l_nose
    v_mid = np.pi * (r_max**2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0/3.0) * np.pi * l_tail * (r_max**2 + r_max*r_tail_tip + r_tail_tip**2)
    return v_nose + v_mid + v_tail


def calculate_total_mass_and_weight(p):
    mass_kg, _ = _mass_and_weight(p)
    return mass_kg, mass_kg * G


def _mass_and_weight(p):
    l_nose = p["fuse_length"] * p["nose_ratio"]
    l_tail = p["fuse_length"] * p["tail_ratio"]
    l_mid = p["fuse_length"] - l_nose - l_tail
    r_max = p["fuse_max_diam"] / 2.0
    v_nose = (2.0/3.0) * np.pi * (r_max**2) * l_nose
    v_mid = np.pi * (r_max**2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0/3.0) * np.pi * l_tail * (r_max**2 + r_max*r_tail_tip + r_tail_tip**2)
    v_fuse = v_nose + v_mid + v_tail

    t_main = p["naca_t"] / 100.0
    v_main = 2.0 * wing_panel_volume(p["wing_span"], p["wing_root_chord"], p["wing_tip_chord"], t_main)
    v_h = 2.0 * wing_panel_volume(p["h_stab_span"], p["h_stab_root"], p["h_stab_tip"], 0.10)
    v_v = wing_panel_volume(p["v_stab_height"], p["v_stab_root"], p["v_stab_tip"], 0.10)

    v_total_m3 = (v_fuse + v_main + v_h + v_v) * 1e-9
    mass_kg = v_total_m3 * RHO_MATERIAL
    return mass_kg, mass_kg * G


# ---------------------------------------------------------------------------
# Aero evaluation (ported from test2.py)
# ---------------------------------------------------------------------------
def evaluate_aero(p):
    mass_kg, weight_n = calculate_total_mass_and_weight(p)
    s_ref = (2.0 * p["wing_span"] * (p["wing_root_chord"] + p["wing_tip_chord"]) / 2.0) * 1e-6
    scale = 1e-3

    naca_str = f"naca{int(round(p['naca_m']))}{int(p['naca_p'])}{int(p['naca_t']):02d}"
    airfoil_main = asb.Airfoil(naca_str)
    airfoil_tail = asb.Airfoil("naca0010")

    main_wing = asb.Wing(
        name="Main Wing", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0,0,0], chord=p["wing_root_chord"]*scale, twist=0, airfoil=airfoil_main),
            asb.WingXSec(
                xyz_le=[
                    p["wing_span"]*scale*np.tan(np.radians(p["wing_sweep"])),
                    p["wing_span"]*scale,
                    p["wing_span"]*scale*np.tan(np.radians(p["wing_dihedral"])),
                ],
                chord=p["wing_tip_chord"]*scale, twist=p["wing_twist"], airfoil=airfoil_main,
            ),
        ],
    ).translate([p["wing_x_pos"]*scale, 0, p["wing_z_pos"]*scale])

    h_stab = asb.Wing(
        name="Horizontal Stabilizer", symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0,0,0], chord=p["h_stab_root"]*scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["h_stab_root"]-p["h_stab_tip"])*scale, p["h_stab_span"]*scale, 0],
                chord=p["h_stab_tip"]*scale, twist=0, airfoil=airfoil_tail,
            ),
        ],
    ).translate([p["tail_x_pos"]*scale, 0, 0])

    v_stab = asb.Wing(
        name="Vertical Stabilizer", symmetric=False,
        xsecs=[
            asb.WingXSec(xyz_le=[0,0,0], chord=p["v_stab_root"]*scale, twist=0, airfoil=airfoil_tail),
            asb.WingXSec(
                xyz_le=[(p["v_stab_root"]-p["v_stab_tip"])*scale, 0, p["v_stab_height"]*scale],
                chord=p["v_stab_tip"]*scale, twist=0, airfoil=airfoil_tail,
            ),
        ],
    ).translate([p["tail_x_pos"]*scale, 0, 0])

    airplane = asb.Airplane(name="Parametric Model", xyz_ref=CG_POS,
                            wings=[main_wing, h_stab, v_stab])

    cl_req = (2.0 * weight_n) / (RHO_AIR * (V_CRUISE**2) * s_ref)

    alpha_sweep = np.linspace(-2.0, 10.0, 13)
    cl_list, cd_list, cm_list = [], [], []
    for alpha in alpha_sweep:
        op_point = asb.OperatingPoint(velocity=V_CRUISE, alpha=alpha)
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point,
                                      spanwise_resolution=8, chordwise_resolution=3)
        res = vlm.run()
        cl_list.append(res["CL"]); cd_list.append(res["CD"]); cm_list.append(res["Cm"])

    if cl_req < min(cl_list) or cl_req > max(cl_list):
        return None

    cd_trim = float(np.interp(cl_req, cl_list, cd_list))
    alpha_trim = float(np.interp(cl_req, cl_list, alpha_sweep))
    cm_trim = float(np.interp(cl_req, cl_list, cm_list))
    cm_alpha = float(np.gradient(cm_list, np.radians(alpha_sweep))[6])
    return {"ld": cl_req/cd_trim, "alpha_trim": alpha_trim, "cm_alpha": cm_alpha}


# ---------------------------------------------------------------------------
# Gene decoding + fitness
# ---------------------------------------------------------------------------
def decode_genes(genes):
    p = BASELINE.copy()
    for i, (name, lo, hi) in enumerate(GENE_BOUNDS):
        u = max(0.0, min(1.0, genes[i]))  # clip to [0,1]
        p[name] = lo + (hi - lo) * u
    return p


def fitness_function(p):
    # --- HARD volume constraint (death penalty) ---
    v_fuse = calculate_fuselage_volume_mm3(p)
    if v_fuse < V_MIN_FUSE_MM3:
        return -1e9, v_fuse, None

    result = evaluate_aero(p)
    if result is None:
        return -1e9, v_fuse, None

    alpha_penalty = W_ALPHA * abs(result["alpha_trim"])
    stability_penalty = W_STABILITY * max(0.0, result["cm_alpha"]) ** 2
    fitness = result["ld"] - alpha_penalty - stability_penalty
    return fitness, v_fuse, result


# ---------------------------------------------------------------------------
# gRPC service
# ---------------------------------------------------------------------------
class EvaluatorServicer(eval_pb2_grpc.EvaluatorServicer):
    async def Evaluate(self, request, context):
        async with _semaphore:
            genes = list(request.genes)
            if len(genes) != GENES_LEN:
                logger.warning(f"Worker '{WORKER_ID}' got {len(genes)} genes, expected {GENES_LEN}")
                return eval_pb2.EvaluationResult(worker_id=WORKER_ID, fitness=-1e9)

            p = decode_genes(genes)
            # Run CPU-heavy aero eval in a thread so we don't block the event loop
            loop = asyncio.get_running_loop()
            fitness, v_fuse, res = await loop.run_in_executor(None, fitness_function, p)

            logger.info(f"Worker '{WORKER_ID}' | V_fuse={v_fuse:.0f} mm^3 "
                        f"(min {V_MIN_FUSE_MM3:.0f}) | fit={fitness:.3f}")
            return eval_pb2.EvaluationResult(worker_id=WORKER_ID, fitness=fitness)

    async def EvaluateBatch(self, request, context):
        async with _semaphore:
            loop = asyncio.get_running_loop()
            results = []
            for ind in request.individuals:
                genes = list(ind.genes)
                if len(genes) != GENES_LEN:
                    results.append(eval_pb2.EvaluationResult(worker_id=WORKER_ID, fitness=-1e9))
                    continue
                p = decode_genes(genes)
                fitness, v_fuse, _ = await loop.run_in_executor(None, fitness_function, p)
                results.append(eval_pb2.EvaluationResult(worker_id=WORKER_ID, fitness=fitness))
            return eval_pb2.BatchResponse(results=results)


async def serve():
    server = grpc.aio.server()
    eval_pb2_grpc.add_EvaluatorServicer_to_server(EvaluatorServicer(), server)
    server.add_insecure_port("[::]:50051")
    logger.info(f"Worker '{WORKER_ID}' running aero evaluator on :50051 "
                f"(V_MIN_FUSE_MM3={V_MIN_FUSE_MM3})")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
