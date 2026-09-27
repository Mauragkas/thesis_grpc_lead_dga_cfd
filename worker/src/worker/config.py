from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class GeneBound:
    name: str
    low: float
    high: float


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str
    v_min_fuse_mm3: float
    rho_material: float = 250.0
    rho_air: float = 1.225
    g: float = 9.81
    v_cruise: float = 15.0
    cg_pos: tuple[float, float, float] = (0.08, 0.0, 0.0)
    w_alpha: float = 0.5
    w_stability: float = 20.0
    w_moment: float = 30.0
    m_avionics_kg: float = 0.080
    areal_density_fuse: float = 1.20
    areal_density_wing: float = 1.00
    semaphore_size: int = 1
    # Powertrain & dynamic mission configuration
    enable_mission_sim: bool = True
    battery_cells: int = 3
    battery_capacity_ah: float = 2.2
    motor_kv: float = 1000.0
    motor_resistance_ohms: float = 0.06
    motor_max_current_a: float = 40.0
    prop_diameter_m: float = 0.254
    prop_pitch_m: float = 0.1524
    max_takeoff_distance_m: float = 30.0
    target_flight_time_s: float = 30.0
    w_takeoff_penalty: float = 5.0
    w_energy_penalty: float = 0.5
    # Tail sizing volume coefficient targets (Stan Hall method)
    enable_tail_sizing: bool = False
    target_vh: float = 0.50
    target_vv: float = 0.04


BASELINE: dict[str, float] = {
    "fuse_length": 250.0,
    "fuse_max_diam": 20.0,
    "nose_ratio": 0.25,
    "tail_ratio": 0.35,
    "wing_span": 140.0,
    "wing_root_chord": 55.0,
    "wing_tip_chord": 25.0,
    "wing_sweep": 12.0,
    "wing_dihedral": 2.5,
    "wing_twist": -2.0,
    "wing_x_pos": 75.0,
    "wing_z_pos": -2.0,
    "naca_m": 2,
    "naca_p": 4,
    "naca_t": 12,
    "tail_x_pos": 205.0,
    "v_stab_height": 45.0,
    "v_stab_root": 35.0,
    "v_stab_tip": 18.0,
    "h_stab_span": 45.0,
    "h_stab_root": 28.0,
    "h_stab_tip": 15.0,
}

GENE_BOUNDS: tuple[GeneBound, ...] = (
    GeneBound("wing_span", 80.0, 220.0),
    GeneBound("wing_root_chord", 35.0, 75.0),
    GeneBound("wing_tip_chord", 10.0, 45.0),
    GeneBound("wing_sweep", 0.0, 25.0),
    GeneBound("wing_dihedral", 0.0, 4.0),
    GeneBound("wing_twist", -4.0, 0.0),
    GeneBound("wing_x_pos", 55.0, 95.0),
    GeneBound("naca_m", 0.0, 5.0),
    GeneBound("fuse_length", 200.0, 350.0),
    GeneBound("fuse_max_diam", 14.0, 32.0),
)


def load_config() -> WorkerConfig:
    return WorkerConfig(
        worker_id=os.getenv("WORKER_ID") or os.getenv("HOSTNAME", "unknown"),
        v_min_fuse_mm3=float(os.getenv("V_MIN_FUSE_MM3", "20000.0")),
        semaphore_size=int(os.getenv("WORKER_CONCURRENCY", "1")),
        enable_mission_sim=os.getenv("ENABLE_MISSION_SIM", "true").lower() in ("true", "1", "yes"),
        max_takeoff_distance_m=float(os.getenv("MAX_TAKEOFF_DISTANCE_M", "30.0")),
        target_flight_time_s=float(os.getenv("TARGET_FLIGHT_TIME_S", "30.0")),
        enable_tail_sizing=os.getenv("ENABLE_TAIL_SIZING", "false").lower() in ("true", "1", "yes"),
        target_vh=float(os.getenv("TARGET_VH", "0.50")),
        target_vv=float(os.getenv("TARGET_VV", "0.04")),
    )
