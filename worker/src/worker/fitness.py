from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

try:
    from .aero import AeroEvaluator, AeroResult
    from .config import BASELINE, GENE_BOUNDS, GeneBound, WorkerConfig
    from .geometry import fuselage_volume_mm3, decode_genes, TailVolumeTargets
    from .mission.trajectory import MissionConfig, MissionOutcome, MissionSimulator
    from .powertrain.battery import Battery, Capacity
    from .powertrain.motor import Motor, MotorKV, MotorResistance
    from .powertrain.propeller import Propeller, PropellerDimension
    from .powertrain.powertrain import Powertrain
except (ImportError, ValueError):
    from aero import AeroEvaluator, AeroResult
    from config import BASELINE, GENE_BOUNDS, GeneBound, WorkerConfig
    from geometry import fuselage_volume_mm3, decode_genes, TailVolumeTargets
    from mission.trajectory import MissionConfig, MissionOutcome, MissionSimulator
    from powertrain.battery import Battery, Capacity
    from powertrain.motor import Motor, MotorKV, MotorResistance
    from powertrain.propeller import Propeller, PropellerDimension
    from powertrain.powertrain import Powertrain

REJECT_FITNESS = -1e9


@dataclass(frozen=True)
class EvaluationOutcome:
    fitness: float
    fuselage_volume_mm3: float
    aero: AeroResult | None
    rejected: bool
    mission: MissionOutcome | None = None


class FitnessEvaluator:
    def __init__(
        self,
        aero_evaluator: AeroEvaluator,
        config: WorkerConfig,
        baseline: dict[str, float] = BASELINE,
        bounds: Sequence[GeneBound] = GENE_BOUNDS,
        logger: logging.Logger | None = None,
        mission_simulator: MissionSimulator | None = None,
    ):
        self._aero_evaluator = aero_evaluator
        self._config = config
        self._baseline = baseline
        self._bounds = bounds
        self._logger = logger or logging.getLogger("worker.fitness")
        self._mission_simulator = mission_simulator or self._create_default_simulator(config)
        self._tail_targets = (
            TailVolumeTargets(vh=config.target_vh, vv=config.target_vv)
            if getattr(config, "enable_tail_sizing", False)
            else None
        )

    @staticmethod
    def _create_default_simulator(config: WorkerConfig) -> MissionSimulator:
        battery = Battery(
            cell_count=config.battery_cells,
            capacity=Capacity(config.battery_capacity_ah),
            internal_resistance_ohms=0.03,
        )
        motor = Motor(
            kv=MotorKV(config.motor_kv),
            resistance=MotorResistance(config.motor_resistance_ohms),
            no_load_current_a=0.8,
            max_current_a=config.motor_max_current_a,
        )
        propeller = Propeller(
            dimensions=PropellerDimension(
                diameter_m=config.prop_diameter_m,
                pitch_m=config.prop_pitch_m,
            ),
            rho_air=config.rho_air,
        )
        powertrain = Powertrain(battery=battery, motor=motor, propeller=propeller)
        mission_cfg = MissionConfig(
            cruise_altitude_m=20.0,
            target_flight_time_s=config.target_flight_time_s,
            rolling_friction_coeff=0.04,
            max_takeoff_distance_m=config.max_takeoff_distance_m,
            rho_air=config.rho_air,
            g=config.g,
        )
        return MissionSimulator(powertrain=powertrain, config=mission_cfg)

    def evaluate_genes(self, genes: Sequence[float]) -> EvaluationOutcome:
        params = decode_genes(genes, self._baseline, self._bounds, tail_targets=self._tail_targets)
        self._logger.debug("decoded genes=%s -> params=%s", list(genes), params)
        return self.evaluate_params(params)

    def evaluate_params(self, params: dict[str, float]) -> EvaluationOutcome:
        v_fuse = fuselage_volume_mm3(params)

        if v_fuse < self._config.v_min_fuse_mm3:
            self._logger.debug(
                "reject v_fuse=%.0f < v_min=%.0f",
                v_fuse,
                self._config.v_min_fuse_mm3,
            )
            return self._rejected(v_fuse)

        aero = self._aero_evaluator.evaluate(params)
        if aero is None:
            self._logger.debug(
                "reject aero=None (trim out of sweep) v_fuse=%.0f", v_fuse
            )
            return self._rejected(v_fuse)

        alpha_penalty = self._config.w_alpha * abs(aero.alpha_trim)
        stability_penalty = self._config.w_stability * max(0.0, aero.cm_alpha) ** 2
        moment_penalty = getattr(self._config, "w_moment", 30.0) * (aero.cm_trim**2)
        base_fitness = aero.ld - alpha_penalty - stability_penalty - moment_penalty

        mission_outcome: MissionOutcome | None = None
        mission_penalty = 0.0

        if self._config.enable_mission_sim and aero.polar is not None:
            mission_outcome = self._mission_simulator.simulate(
                polar=aero.polar,
                mass_kg=aero.mass_kg,
                wing_area_m2=aero.s_ref,
            )
            if not mission_outcome.completed:
                self._logger.debug(
                    "reject mission failed: %s (takeoff_dist=%.1fm)",
                    mission_outcome.failure_reason,
                    mission_outcome.takeoff_distance_m,
                )
                return self._rejected(v_fuse, mission_outcome=mission_outcome)

            to_ratio = mission_outcome.takeoff_distance_m / max(self._config.max_takeoff_distance_m, 1.0)
            mission_penalty = self._config.w_takeoff_penalty * (to_ratio**2)
            energy_penalty = self._config.w_energy_penalty * (mission_outcome.energy_consumed_wh / 5.0)
            fitness = base_fitness - mission_penalty - energy_penalty
        else:
            fitness = base_fitness

        self._logger.debug(
            "fit=%.4f ld=%.3f alpha=%.2f cm_alpha=%.3f cm_trim=%.3f v_fuse=%.0f mission_pen=%.3f",
            fitness,
            aero.ld,
            aero.alpha_trim,
            aero.cm_alpha,
            aero.cm_trim,
            v_fuse,
            mission_penalty,
        )

        return EvaluationOutcome(
            fitness=fitness,
            fuselage_volume_mm3=v_fuse,
            aero=aero,
            rejected=False,
            mission=mission_outcome,
        )

    def _rejected(
        self,
        v_fuse: float,
        mission_outcome: MissionOutcome | None = None,
    ) -> EvaluationOutcome:
        return EvaluationOutcome(
            fitness=REJECT_FITNESS,
            fuselage_volume_mm3=v_fuse,
            aero=None,
            rejected=True,
            mission=mission_outcome,
        )
