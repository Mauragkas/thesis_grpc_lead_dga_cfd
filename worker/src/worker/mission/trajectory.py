from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .polar import AeroPolar
from worker.powertrain.battery import StateOfCharge
from worker.powertrain.powertrain import ThrustProducer


@dataclass(frozen=True)
class MissionConfig:
    """Mission profile parameters."""

    cruise_altitude_m: float = 20.0
    target_flight_time_s: float = 30.0
    rolling_friction_coeff: float = 0.04
    max_takeoff_distance_m: float = 40.0
    rho_air: float = 1.225
    g: float = 9.81


@dataclass(frozen=True)
class MissionOutcome:
    """Results of dynamic mission simulation."""

    completed: bool
    takeoff_distance_m: float
    total_distance_m: float
    energy_consumed_wh: float
    final_soc: float
    max_current_a: float
    failure_reason: str | None = None


class MissionSimulator:
    """Multi-phase time-domain trajectory simulation."""

    def __init__(
        self,
        powertrain: ThrustProducer,
        config: MissionConfig = MissionConfig(),
    ) -> None:
        self.powertrain = powertrain
        self.config = config

    def simulate(
        self,
        polar: AeroPolar,
        mass_kg: float,
        wing_area_m2: float,
    ) -> MissionOutcome:
        """Simulate ground roll, climb, and cruise phases."""
        weight_n = mass_kg * self.config.g
        v_stall = polar.stall_speed(weight_n, wing_area_m2, self.config.rho_air)
        v_rotate = 1.15 * v_stall

        # Phase 0: Ground roll simulation
        soc = StateOfCharge(1.0)
        roll_outcome = self._simulate_ground_roll(
            polar=polar,
            mass_kg=mass_kg,
            wing_area_m2=wing_area_m2,
            v_rotate=v_rotate,
            current_soc=soc,
        )

        if not roll_outcome["completed"]:
            return MissionOutcome(
                completed=False,
                takeoff_distance_m=roll_outcome["distance_m"],
                total_distance_m=roll_outcome["distance_m"],
                energy_consumed_wh=roll_outcome["energy_wh"],
                final_soc=roll_outcome["final_soc"].value,
                max_current_a=roll_outcome["max_current_a"],
                failure_reason=roll_outcome.get("reason", "takeoff_failed"),
            )

        # Phase 1 & 2: Air flight (Climb & Cruise)
        air_outcome = self._simulate_air_flight(
            polar=polar,
            mass_kg=mass_kg,
            wing_area_m2=wing_area_m2,
            start_distance_m=roll_outcome["distance_m"],
            start_time_s=roll_outcome["time_s"],
            start_speed_m_s=v_rotate,
            current_soc=roll_outcome["final_soc"],
            max_current_so_far=roll_outcome["max_current_a"],
            energy_so_far_wh=roll_outcome["energy_wh"],
        )

        return air_outcome

    def _simulate_ground_roll(
        self,
        polar: AeroPolar,
        mass_kg: float,
        wing_area_m2: float,
        v_rotate: float,
        current_soc: StateOfCharge,
    ) -> dict:
        """Simulate ground roll until v_rotate or max takeoff distance exceeded."""
        weight_n = mass_kg * self.config.g
        ground_aoa_deg = 0.0
        cl_ground = polar.lift_coefficient(ground_aoa_deg)
        cd_ground = polar.drag_coefficient(ground_aoa_deg)

        def ground_roll_ode(t: float, y: np.ndarray) -> list[float]:
            # y = [x, v, energy_j, soc_val]
            x, v, _, soc_val = y
            soc_obj = StateOfCharge(max(0.0, min(1.0, soc_val)))
            pt = self.powertrain.evaluate(airspeed_m_s=max(0.0, v), throttle=1.0, soc=soc_obj)

            q_inf = 0.5 * self.config.rho_air * (v**2) * wing_area_m2
            lift = q_inf * cl_ground
            drag = q_inf * cd_ground

            normal_force = max(0.0, weight_n - lift)
            f_friction = self.config.rolling_friction_coeff * normal_force

            net_force = pt.thrust_n - drag - f_friction
            a = net_force / mass_kg

            p_electric = pt.electric_power_w
            # d(soc)/dt = -current / (3600 * capacity)
            current = pt.current_a
            # Approximate capacity depletion rate (assuming 2.2Ah standard if not queried)
            dsoc_dt = -current / (3600.0 * 2.2)

            return [v, a, p_electric, dsoc_dt]

        def rotate_event(t: float, y: np.ndarray) -> float:
            return y[1] - v_rotate

        rotate_event.terminal = True
        rotate_event.direction = 1

        def max_dist_event(t: float, y: np.ndarray) -> float:
            return self.config.max_takeoff_distance_m - y[0]

        max_dist_event.terminal = True
        max_dist_event.direction = -1

        y0 = [0.0, 0.5, 0.0, current_soc.value]
        sol = solve_ivp(
            ground_roll_ode,
            t_span=(0.0, 30.0),
            y0=y0,
            events=[rotate_event, max_dist_event],
            max_step=0.1,
        )

        final_x = float(sol.y[0, -1])
        final_v = float(sol.y[1, -1])
        final_energy_wh = float(sol.y[2, -1]) / 3600.0
        final_soc = StateOfCharge(max(0.0, min(1.0, float(sol.y[3, -1]))))
        final_t = float(sol.t[-1])

        pt_static = self.powertrain.evaluate(airspeed_m_s=0.0, throttle=1.0, soc=current_soc)
        max_current = pt_static.current_a

        if final_x >= self.config.max_takeoff_distance_m and final_v < v_rotate:
            return {
                "completed": False,
                "reason": "takeoff_distance_exceeded",
                "distance_m": final_x,
                "time_s": final_t,
                "energy_wh": final_energy_wh,
                "final_soc": final_soc,
                "max_current_a": max_current,
            }

        return {
            "completed": True,
            "distance_m": final_x,
            "time_s": final_t,
            "energy_wh": final_energy_wh,
            "final_soc": final_soc,
            "max_current_a": max_current,
        }

    def _simulate_air_flight(
        self,
        polar: AeroPolar,
        mass_kg: float,
        wing_area_m2: float,
        start_distance_m: float,
        start_time_s: float,
        start_speed_m_s: float,
        current_soc: StateOfCharge,
        max_current_so_far: float,
        energy_so_far_wh: float,
    ) -> MissionOutcome:
        """Simulate cruise flight to accumulate distance until target_flight_time_s."""
        remaining_time = max(0.0, self.config.target_flight_time_s - start_time_s)
        weight_n = mass_kg * self.config.g
        v_cruise = max(start_speed_m_s, 14.0)

        # Required CL at cruise
        q_cruise = 0.5 * self.config.rho_air * (v_cruise**2) * wing_area_m2
        cl_req = weight_n / max(q_cruise, 1e-4)

        alpha_trim = polar.trim_alpha_for_cl(cl_req)
        if alpha_trim is None:
            return MissionOutcome(
                completed=False,
                takeoff_distance_m=start_distance_m,
                total_distance_m=start_distance_m,
                energy_consumed_wh=energy_so_far_wh,
                final_soc=current_soc.value,
                max_current_a=max_current_so_far,
                failure_reason="cruise_cl_untrimmable",
            )

        cd_trim = polar.drag_coefficient(alpha_trim)
        drag_cruise_n = q_cruise * cd_trim

        # Required thrust in level cruise = drag
        throttle_guess = 0.6
        pt_cruise = self.powertrain.evaluate(v_cruise, throttle_guess, current_soc)

        # Calculate cruise distance and energy
        cruise_distance_m = v_cruise * remaining_time
        total_distance_m = start_distance_m + cruise_distance_m
        cruise_energy_wh = (pt_cruise.electric_power_w * remaining_time) / 3600.0
        total_energy_wh = energy_so_far_wh + cruise_energy_wh

        final_soc_val = max(0.0, current_soc.value - (pt_cruise.current_a * remaining_time) / (3600.0 * 2.2))
        max_current = max(max_current_so_far, pt_cruise.current_a)

        return MissionOutcome(
            completed=True,
            takeoff_distance_m=start_distance_m,
            total_distance_m=total_distance_m,
            energy_consumed_wh=total_energy_wh,
            final_soc=final_soc_val,
            max_current_a=max_current,
            failure_reason=None,
        )
