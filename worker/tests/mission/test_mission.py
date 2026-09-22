import numpy as np
import pytest

from worker.mission.polar import AeroPolar
from worker.mission.trajectory import MissionConfig, MissionOutcome, MissionSimulator
from worker.powertrain.battery import Battery, Capacity, StateOfCharge
from worker.powertrain.motor import Motor, MotorKV, MotorResistance
from worker.powertrain.propeller import Propeller, PropellerDimension
from worker.powertrain.powertrain import Powertrain


@pytest.fixture
def sample_polar() -> AeroPolar:
    alpha = np.linspace(-2.0, 10.0, 13)
    cl = 0.08 * alpha + 0.3
    cd = 0.025 + 0.002 * (alpha**2)
    cm = -0.05 - 0.01 * alpha
    return AeroPolar(alpha_deg=alpha, cl=cl, cd=cd, cm=cm)


@pytest.fixture
def sample_powertrain() -> Powertrain:
    battery = Battery(cell_count=3, capacity=Capacity(2.2), internal_resistance_ohms=0.03)
    motor = Motor(
        kv=MotorKV(1000.0),
        resistance=MotorResistance(0.06),
        no_load_current_a=0.8,
        max_current_a=40.0,
    )
    prop = Propeller(dimensions=PropellerDimension(diameter_m=0.254, pitch_m=0.1524))
    return Powertrain(battery=battery, motor=motor, propeller=prop)


def test_mission_simulation_completes_successfully(
    sample_polar: AeroPolar,
    sample_powertrain: Powertrain,
):
    simulator = MissionSimulator(
        powertrain=sample_powertrain,
        config=MissionConfig(
            cruise_altitude_m=20.0,
            target_flight_time_s=30.0,
            rolling_friction_coeff=0.04,
            max_takeoff_distance_m=40.0,
        ),
    )

    # 1.2 kg plane (Weight = 11.77 N), wing area 0.15 m^2
    outcome = simulator.simulate(
        polar=sample_polar,
        mass_kg=1.2,
        wing_area_m2=0.15,
    )

    assert isinstance(outcome, MissionOutcome)
    assert outcome.completed is True
    assert outcome.failure_reason is None
    # Takeoff should occur within 40m
    assert 0.0 < outcome.takeoff_distance_m <= 40.0
    # Total distance > takeoff distance
    assert outcome.total_distance_m > outcome.takeoff_distance_m
    # Energy consumed should be positive
    assert outcome.energy_consumed_wh > 0.0
    # Battery SoC should be depleted somewhat
    assert 0.0 < outcome.final_soc < 1.0
    assert outcome.max_current_a > 0.0


def test_mission_fails_if_aircraft_cannot_takeoff(
    sample_polar: AeroPolar,
    sample_powertrain: Powertrain,
):
    simulator = MissionSimulator(
        powertrain=sample_powertrain,
        config=MissionConfig(
            cruise_altitude_m=20.0,
            target_flight_time_s=30.0,
            rolling_friction_coeff=0.04,
            max_takeoff_distance_m=10.0,  # Unrealistic 10m threshold
        ),
    )

    # Overweight 5.0 kg plane on small wing cannot lift off within 10m
    outcome = simulator.simulate(
        polar=sample_polar,
        mass_kg=5.0,
        wing_area_m2=0.08,
    )

    assert outcome.completed is False
    assert outcome.failure_reason == "takeoff_distance_exceeded"
