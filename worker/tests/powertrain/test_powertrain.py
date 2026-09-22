import pytest
from worker.powertrain.battery import Battery, Capacity, StateOfCharge
from worker.powertrain.motor import Motor, MotorKV, MotorResistance
from worker.powertrain.propeller import Propeller, PropellerDimension
from worker.powertrain.powertrain import Powertrain, PowertrainOperatingPoint


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


def test_powertrain_zero_throttle_produces_zero_thrust(sample_powertrain: Powertrain):
    point = sample_powertrain.evaluate(airspeed_m_s=10.0, throttle=0.0, soc=StateOfCharge(1.0))
    assert point.thrust_n == 0.0
    assert point.current_a == 0.0
    assert point.rpm == 0.0
    assert point.electric_power_w == 0.0


def test_powertrain_full_throttle_static_equilibrium(sample_powertrain: Powertrain):
    # Airspeed = 0 m/s, full throttle (1.0), full battery
    point = sample_powertrain.evaluate(airspeed_m_s=0.0, throttle=1.0, soc=StateOfCharge(1.0))

    # Expect equilibrium RPM around 7000 - 11000 RPM for 3S 1000KV on 10x6
    assert 6000.0 <= point.rpm <= 11000.0
    # Current should be realistic for this setup (~10 to 30 A)
    assert 5.0 <= point.current_a <= 35.0
    # Thrust should be positive (~4 to 12 N)
    assert 4.0 <= point.thrust_n <= 15.0
    # Terminal voltage should sag below 12.6V
    assert point.terminal_voltage_v < 12.6
    assert point.electric_power_w == pytest.approx(point.current_a * point.terminal_voltage_v, rel=1e-2)


def test_powertrain_thrust_drops_with_forward_speed(sample_powertrain: Powertrain):
    soc = StateOfCharge(0.9)
    static_pt = sample_powertrain.evaluate(airspeed_m_s=0.0, throttle=1.0, soc=soc)
    cruise_pt = sample_powertrain.evaluate(airspeed_m_s=15.0, throttle=1.0, soc=soc)

    assert cruise_pt.thrust_n < static_pt.thrust_n
    # Power required at higher advance ratio is typically lower at full throttle
    assert cruise_pt.current_a <= static_pt.current_a
