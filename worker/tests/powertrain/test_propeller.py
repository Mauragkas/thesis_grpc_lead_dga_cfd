import pytest
from worker.powertrain.propeller import Propeller, PropellerDimension


def test_propeller_dimensions_must_be_positive():
    with pytest.raises(ValueError, match="Diameter must be positive"):
        PropellerDimension(diameter_m=-0.2, pitch_m=0.1)
    with pytest.raises(ValueError, match="Pitch must be positive"):
        PropellerDimension(diameter_m=0.25, pitch_m=-0.1)


def test_propeller_static_thrust_is_positive_when_spinning():
    # 10x6 inch prop ~ 0.254m diameter, 0.1524m pitch
    dim = PropellerDimension(diameter_m=0.254, pitch_m=0.1524)
    prop = Propeller(dimensions=dim)

    # Static: airspeed V = 0 m/s, 8000 RPM
    thrust_static = prop.thrust(rpm=8000.0, airspeed_m_s=0.0)
    torque_static = prop.torque(rpm=8000.0, airspeed_m_s=0.0)

    assert thrust_static > 0.0
    assert torque_static > 0.0
    # For a 10x6 prop at 8000 RPM, static thrust is typically ~5 to 12 N
    assert 4.0 <= thrust_static <= 15.0


def test_propeller_thrust_decreases_as_airspeed_increases():
    dim = PropellerDimension(diameter_m=0.254, pitch_m=0.1524)
    prop = Propeller(dimensions=dim)
    rpm = 8000.0

    t_0 = prop.thrust(rpm=rpm, airspeed_m_s=0.0)
    t_10 = prop.thrust(rpm=rpm, airspeed_m_s=10.0)
    t_25 = prop.thrust(rpm=rpm, airspeed_m_s=25.0)

    assert t_0 > t_10 > t_25
    # Above zero-thrust pitch speed, thrust drops to 0
    assert prop.thrust(rpm=rpm, airspeed_m_s=50.0) == 0.0


def test_zero_rpm_produces_zero_thrust_and_torque():
    dim = PropellerDimension(diameter_m=0.254, pitch_m=0.1524)
    prop = Propeller(dimensions=dim)

    assert prop.thrust(rpm=0.0, airspeed_m_s=10.0) == 0.0
    assert prop.torque(rpm=0.0, airspeed_m_s=10.0) == 0.0
