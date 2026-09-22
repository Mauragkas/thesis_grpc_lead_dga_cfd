import pytest
from worker.powertrain.motor import Motor, MotorKV, MotorResistance


def test_motor_kv_must_be_positive():
    with pytest.raises(ValueError, match="KV must be positive"):
        MotorKV(-100.0)
    with pytest.raises(ValueError, match="KV must be positive"):
        MotorKV(0.0)


def test_motor_resistance_must_be_positive():
    with pytest.raises(ValueError, match="Resistance must be positive"):
        MotorResistance(-0.01)
    with pytest.raises(ValueError, match="Resistance must be positive"):
        MotorResistance(0.0)


def test_motor_back_emf_scales_with_rpm():
    # 1000 KV motor
    motor = Motor(
        kv=MotorKV(1000.0),
        resistance=MotorResistance(0.05),
        no_load_current_a=0.8,
        max_current_a=40.0,
    )
    # At 5000 RPM, back-emf = 5000 / 1000 = 5.0 V
    assert motor.back_emf(5000.0) == pytest.approx(5.0)


def test_motor_current_under_applied_voltage():
    motor = Motor(
        kv=MotorKV(1000.0),
        resistance=MotorResistance(0.1),
        no_load_current_a=1.0,
        max_current_a=45.0,
    )
    # Applied 11.0V, back-emf at 10000 RPM is 10.0V
    # Current = (11.0 - 10.0) / 0.1 = 10.0 A
    assert motor.current(voltage=11.0, rpm=10000.0) == pytest.approx(10.0)


def test_motor_torque_calculation():
    # Motor torque constant kt = 30 / (pi * kv) = 30 / (3.14159 * 1000) ~ 0.009549 Nm/A
    motor = Motor(
        kv=MotorKV(1000.0),
        resistance=MotorResistance(0.05),
        no_load_current_a=1.0,
        max_current_a=45.0,
    )
    # At 11.0 A current, effective torque current is 10.0 A -> ~0.0955 Nm
    torque = motor.torque(current_amps=11.0)
    assert torque == pytest.approx(0.0955, rel=1e-2)


def test_motor_detects_overcurrent():
    motor = Motor(
        kv=MotorKV(900.0),
        resistance=MotorResistance(0.08),
        no_load_current_a=1.0,
        max_current_a=35.0,
    )
    assert not motor.is_overcurrent(30.0)
    assert motor.is_overcurrent(40.0)
