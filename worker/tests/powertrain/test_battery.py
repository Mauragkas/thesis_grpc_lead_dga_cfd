import pytest
from worker.powertrain.battery import Battery, Capacity, StateOfCharge


def test_capacity_rejects_non_positive_value():
    with pytest.raises(ValueError, match="Capacity must be positive"):
        Capacity(-1.0)
    with pytest.raises(ValueError, match="Capacity must be positive"):
        Capacity(0.0)


def test_state_of_charge_clamps_or_validates_range():
    with pytest.raises(ValueError, match="State of charge must be between 0 and 1"):
        StateOfCharge(-0.1)
    with pytest.raises(ValueError, match="State of charge must be between 0 and 1"):
        StateOfCharge(1.1)


def test_battery_open_circuit_voltage_at_full_and_empty():
    # 3S battery (3 cells), 2.2 Ah, 0.03 ohm internal resistance
    battery = Battery(cell_count=3, capacity=Capacity(2.2), internal_resistance_ohms=0.03)

    v_full = battery.open_circuit_voltage(StateOfCharge(1.0))
    v_empty = battery.open_circuit_voltage(StateOfCharge(0.0))

    # 3 cells: full is ~12.6V (4.2V/cell), empty is ~9.9V (3.3V/cell)
    assert v_full == pytest.approx(12.6, rel=0.05)
    assert v_empty == pytest.approx(9.9, rel=0.05)
    assert v_full > v_empty


def test_battery_terminal_voltage_sags_under_load():
    battery = Battery(cell_count=3, capacity=Capacity(2.2), internal_resistance_ohms=0.03)
    soc = StateOfCharge(0.8)

    v_no_load = battery.terminal_voltage(current_amps=0.0, soc=soc)
    v_loaded = battery.terminal_voltage(current_amps=20.0, soc=soc)

    # Sag = I * R = 20 * 0.03 = 0.6V
    assert v_no_load - v_loaded == pytest.approx(0.6, abs=1e-3)


def test_battery_discharge_reduces_soc_proportionally():
    battery = Battery(cell_count=3, capacity=Capacity(2.0), internal_resistance_ohms=0.03)
    initial_soc = StateOfCharge(1.0)

    # Discharging at 2.0A for 1800s (0.5 hour) consumes 1.0 Ah -> SoC should decrease from 1.0 to 0.5
    next_soc = battery.discharge(current_amps=2.0, duration_seconds=1800.0, current_soc=initial_soc)
    assert next_soc.value == pytest.approx(0.5, abs=1e-3)


def test_battery_discharge_does_not_drop_below_zero():
    battery = Battery(cell_count=3, capacity=Capacity(1.0), internal_resistance_ohms=0.03)
    initial_soc = StateOfCharge(0.1)

    # Discharging 50A for 100s consumes 1.38 Ah > 0.1 Ah remaining
    next_soc = battery.discharge(current_amps=50.0, duration_seconds=100.0, current_soc=initial_soc)
    assert next_soc.value == 0.0
