"""Tests for HVACOptimizer."""

from __future__ import annotations

import pytest

from gridmind.building.hvac import HVACOptimizer, HVACOutput
from gridmind.models import Building, HVACMode, Zone


@pytest.fixture()
def hvac() -> HVACOptimizer:
    return HVACOptimizer(Building())


class TestHVACOptimizer:
    def test_heating_when_cold(self, hvac: HVACOptimizer):
        zone = Zone()
        out = hvac.control(zone, zone_temp_c=15.0, occupancy_fraction=1.0)
        assert out.mode == HVACMode.HEATING
        assert out.thermal_power_w > 0
        assert out.electrical_power_w > 0

    def test_cooling_when_hot(self, hvac: HVACOptimizer):
        zone = Zone()
        out = hvac.control(zone, zone_temp_c=30.0, occupancy_fraction=1.0)
        assert out.mode == HVACMode.COOLING
        assert out.thermal_power_w < 0
        assert out.electrical_power_w > 0

    def test_off_in_dead_band(self, hvac: HVACOptimizer):
        zone = Zone()
        out = hvac.control(zone, zone_temp_c=22.5, occupancy_fraction=1.0)
        assert out.mode == HVACMode.OFF
        assert out.thermal_power_w == 0.0

    def test_setback_when_unoccupied(self, hvac: HVACOptimizer):
        h_sp, c_sp = hvac.active_setpoints(0.0)
        assert h_sp == pytest.approx(16.0)  # setback_heating_c
        assert c_sp == pytest.approx(28.0)  # setback_cooling_c

    def test_blended_setpoints(self, hvac: HVACOptimizer):
        h_sp, c_sp = hvac.active_setpoints(0.5)
        assert h_sp == pytest.approx(0.5 * 21.0 + 0.5 * 16.0)

    def test_precondition_power(self, hvac: HVACOptimizer):
        zone = Zone()
        power = hvac.pre_condition_power(zone, current_temp_c=18.0, target_temp_c=21.0)
        expected = zone.thermal_capacitance_j_per_k * 3.0 / 3600.0
        assert power == pytest.approx(expected)

    def test_precondition_clamped(self, hvac: HVACOptimizer):
        zone = Zone(hvac_capacity_w=1000.0)
        power = hvac.pre_condition_power(zone, 10.0, 25.0)
        assert abs(power) <= 1000.0
