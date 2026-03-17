"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gridmind.models import Building, EnergyProfile, OccupancySchedule, WeatherData, Zone


class TestZone:
    def test_defaults(self):
        z = Zone()
        assert z.area_m2 == 200.0
        assert z.volume_m3 == 200.0 * 3.0

    def test_custom(self):
        z = Zone(area_m2=100, height_m=4)
        assert z.volume_m3 == 400.0

    def test_negative_area_rejected(self):
        with pytest.raises(ValidationError):
            Zone(area_m2=-1)


class TestWeatherData:
    def test_valid(self):
        w = WeatherData(outdoor_temp_c=[20.0], solar_irradiance_w_m2=[500.0])
        assert w.hours == 1
        assert w.wind_speed_m_s == [0.0]

    def test_mismatched_lengths(self):
        with pytest.raises(ValidationError):
            WeatherData(outdoor_temp_c=[20.0, 21.0], solar_irradiance_w_m2=[500.0])


class TestBuilding:
    def test_defaults(self):
        b = Building()
        assert len(b.zones) == 1
        assert b.heating_setpoint_c < b.cooling_setpoint_c

    def test_invalid_setpoints(self):
        with pytest.raises(ValidationError):
            Building(heating_setpoint_c=25, cooling_setpoint_c=20)


class TestOccupancy:
    def test_24h_length(self):
        sched = OccupancySchedule()
        assert len(sched.fractions) == 24

    def test_tiling(self):
        sched = OccupancySchedule()
        out = sched.for_hours(48)
        assert len(out) == 48
        assert out[:24] == out[24:]


class TestEnergyProfile:
    def test_totals(self):
        ep = EnergyProfile(
            hours=3,
            hvac_kwh=[1.0, 2.0, 3.0],
            lighting_kwh=[0.5, 0.5, 0.5],
            total_kwh=[1.5, 2.5, 3.5],
            cost_usd=[0.10, 0.20, 0.30],
        )
        assert ep.total_energy_kwh == pytest.approx(7.5)
        assert ep.total_cost == pytest.approx(0.60)
