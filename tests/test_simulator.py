"""Tests for BuildingSimulator."""

from __future__ import annotations

import pytest

from gridmind.models import Building, WeatherData, Zone
from gridmind.simulator import BuildingSimulator


class TestBuildingSimulator:
    def test_returns_correct_hours(self, building, weather_24h):
        sim = BuildingSimulator(building, weather_24h)
        profile = sim.run()
        assert profile.hours == 24
        assert len(profile.hvac_kwh) == 24
        assert len(profile.lighting_kwh) == 24
        assert len(profile.total_kwh) == 24

    def test_temperatures_reasonable(self, building, weather_24h):
        sim = BuildingSimulator(building, weather_24h)
        profile = sim.run()
        temps = profile.zone_temperatures_c[0]
        # Indoor temperature should stay between 5 and 40 C
        assert all(5 < t < 40 for t in temps)

    def test_energy_non_negative(self, building, weather_24h):
        sim = BuildingSimulator(building, weather_24h)
        profile = sim.run()
        assert all(e >= 0 for e in profile.total_kwh)

    def test_hvac_modes_populated(self, building, weather_24h):
        sim = BuildingSimulator(building, weather_24h)
        profile = sim.run()
        assert len(profile.hvac_modes) == 1  # one zone
        assert len(profile.hvac_modes[0]) == 24

    def test_initial_temp_respected(self, building, weather_24h):
        sim = BuildingSimulator(building, weather_24h, initial_temp_c=18.0)
        profile = sim.run()
        # First-hour temperature should be near 18 C (will drift during the hour)
        assert abs(profile.zone_temperatures_c[0][0] - 18.0) < 5.0

    def test_multi_zone(self, weather_24h):
        b = Building(zones=[Zone(name="A"), Zone(name="B")])
        sim = BuildingSimulator(b, weather_24h)
        profile = sim.run()
        assert len(profile.zone_temperatures_c) == 2
