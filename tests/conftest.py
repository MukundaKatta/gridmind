"""Shared fixtures for GRIDMIND tests."""

from __future__ import annotations

import numpy as np
import pytest

from gridmind.models import Building, WeatherData, Zone


@pytest.fixture()
def zone() -> Zone:
    return Zone(name="TestZone")


@pytest.fixture()
def building() -> Building:
    return Building(name="TestBuilding", zones=[Zone(name="Z1")])


@pytest.fixture()
def weather_24h() -> WeatherData:
    """Synthetic 24-hour weather."""
    h = np.arange(24)
    temp = 15.0 + 10.0 * np.sin(2 * np.pi * (h - 6) / 24)
    solar = np.clip(800.0 * np.sin(np.pi * (h - 6) / 12), 0, None)
    return WeatherData(
        outdoor_temp_c=temp.tolist(),
        solar_irradiance_w_m2=solar.tolist(),
    )
