"""Tests for DemandPredictor."""

from __future__ import annotations

import numpy as np
import pytest

from gridmind.models import Building, WeatherData
from gridmind.optimizer.predictor import DemandPredictor


class TestDemandPredictor:
    def test_output_length(self, building, weather_24h):
        pred = DemandPredictor(building)
        demand = pred.predict(weather_24h)
        assert len(demand) == 24

    def test_non_negative(self, building, weather_24h):
        pred = DemandPredictor(building)
        demand = pred.predict(weather_24h)
        assert np.all(demand >= 0)

    def test_higher_demand_with_extreme_temps(self, building):
        mild = WeatherData(
            outdoor_temp_c=[18.0] * 24,
            solar_irradiance_w_m2=[0.0] * 24,
        )
        cold = WeatherData(
            outdoor_temp_c=[0.0] * 24,
            solar_irradiance_w_m2=[0.0] * 24,
        )
        pred = DemandPredictor(building)
        d_mild = pred.predict(mild)
        d_cold = pred.predict(cold)
        assert np.sum(d_cold) > np.sum(d_mild)

    def test_occupancy_increases_demand(self, building):
        weather = WeatherData(
            outdoor_temp_c=[20.0] * 24,
            solar_irradiance_w_m2=[0.0] * 24,
        )
        pred = DemandPredictor(building)
        demand = pred.predict(weather)
        # Hours with higher occupancy should have higher demand
        # Occupied hours (8-12, 13-17) vs unoccupied (0-5)
        assert demand[10] > demand[2]
