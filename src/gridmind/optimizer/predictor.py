"""DemandPredictor — forecast hourly energy demand from weather and occupancy.

Uses a simple regression-style model:

    E_predicted[h] = (alpha * |T_out[h] - T_balance|
                      + beta * occupancy[h]
                      + gamma * solar[h] / 1000)
                     * base_load_kw

The coefficients are tunable.  For a real deployment they would be learned
from historical data; here they provide a physics-informed heuristic.
"""

from __future__ import annotations

import numpy as np

from gridmind.models import Building, WeatherData


class DemandPredictor:
    """Predict hourly building energy demand (kWh)."""

    def __init__(
        self,
        building: Building,
        balance_temp_c: float = 18.0,
        alpha: float = 0.15,
        beta: float = 0.50,
        gamma: float = 0.10,
    ) -> None:
        self.building = building
        self.balance_temp = balance_temp_c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Base load derived from total installed capacity
        total_hvac = sum(z.hvac_capacity_w for z in building.zones)
        total_light = sum(z.lighting_power_w for z in building.zones)
        self.base_load_kw = (total_hvac + total_light) / 1000.0

    def predict(self, weather: WeatherData) -> np.ndarray:
        """Return predicted demand (kWh) for each hour."""
        n = weather.hours
        occupancy = self.building.occupancy.for_hours(n)

        t_out = np.array(weather.outdoor_temp_c)
        occ = np.array(occupancy)
        solar = np.array(weather.solar_irradiance_w_m2)

        temp_term = self.alpha * np.abs(t_out - self.balance_temp)
        occ_term = self.beta * occ
        solar_term = self.gamma * solar / 1000.0

        demand = (temp_term + occ_term + solar_term) * self.base_load_kw
        return np.maximum(demand, 0.0)
