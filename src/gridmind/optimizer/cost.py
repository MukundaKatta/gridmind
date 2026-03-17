"""CostOptimizer — time-of-use (TOU) electricity pricing.

Supports a simple three-tier TOU tariff:

    off-peak   : 22:00 - 06:00   (cheapest)
    mid-peak   : 06:00 - 10:00, 18:00 - 22:00
    on-peak    : 10:00 - 18:00   (most expensive)

An optional demand charge ($/kW) penalises the highest single-hour demand
in the billing period.
"""

from __future__ import annotations

import numpy as np


class CostOptimizer:
    """Apply TOU pricing to an hourly energy profile."""

    def __init__(
        self,
        off_peak_rate: float = 0.06,
        mid_peak_rate: float = 0.12,
        on_peak_rate: float = 0.20,
        demand_charge_per_kw: float = 8.0,
    ) -> None:
        self.off_peak = off_peak_rate
        self.mid_peak = mid_peak_rate
        self.on_peak = on_peak_rate
        self.demand_charge = demand_charge_per_kw

    def hourly_rate(self, hour_of_day: int) -> float:
        """Return the $/kWh rate for a given hour (0-23)."""
        h = hour_of_day % 24
        if 22 <= h or h < 6:
            return self.off_peak
        if (6 <= h < 10) or (18 <= h < 22):
            return self.mid_peak
        return self.on_peak

    def rate_schedule(self, n_hours: int) -> np.ndarray:
        """Build a full rate array ($/kWh) for *n_hours* starting at hour 0."""
        return np.array([self.hourly_rate(h) for h in range(n_hours)])

    def compute_cost(self, energy_kwh: np.ndarray) -> np.ndarray:
        """Return hourly cost ($) array.  Does not include demand charge."""
        n = len(energy_kwh)
        rates = self.rate_schedule(n)
        return energy_kwh * rates

    def total_cost(self, energy_kwh: np.ndarray) -> float:
        """Total cost ($) including the monthly demand charge."""
        hourly = self.compute_cost(energy_kwh)
        peak_demand_kw = float(np.max(energy_kwh))  # kWh in 1 h ≈ kW avg
        return float(np.sum(hourly)) + self.demand_charge * peak_demand_kw
