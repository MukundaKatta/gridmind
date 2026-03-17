"""Tests for CostOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from gridmind.optimizer.cost import CostOptimizer


@pytest.fixture()
def cost() -> CostOptimizer:
    return CostOptimizer()


class TestCostOptimizer:
    def test_off_peak_rate(self, cost):
        assert cost.hourly_rate(0) == 0.06
        assert cost.hourly_rate(3) == 0.06
        assert cost.hourly_rate(23) == 0.06

    def test_mid_peak_rate(self, cost):
        assert cost.hourly_rate(7) == 0.12
        assert cost.hourly_rate(19) == 0.12

    def test_on_peak_rate(self, cost):
        assert cost.hourly_rate(12) == 0.20
        assert cost.hourly_rate(15) == 0.20

    def test_rate_schedule_length(self, cost):
        rates = cost.rate_schedule(48)
        assert len(rates) == 48

    def test_compute_cost(self, cost):
        energy = np.array([10.0] * 24)
        hourly_cost = cost.compute_cost(energy)
        assert len(hourly_cost) == 24
        assert hourly_cost[12] == pytest.approx(10.0 * 0.20)

    def test_total_cost_includes_demand_charge(self, cost):
        energy = np.array([10.0] * 24)
        total = cost.total_cost(energy)
        hourly_sum = float(np.sum(cost.compute_cost(energy)))
        demand = 8.0 * 10.0  # demand_charge * peak_kw
        assert total == pytest.approx(hourly_sum + demand)

    def test_zero_energy(self, cost):
        energy = np.zeros(24)
        assert cost.total_cost(energy) == pytest.approx(0.0)
