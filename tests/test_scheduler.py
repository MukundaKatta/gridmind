"""Tests for EnergyScheduler."""

from __future__ import annotations

import numpy as np
import pytest

from gridmind.optimizer.cost import CostOptimizer
from gridmind.optimizer.scheduler import EnergyScheduler


@pytest.fixture()
def cost_opt() -> CostOptimizer:
    return CostOptimizer()


@pytest.fixture()
def scheduler(cost_opt) -> EnergyScheduler:
    return EnergyScheduler(cost_opt, flexibility_fraction=0.30)


class TestEnergyScheduler:
    def test_energy_conserved(self, scheduler):
        baseline = np.array([5.0] * 24)
        optimized = scheduler.optimize(baseline)
        assert np.sum(optimized) == pytest.approx(np.sum(baseline), rel=1e-3)

    def test_cost_reduced_or_equal(self, scheduler, cost_opt):
        baseline = np.array([5.0] * 24)
        optimized = scheduler.optimize(baseline)
        assert cost_opt.total_cost(optimized) <= cost_opt.total_cost(baseline) + 0.01

    def test_bounds_respected(self, scheduler):
        baseline = np.array([10.0] * 24)
        optimized = scheduler.optimize(baseline)
        lb = baseline * 0.70
        ub = baseline * 1.30
        assert np.all(optimized >= lb - 1e-6)
        assert np.all(optimized <= ub + 1e-6)

    def test_shifts_load_to_off_peak(self, scheduler):
        """On-peak hours should have less energy after optimization."""
        baseline = np.array([10.0] * 24)
        optimized = scheduler.optimize(baseline)
        # On-peak hours: 10-17
        on_peak_baseline = baseline[10:18].sum()
        on_peak_opt = optimized[10:18].sum()
        assert on_peak_opt <= on_peak_baseline + 0.01
