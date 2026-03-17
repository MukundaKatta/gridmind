"""EnergyScheduler — optimise hourly energy use via load shifting.

Given a predicted demand profile and TOU pricing, the scheduler shifts
flexible loads (e.g., pre-conditioning, battery charging) from expensive
hours to cheap hours, subject to comfort constraints.

The optimisation minimises:

    sum_h [ rate(h) * E(h) ]  +  demand_charge * max_h(E(h))

using ``scipy.optimize.minimize`` (SLSQP) with equality constraint on
total energy and box bounds per hour.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from gridmind.optimizer.cost import CostOptimizer


class EnergyScheduler:
    """Shift flexible load to minimise electricity cost."""

    def __init__(
        self,
        cost_optimizer: CostOptimizer,
        flexibility_fraction: float = 0.30,
    ) -> None:
        self.cost = cost_optimizer
        self.flex = flexibility_fraction

    def optimize(self, baseline_kwh: np.ndarray) -> np.ndarray:
        """Return an optimised hourly energy schedule (kWh).

        The total energy is conserved; up to *flexibility_fraction* of each
        hour's load may be shifted to other hours.
        """
        n = len(baseline_kwh)
        total_energy = float(np.sum(baseline_kwh))
        rates = self.cost.rate_schedule(n)

        # Decision variable: scheduled energy per hour
        x0 = baseline_kwh.copy()

        # Bounds: each hour between (1-flex)*baseline and (1+flex)*baseline,
        # but at least 0.
        lb = np.maximum(baseline_kwh * (1 - self.flex), 0.0)
        ub = baseline_kwh * (1 + self.flex)
        bounds = list(zip(lb.tolist(), ub.tolist()))

        # Objective: minimise energy cost (ignoring demand charge for speed)
        def objective(x: np.ndarray) -> float:
            return float(np.dot(rates, x))

        # Equality constraint: total energy is preserved
        constraints = [
            {"type": "eq", "fun": lambda x: float(np.sum(x) - total_energy)}
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if result.success:
            return np.array(result.x)
        # Fall back to baseline on failure
        return baseline_kwh.copy()
