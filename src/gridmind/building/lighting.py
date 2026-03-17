"""LightingOptimizer — occupancy and daylight-based lighting control.

Lighting power at each hour is determined by:

1. **Occupancy factor** — lights dim proportionally to occupancy.
2. **Daylight harvesting** — when solar irradiance exceeds a threshold
   the artificial lighting fraction is reduced.
3. **Smooth transitions** — an exponential moving-average filter prevents
   abrupt changes in dimming level.

Electrical power:

    P_light = P_installed * dim_level
"""

from __future__ import annotations

import numpy as np

from gridmind.models import Zone


class LightingOptimizer:
    """Compute hourly lighting power for a zone."""

    def __init__(
        self,
        zone: Zone,
        daylight_threshold_w_m2: float = 300.0,
        min_dim_fraction: float = 0.05,
        smoothing_alpha: float = 0.3,
    ) -> None:
        self.zone = zone
        self.daylight_threshold = daylight_threshold_w_m2
        self.min_dim = min_dim_fraction
        self.alpha = smoothing_alpha
        self._prev_dim: float = 0.0

    def _daylight_factor(self, solar_irradiance: float) -> float:
        """Fraction of artificial light displaced by daylight (0-1)."""
        if solar_irradiance <= 0:
            return 0.0
        return min(1.0, solar_irradiance / self.daylight_threshold)

    def dim_level(
        self, occupancy_fraction: float, solar_irradiance: float
    ) -> float:
        """Return the dimming level (0-1) after occupancy and daylight adjustment."""
        occ = max(0.0, min(1.0, occupancy_fraction))
        if occ == 0.0:
            raw = 0.0
        else:
            daylight_offset = self._daylight_factor(solar_irradiance)
            raw = max(self.min_dim, occ * (1.0 - 0.6 * daylight_offset))

        # Exponential smoothing to prevent abrupt jumps
        smoothed = self.alpha * raw + (1.0 - self.alpha) * self._prev_dim
        self._prev_dim = smoothed
        return smoothed

    def power_w(
        self, occupancy_fraction: float, solar_irradiance: float
    ) -> float:
        """Electrical power consumed by lighting this hour (W)."""
        return self.zone.lighting_power_w * self.dim_level(
            occupancy_fraction, solar_irradiance
        )

    def schedule(
        self,
        occupancy_fractions: list[float],
        solar_irradiances: list[float],
    ) -> np.ndarray:
        """Return an array of hourly lighting power (W) for the full period."""
        self._prev_dim = 0.0  # reset state
        powers = np.zeros(len(occupancy_fractions))
        for h in range(len(occupancy_fractions)):
            powers[h] = self.power_w(occupancy_fractions[h], solar_irradiances[h])
        return powers
