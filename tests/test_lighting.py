"""Tests for LightingOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from gridmind.building.lighting import LightingOptimizer
from gridmind.models import Zone


@pytest.fixture()
def lighting() -> LightingOptimizer:
    return LightingOptimizer(Zone())


class TestLightingOptimizer:
    def test_off_when_unoccupied(self):
        lo = LightingOptimizer(Zone(), smoothing_alpha=1.0)
        power = lo.power_w(0.0, 0.0)
        assert power == pytest.approx(0.0)

    def test_full_power_at_night_occupied(self):
        lo = LightingOptimizer(Zone(), smoothing_alpha=1.0)
        power = lo.power_w(1.0, 0.0)
        # Should be close to max lighting (minus min_dim effects)
        assert power > 0.9 * Zone().lighting_power_w

    def test_daylight_reduces_power(self):
        lo1 = LightingOptimizer(Zone(), smoothing_alpha=1.0)
        lo2 = LightingOptimizer(Zone(), smoothing_alpha=1.0)
        night = lo1.power_w(1.0, 0.0)
        day = lo2.power_w(1.0, 600.0)
        assert day < night

    def test_schedule_length(self, lighting: LightingOptimizer):
        occ = [0.5] * 24
        solar = [300.0] * 24
        sched = lighting.schedule(occ, solar)
        assert len(sched) == 24
        assert np.all(sched >= 0)

    def test_smoothing(self):
        """With low alpha, sudden occupancy changes should be dampened."""
        lo = LightingOptimizer(Zone(), smoothing_alpha=0.1)
        p1 = lo.power_w(0.0, 0.0)
        p2 = lo.power_w(1.0, 0.0)
        # p2 should be much less than full power due to smoothing from 0
        assert p2 < 0.5 * Zone().lighting_power_w
