"""Tests for BuildingEnvelope."""

from __future__ import annotations

import pytest

from gridmind.building.envelope import AIR_CP_J_KG_K, AIR_DENSITY_KG_M3, BuildingEnvelope
from gridmind.models import Zone


@pytest.fixture()
def envelope() -> BuildingEnvelope:
    return BuildingEnvelope(Zone())


class TestBuildingEnvelope:
    def test_conduction_direction(self, envelope: BuildingEnvelope):
        """Heat should flow into zone when outdoor > indoor."""
        flows = envelope.heat_flows(20.0, 30.0, 0.0)
        assert flows.q_envelope > 0

    def test_conduction_zero_when_equal(self, envelope: BuildingEnvelope):
        flows = envelope.heat_flows(22.0, 22.0, 0.0)
        assert flows.q_envelope == pytest.approx(0.0)
        assert flows.q_infiltration == pytest.approx(0.0)

    def test_solar_gain_positive(self, envelope: BuildingEnvelope):
        flows = envelope.heat_flows(22.0, 22.0, 500.0)
        expected = 0.4 * 30.0 * 500.0  # SHGC * A_win * I
        assert flows.q_solar == pytest.approx(expected)

    def test_solar_gain_zero_at_night(self, envelope: BuildingEnvelope):
        flows = envelope.heat_flows(22.0, 22.0, 0.0)
        assert flows.q_solar == pytest.approx(0.0)

    def test_infiltration_sign(self, envelope: BuildingEnvelope):
        """Cold outdoor air should cool the zone."""
        flows = envelope.heat_flows(22.0, 10.0, 0.0)
        assert flows.q_infiltration < 0

    def test_total_is_sum(self, envelope: BuildingEnvelope):
        flows = envelope.heat_flows(22.0, 30.0, 400.0)
        assert flows.total == pytest.approx(
            flows.q_envelope + flows.q_solar + flows.q_infiltration
        )
