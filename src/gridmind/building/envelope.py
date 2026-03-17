"""BuildingEnvelope — thermal resistance/capacitance modelling.

The envelope is the boundary between a zone and the outdoors.  Heat transfer
through the envelope obeys the lumped-capacitance (RC-network) equation:

    Q_env = (T_outside - T_zone) / R_env          [W]

where R_env (K/W) is the total thermal resistance of walls, roof, and
glazing in parallel.  Solar heat gain through windows is:

    Q_solar = SHGC * A_window * I_solar            [W]

Infiltration load (sensible only):

    Q_inf = rho * c_p * V_dot * (T_outside - T_zone)  [W]

where V_dot = Volume * ACH / 3600.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridmind.models import Zone

# Dry-air properties at ~20 C
AIR_DENSITY_KG_M3 = 1.2
AIR_CP_J_KG_K = 1005.0


@dataclass
class EnvelopeFlows:
    """Instantaneous heat flows through the envelope (W)."""

    q_envelope: float
    q_solar: float
    q_infiltration: float

    @property
    def total(self) -> float:
        return self.q_envelope + self.q_solar + self.q_infiltration


class BuildingEnvelope:
    """Compute heat flows across the building envelope for a single zone."""

    def __init__(self, zone: Zone) -> None:
        self.zone = zone

    # --- public API -------------------------------------------------------

    def heat_flows(
        self,
        zone_temp_c: float,
        outdoor_temp_c: float,
        solar_irradiance_w_m2: float,
    ) -> EnvelopeFlows:
        """Return all envelope-related heat flows for the current timestep."""
        q_env = self._conduction(zone_temp_c, outdoor_temp_c)
        q_sol = self._solar_gain(solar_irradiance_w_m2)
        q_inf = self._infiltration(zone_temp_c, outdoor_temp_c)
        return EnvelopeFlows(q_envelope=q_env, q_solar=q_sol, q_infiltration=q_inf)

    # --- private helpers --------------------------------------------------

    def _conduction(self, t_zone: float, t_out: float) -> float:
        """Steady-state conduction: Q = (T_out - T_zone) / R_env."""
        return (t_out - t_zone) / self.zone.envelope_resistance_k_per_w

    def _solar_gain(self, irradiance: float) -> float:
        """Solar gain through glazing: Q = SHGC * A_win * I."""
        return self.zone.window_shgc * self.zone.window_area_m2 * irradiance

    def _infiltration(self, t_zone: float, t_out: float) -> float:
        """Sensible infiltration load: rho * cp * V_dot * dT."""
        v_dot = self.zone.volume_m3 * self.zone.infiltration_ach / 3600.0
        return AIR_DENSITY_KG_M3 * AIR_CP_J_KG_K * v_dot * (t_out - t_zone)
