"""HVACOptimizer — temperature setpoint scheduling and HVAC control.

The optimizer decides the HVAC mode (heating / cooling / off) and the
delivered power at each timestep based on:

* Active setpoints (which depend on occupancy)
* Current zone temperature
* Dead-band logic to avoid short-cycling

Energy delivered by the HVAC in one hour:

    E_hvac = P_hvac * dt   (J, with dt = 3600 s for hourly steps)

The COP (coefficient of performance) converts delivered thermal energy
to electrical energy consumption:

    E_elec = E_hvac / COP
"""

from __future__ import annotations

from dataclasses import dataclass

from gridmind.models import Building, HVACMode, Zone


@dataclass
class HVACOutput:
    """Result of one HVAC control step."""

    mode: HVACMode
    thermal_power_w: float          # positive = heating, negative = cooling
    electrical_power_w: float       # always >= 0


class HVACOptimizer:
    """Proportional dead-band HVAC controller with setpoint scheduling."""

    def __init__(
        self,
        building: Building,
        heating_cop: float = 3.0,
        cooling_cop: float = 2.5,
        dead_band_c: float = 0.5,
    ) -> None:
        self.building = building
        self.heating_cop = heating_cop
        self.cooling_cop = cooling_cop
        self.dead_band_c = dead_band_c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def active_setpoints(self, occupancy_fraction: float) -> tuple[float, float]:
        """Return (heating_sp, cooling_sp) blended between occupied and setback."""
        occ = max(0.0, min(1.0, occupancy_fraction))
        h_sp = occ * self.building.heating_setpoint_c + (1 - occ) * self.building.setback_heating_c
        c_sp = occ * self.building.cooling_setpoint_c + (1 - occ) * self.building.setback_cooling_c
        return h_sp, c_sp

    def control(
        self,
        zone: Zone,
        zone_temp_c: float,
        occupancy_fraction: float,
    ) -> HVACOutput:
        """Determine HVAC output for a single zone at the current timestep."""
        h_sp, c_sp = self.active_setpoints(occupancy_fraction)

        # Dead-band: no action when temperature is within band
        if zone_temp_c < h_sp - self.dead_band_c:
            # Need heating
            error = h_sp - zone_temp_c
            power = min(zone.hvac_capacity_w, zone.hvac_capacity_w * error / 3.0)
            return HVACOutput(
                mode=HVACMode.HEATING,
                thermal_power_w=power,
                electrical_power_w=power / self.heating_cop,
            )

        if zone_temp_c > c_sp + self.dead_band_c:
            # Need cooling (thermal power is negative, electrical is positive)
            error = zone_temp_c - c_sp
            power = min(zone.hvac_capacity_w, zone.hvac_capacity_w * error / 3.0)
            return HVACOutput(
                mode=HVACMode.COOLING,
                thermal_power_w=-power,
                electrical_power_w=power / self.cooling_cop,
            )

        # Within dead-band — HVAC off
        return HVACOutput(
            mode=HVACMode.OFF,
            thermal_power_w=0.0,
            electrical_power_w=0.0,
        )

    def pre_condition_power(
        self,
        zone: Zone,
        current_temp_c: float,
        target_temp_c: float,
        lead_time_hours: float = 1.0,
    ) -> float:
        """Estimate thermal power (W) needed to reach *target_temp_c* within
        *lead_time_hours*, assuming a simplified energy balance:

            P = C_z * (target - current) / (lead_time * 3600)
        """
        dt = target_temp_c - current_temp_c
        if abs(dt) < 0.1:
            return 0.0
        power = zone.thermal_capacitance_j_per_k * dt / (lead_time_hours * 3600.0)
        return max(-zone.hvac_capacity_w, min(zone.hvac_capacity_w, power))
