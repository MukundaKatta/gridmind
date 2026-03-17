"""BuildingSimulator — thermal dynamics engine.

Solves the lumped-capacitance energy-balance ODE for each zone at hourly
time-steps using ``scipy.integrate.solve_ivp``.

For zone *z* the governing equation is:

    C_z * dT_z/dt = Q_hvac + Q_solar + Q_internal
                    + Q_infiltration + Q_envelope

where:
    Q_envelope    = (T_outside - T_z) / R_env
    Q_solar       = SHGC * A_window * I_solar
    Q_infiltration = rho * c_p * V_dot * (T_outside - T_z)
    Q_internal    = constant internal gains (people + equipment)
    Q_hvac        = controlled heating (+) or cooling (-) power

The simulator steps hour-by-hour, calling the HVAC controller and lighting
optimizer at each step, then integrating the ODE over a 3600-second interval.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from gridmind.building.envelope import BuildingEnvelope
from gridmind.building.hvac import HVACOptimizer
from gridmind.building.lighting import LightingOptimizer
from gridmind.models import Building, EnergyProfile, WeatherData


class BuildingSimulator:
    """Run an hourly thermal simulation for a building."""

    DT_SECONDS = 3600.0  # one-hour timestep

    def __init__(
        self,
        building: Building,
        weather: WeatherData,
        initial_temp_c: float | None = None,
    ) -> None:
        self.building = building
        self.weather = weather
        self.n_hours = weather.hours

        default_temp = (building.heating_setpoint_c + building.cooling_setpoint_c) / 2.0
        self.initial_temp = initial_temp_c if initial_temp_c is not None else default_temp

        # Per-zone helpers
        self.envelopes = [BuildingEnvelope(z) for z in building.zones]
        self.hvac = HVACOptimizer(building)
        self.lighting_opts = [LightingOptimizer(z) for z in building.zones]

    # ------------------------------------------------------------------

    def run(self) -> EnergyProfile:
        """Execute the simulation and return an EnergyProfile."""
        n_zones = len(self.building.zones)
        occupancy = self.building.occupancy.for_hours(self.n_hours)

        # Storage arrays
        zone_temps = np.full((n_zones, self.n_hours), self.initial_temp)
        hvac_kw = np.zeros(self.n_hours)
        lighting_kw = np.zeros(self.n_hours)
        hvac_modes: list[list[str]] = [[] for _ in range(n_zones)]

        current_temps = np.full(n_zones, self.initial_temp)

        for h in range(self.n_hours):
            t_out = self.weather.outdoor_temp_c[h]
            irr = self.weather.solar_irradiance_w_m2[h]
            occ = occupancy[h]

            hour_hvac_elec = 0.0
            hour_light_elec = 0.0

            for zi, zone in enumerate(self.building.zones):
                # 1) HVAC decision
                hvac_out = self.hvac.control(zone, current_temps[zi], occ)
                hvac_modes[zi].append(hvac_out.mode.value)

                # 2) Envelope heat flows
                env_flows = self.envelopes[zi].heat_flows(current_temps[zi], t_out, irr)

                # 3) Total heat input for ODE
                q_total = (
                    hvac_out.thermal_power_w
                    + env_flows.total
                    + zone.internal_gain_w * occ
                )

                # 4) Integrate ODE:  C dT/dt = Q_total  =>  dT/dt = Q_total / C
                def _ode(_t: float, _y: np.ndarray, q: float = q_total, c: float = zone.thermal_capacitance_j_per_k) -> np.ndarray:
                    return np.array([q / c])

                sol = solve_ivp(
                    _ode,
                    [0, self.DT_SECONDS],
                    [current_temps[zi]],
                    method="RK45",
                    max_step=600.0,
                )
                current_temps[zi] = float(sol.y[0, -1])
                zone_temps[zi, h] = current_temps[zi]

                # 5) Accumulate electrical energy
                hour_hvac_elec += hvac_out.electrical_power_w

                # 6) Lighting
                light_w = self.lighting_opts[zi].power_w(occ, irr)
                hour_light_elec += light_w

            hvac_kw[h] = hour_hvac_elec / 1000.0
            lighting_kw[h] = hour_light_elec / 1000.0

        total_kw = hvac_kw + lighting_kw

        return EnergyProfile(
            hours=self.n_hours,
            hvac_kwh=hvac_kw.tolist(),
            lighting_kwh=lighting_kw.tolist(),
            total_kwh=total_kw.tolist(),
            zone_temperatures_c=[zt.tolist() for zt in zone_temps],
            hvac_modes=hvac_modes,
        )
