"""Pydantic data models for GRIDMIND."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


class HVACMode(str, Enum):
    """Operating mode for an HVAC system."""

    HEATING = "heating"
    COOLING = "cooling"
    OFF = "off"


class WeatherData(BaseModel):
    """Hourly weather conditions for the simulation period.

    All arrays must have the same length (one entry per hour).
    """

    outdoor_temp_c: list[float] = Field(
        ..., description="Outdoor dry-bulb temperature (deg-C) per hour."
    )
    solar_irradiance_w_m2: list[float] = Field(
        ..., description="Global horizontal irradiance (W/m2) per hour."
    )
    wind_speed_m_s: list[float] = Field(
        default_factory=list,
        description="Wind speed (m/s) per hour — optional.",
    )

    @model_validator(mode="after")
    def _check_lengths(self) -> "WeatherData":
        n = len(self.outdoor_temp_c)
        if len(self.solar_irradiance_w_m2) != n:
            raise ValueError("solar_irradiance_w_m2 length must match outdoor_temp_c")
        if self.wind_speed_m_s and len(self.wind_speed_m_s) != n:
            raise ValueError("wind_speed_m_s length must match outdoor_temp_c")
        if not self.wind_speed_m_s:
            self.wind_speed_m_s = [0.0] * n
        return self

    @property
    def hours(self) -> int:
        return len(self.outdoor_temp_c)


class Zone(BaseModel):
    """A thermal zone inside a building."""

    name: str = "Zone-1"
    area_m2: float = Field(default=200.0, gt=0, description="Floor area (m2).")
    height_m: float = Field(default=3.0, gt=0, description="Ceiling height (m).")
    thermal_capacitance_j_per_k: float = Field(
        default=3.0e6,
        gt=0,
        description="Lumped thermal capacitance C_z (J/K).",
    )
    envelope_resistance_k_per_w: float = Field(
        default=0.004,
        gt=0,
        description="Envelope thermal resistance R_env (K/W).",
    )
    window_area_m2: float = Field(
        default=30.0, ge=0, description="Total glazing area (m2)."
    )
    window_shgc: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Solar heat gain coefficient of glazing.",
    )
    infiltration_ach: float = Field(
        default=0.3,
        ge=0,
        description="Infiltration rate (air changes per hour).",
    )
    internal_gain_w: float = Field(
        default=1500.0,
        ge=0,
        description="Constant internal heat gain from equipment/people (W).",
    )
    hvac_capacity_w: float = Field(
        default=15000.0,
        gt=0,
        description="Maximum HVAC heating/cooling capacity (W).",
    )
    lighting_power_w: float = Field(
        default=2000.0,
        ge=0,
        description="Installed lighting power (W).",
    )

    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.height_m


class EnergyProfile(BaseModel):
    """Hourly energy breakdown produced by a simulation run."""

    hours: int
    hvac_kwh: list[float] = Field(default_factory=list)
    lighting_kwh: list[float] = Field(default_factory=list)
    total_kwh: list[float] = Field(default_factory=list)
    zone_temperatures_c: list[list[float]] = Field(
        default_factory=list,
        description="Per-zone temperature traces (zones x hours).",
    )
    cost_usd: list[float] = Field(default_factory=list)
    hvac_modes: list[list[str]] = Field(
        default_factory=list,
        description="Per-zone HVAC mode strings (zones x hours).",
    )

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.cost_usd)) if self.cost_usd else 0.0

    @property
    def total_energy_kwh(self) -> float:
        return float(np.sum(self.total_kwh)) if self.total_kwh else 0.0


class OccupancySchedule(BaseModel):
    """Fractional occupancy (0-1) for each hour."""

    fractions: list[float] = Field(
        default_factory=lambda: (
            [0.0] * 6        # 00:00-05:59  unoccupied
            + [0.3, 0.7]     # 06:00-07:59  arrival
            + [1.0] * 4      # 08:00-11:59  peak
            + [0.6]          # 12:00-12:59  lunch
            + [1.0] * 4      # 13:00-16:59  peak
            + [0.7, 0.3]     # 17:00-18:59  departure
            + [0.0] * 5      # 19:00-23:59  unoccupied
        ),
        description="24-element list of occupancy fractions for a day.",
    )

    def for_hours(self, n_hours: int) -> list[float]:
        """Tile the 24-hour pattern to cover *n_hours*."""
        reps = (n_hours // len(self.fractions)) + 1
        return (self.fractions * reps)[:n_hours]


class Building(BaseModel):
    """Top-level building configuration."""

    name: str = "Default Office"
    zones: list[Zone] = Field(default_factory=lambda: [Zone()])
    occupancy: OccupancySchedule = Field(default_factory=OccupancySchedule)
    heating_setpoint_c: float = Field(default=21.0, description="Heating setpoint (C).")
    cooling_setpoint_c: float = Field(default=24.0, description="Cooling setpoint (C).")
    setback_heating_c: float = Field(
        default=16.0, description="Unoccupied heating setback (C)."
    )
    setback_cooling_c: float = Field(
        default=28.0, description="Unoccupied cooling setback (C)."
    )

    @model_validator(mode="after")
    def _check_setpoints(self) -> "Building":
        if self.heating_setpoint_c >= self.cooling_setpoint_c:
            raise ValueError("heating_setpoint_c must be < cooling_setpoint_c")
        return self
