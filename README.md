# GRIDMIND — Building Energy Optimizer

GRIDMIND simulates building thermal dynamics and optimizes energy consumption
across HVAC, lighting, and envelope systems. It uses physics-based thermal
models, occupancy-aware scheduling, weather-driven demand prediction, and
time-of-use cost optimization to minimize energy spend while maintaining
occupant comfort.

## Features

- **Thermal simulation** — RC-network model of building zones with solar gain,
  internal loads, and inter-zone heat transfer.
- **HVAC optimization** — temperature setpoint scheduling with dead-band
  control and pre-conditioning strategies.
- **Lighting control** — occupancy and daylight-based dimming with smooth
  transitions.
- **Demand prediction** — forecast hourly energy demand from weather and
  occupancy profiles.
- **Cost optimization** — minimize electricity cost under time-of-use tariffs
  with optional peak-demand charges.
- **Rich CLI** — interactive reports, tables, and progress bars via Click and
  Rich.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Run a 24-hour simulation with default building
gridmind simulate --hours 24

# Optimize energy schedule
gridmind optimize --hours 24

# Generate a cost report
gridmind report --hours 24
```

## Project layout

```
src/gridmind/
    cli.py              CLI entry-point (Click)
    models.py           Pydantic data models
    simulator.py        BuildingSimulator — thermal dynamics engine
    report.py           Rich report generation
    building/
        hvac.py         HVACOptimizer — setpoints & scheduling
        lighting.py     LightingOptimizer — occupancy-based control
        envelope.py     BuildingEnvelope — thermal resistance/capacitance
    optimizer/
        scheduler.py    EnergyScheduler — hourly optimization
        predictor.py    DemandPredictor — weather + occupancy forecasting
        cost.py         CostOptimizer — time-of-use pricing
tests/
    test_models.py
    test_simulator.py
    test_hvac.py
    test_lighting.py
    test_envelope.py
    test_scheduler.py
    test_predictor.py
    test_cost.py
```

## Thermal model

Each zone is modelled as a lumped-capacitance (RC) node:

```
C_z * dT_z/dt = Q_hvac + Q_solar + Q_internal + Q_infiltration
                + sum_j [ (T_j - T_z) / R_ij ]
                + (T_outside - T_z) / R_env
```

where `C_z` is the zone thermal capacitance (J/K), `R_env` is the envelope
resistance (K/W), and `Q` terms represent HVAC, solar, internal, and
infiltration heat flows.

## Author

Mukunda Katta

## License

MIT
