"""Rich report generation for GRIDMIND simulation results."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gridmind.models import Building, EnergyProfile
from gridmind.optimizer.cost import CostOptimizer


def _fmt(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}"


def summary_table(profile: EnergyProfile) -> Table:
    """Build a Rich table summarising energy totals."""
    table = Table(title="Energy Summary", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Simulation hours", str(profile.hours))
    table.add_row("Total HVAC (kWh)", _fmt(sum(profile.hvac_kwh)))
    table.add_row("Total Lighting (kWh)", _fmt(sum(profile.lighting_kwh)))
    table.add_row("Total Energy (kWh)", _fmt(profile.total_energy_kwh))
    if profile.cost_usd:
        table.add_row("Total Cost ($)", _fmt(profile.total_cost))
    return table


def hourly_table(profile: EnergyProfile, every_n: int = 1) -> Table:
    """Build a Rich table of hourly data (optionally sampling every *every_n* hours)."""
    table = Table(title="Hourly Breakdown", show_lines=False)
    table.add_column("Hour", justify="right", style="bold")
    table.add_column("HVAC kWh", justify="right")
    table.add_column("Light kWh", justify="right")
    table.add_column("Total kWh", justify="right")
    if profile.cost_usd:
        table.add_column("Cost $", justify="right")
    if profile.zone_temperatures_c:
        table.add_column("Zone 1 Temp (C)", justify="right")

    for h in range(0, profile.hours, every_n):
        row = [
            str(h),
            _fmt(profile.hvac_kwh[h]),
            _fmt(profile.lighting_kwh[h]),
            _fmt(profile.total_kwh[h]),
        ]
        if profile.cost_usd:
            row.append(_fmt(profile.cost_usd[h], 4))
        if profile.zone_temperatures_c:
            row.append(_fmt(profile.zone_temperatures_c[0][h], 1))
        table.add_row(*row)

    return table


def print_report(
    profile: EnergyProfile,
    building: Building,
    cost_optimizer: CostOptimizer | None = None,
    console: Console | None = None,
) -> None:
    """Print a full Rich report to the console."""
    console = console or Console()

    # Attach cost if not already present
    if not profile.cost_usd and cost_optimizer is not None:
        energy = np.array(profile.total_kwh)
        profile.cost_usd = cost_optimizer.compute_cost(energy).tolist()

    console.print()
    console.print(
        Panel(
            f"[bold green]GRIDMIND Report[/bold green]  |  "
            f"Building: {building.name}  |  "
            f"Zones: {len(building.zones)}  |  "
            f"Hours: {profile.hours}",
            expand=False,
        )
    )
    console.print()
    console.print(summary_table(profile))
    console.print()

    step = max(1, profile.hours // 24)
    console.print(hourly_table(profile, every_n=step))
    console.print()
