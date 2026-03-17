"""GRIDMIND CLI — Click entry-point."""

from __future__ import annotations

import numpy as np
import click
from rich.console import Console

from gridmind.models import Building, WeatherData
from gridmind.simulator import BuildingSimulator
from gridmind.optimizer.cost import CostOptimizer
from gridmind.optimizer.predictor import DemandPredictor
from gridmind.optimizer.scheduler import EnergyScheduler
from gridmind.report import print_report

console = Console()


def _sample_weather(hours: int) -> WeatherData:
    """Generate synthetic weather data for demonstration."""
    h = np.arange(hours)
    # Sinusoidal outdoor temperature: peak at 15:00
    temp = 15.0 + 10.0 * np.sin(2 * np.pi * (h - 6) / 24)
    # Solar irradiance: bell curve 06:00-18:00
    solar = np.clip(800.0 * np.sin(np.pi * ((h % 24) - 6) / 12), 0, None)
    return WeatherData(
        outdoor_temp_c=temp.tolist(),
        solar_irradiance_w_m2=solar.tolist(),
    )


@click.group()
@click.version_option(package_name="gridmind")
def main() -> None:
    """GRIDMIND -- Building Energy Optimizer."""


@main.command()
@click.option("--hours", default=24, show_default=True, help="Simulation length in hours.")
def simulate(hours: int) -> None:
    """Run a thermal simulation with the default building."""
    building = Building()
    weather = _sample_weather(hours)
    sim = BuildingSimulator(building, weather)

    with console.status("[bold green]Simulating..."):
        profile = sim.run()

    cost_opt = CostOptimizer()
    print_report(profile, building, cost_optimizer=cost_opt, console=console)


@main.command()
@click.option("--hours", default=24, show_default=True, help="Simulation length in hours.")
def optimize(hours: int) -> None:
    """Predict demand, optimise schedule, and show savings."""
    building = Building()
    weather = _sample_weather(hours)
    cost_opt = CostOptimizer()

    predictor = DemandPredictor(building)
    baseline = predictor.predict(weather)

    scheduler = EnergyScheduler(cost_opt)
    optimized = scheduler.optimize(baseline)

    baseline_cost = cost_opt.total_cost(baseline)
    opt_cost = cost_opt.total_cost(optimized)
    savings = baseline_cost - opt_cost

    console.print()
    console.print(f"[bold cyan]Baseline cost:[/]  ${baseline_cost:,.2f}")
    console.print(f"[bold green]Optimized cost:[/] ${opt_cost:,.2f}")
    console.print(f"[bold yellow]Savings:[/]        ${savings:,.2f}  "
                  f"({savings / baseline_cost * 100:.1f}%)" if baseline_cost > 0 else "")
    console.print()


@main.command()
@click.option("--hours", default=24, show_default=True, help="Simulation length in hours.")
def report(hours: int) -> None:
    """Run simulation and generate a detailed report."""
    building = Building()
    weather = _sample_weather(hours)
    sim = BuildingSimulator(building, weather)

    with console.status("[bold green]Simulating..."):
        profile = sim.run()

    cost_opt = CostOptimizer()
    print_report(profile, building, cost_optimizer=cost_opt, console=console)


if __name__ == "__main__":
    main()
