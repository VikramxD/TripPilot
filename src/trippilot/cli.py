"""
CLI interface for TripPilot.

Provides command-line access to trip planning and other features.
"""

import asyncio
import json
from datetime import date
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

from trippilot import __version__
from trippilot.core.config import settings
from trippilot.core.orchestrator import TripPilotOrchestrator
from trippilot.schemas.travel import BudgetLevel, TravelPreferences, TravelQuery, TravelStyle
from trippilot.utils.logging import setup_logging

app = typer.Typer(
    name="trippilot",
    help="AI-powered travel companion with multi-agent architecture",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"TripPilot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """TripPilot - AI-powered travel companion."""
    setup_logging(level="DEBUG" if debug else "INFO")


@app.command()
def plan(
    destination: str = typer.Argument(..., help="Destination city or region"),
    days: int = typer.Option(5, "--days", "-n", help="Trip duration in days"),
    travelers: int = typer.Option(1, "--travelers", "-t", help="Number of travelers"),
    budget: str = typer.Option("moderate", "--budget", "-b", help="Budget level (budget/moderate/premium/luxury)"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Travel style (adventure/relaxation/cultural/foodie)"),
    interests: Optional[str] = typer.Option(None, "--interests", "-i", help="Comma-separated interests"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
):
    """
    Plan a complete trip with AI agents.

    Example:
        trippilot plan "Tokyo, Japan" --days 7 --budget moderate --style cultural
    """
    console.print(Panel.fit(
        f"[bold blue]Planning trip to {destination}[/bold blue]\n"
        f"Duration: {days} days | Travelers: {travelers} | Budget: {budget}",
        title="TripPilot",
    ))

    # Parse options
    try:
        budget_level = BudgetLevel(budget.lower())
    except ValueError:
        console.print(f"[red]Invalid budget level: {budget}[/red]")
        raise typer.Exit(1)

    styles = []
    if style:
        try:
            styles = [TravelStyle(s.strip().lower()) for s in style.split(",")]
        except ValueError as e:
            console.print(f"[red]Invalid travel style: {e}[/red]")
            raise typer.Exit(1)

    interest_list = [i.strip() for i in interests.split(",")] if interests else []

    # Create query
    query = TravelQuery(
        destination=destination,
        duration_days=days,
        travelers=travelers,
        preferences=TravelPreferences(
            styles=styles or [TravelStyle.CULTURAL],
            budget_level=budget_level,
            interests=interest_list,
        ),
    )

    # Run planning
    async def run_planning():
        orchestrator = TripPilotOrchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Planning your trip...", total=None)

            result = await orchestrator.plan_trip(query)

            progress.update(task, completed=True)

        return result

    result = asyncio.run(run_planning())

    if not result.success:
        console.print(f"[red]Planning failed: {result.error}[/red]")
        raise typer.Exit(1)

    # Display results
    rec = result.recommendation
    if rec and rec.itinerary:
        itinerary = rec.itinerary

        # Overview
        console.print("\n")
        console.print(Panel(
            Markdown(f"# {itinerary.title}\n\n{itinerary.overview}"),
            title="Trip Overview",
        ))

        # Highlights
        if itinerary.highlights:
            console.print("\n[bold]Highlights:[/bold]")
            for h in itinerary.highlights:
                console.print(f"  - {h}")

        # Daily plans
        if itinerary.daily_plans:
            console.print("\n")
            for day in itinerary.daily_plans:
                table = Table(title=f"Day {day.day_number}: {day.title}")
                table.add_column("Time", style="cyan")
                table.add_column("Activity", style="white")

                for activity in day.morning:
                    table.add_row("Morning", f"{activity.name} - {activity.description[:50]}...")
                for activity in day.afternoon:
                    table.add_row("Afternoon", f"{activity.name} - {activity.description[:50]}...")
                for activity in day.evening:
                    table.add_row("Evening", f"{activity.name} - {activity.description[:50]}...")

                if day.tips:
                    table.add_row("Tips", ", ".join(day.tips[:2]))

                console.print(table)
                console.print()

        # Budget
        if itinerary.budget:
            budget_table = Table(title="Budget Estimate")
            budget_table.add_column("Category")
            budget_table.add_column("Amount (USD)", justify="right")

            breakdown = itinerary.budget.breakdown
            budget_table.add_row("Accommodation", f"${breakdown.accommodation:,.0f}")
            budget_table.add_row("Food", f"${breakdown.food:,.0f}")
            budget_table.add_row("Transportation", f"${breakdown.transportation:,.0f}")
            budget_table.add_row("Activities", f"${breakdown.activities:,.0f}")
            budget_table.add_row("Miscellaneous", f"${breakdown.miscellaneous:,.0f}")
            budget_table.add_row("[bold]Total[/bold]", f"[bold]${itinerary.budget.total_estimated:,.0f}[/bold]")

            console.print(budget_table)

        # Save to file if requested
        if output:
            with open(output, "w") as f:
                json.dump(rec.model_dump(), f, indent=2, default=str)
            console.print(f"\n[green]Saved to {output}[/green]")

    # Metrics
    console.print(f"\n[dim]Execution time: {result.total_execution_time_ms:.0f}ms | Tokens used: {result.total_tokens_used}[/dim]")


@app.command()
def research(
    destination: str = typer.Argument(..., help="Destination to research"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
):
    """
    Quick research on a destination.

    Example:
        trippilot research "Barcelona, Spain"
    """
    console.print(f"[bold]Researching {destination}...[/bold]\n")

    async def run_research():
        orchestrator = TripPilotOrchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Researching...", total=None)
            result = await orchestrator.quick_research(destination)
            progress.update(task, completed=True)

        return result

    result = asyncio.run(run_research())

    if not result.success:
        console.print(f"[red]Research failed: {result.error}[/red]")
        raise typer.Exit(1)

    # Display results
    data = result.data
    if data:
        if "destination" in data:
            dest = data["destination"]
            console.print(Panel(
                f"[bold]{dest.get('name', destination)}[/bold]\n"
                f"{dest.get('description', '')}",
                title="Destination Overview",
            ))

        if "attractions" in data:
            console.print("\n[bold]Top Attractions:[/bold]")
            for a in data["attractions"][:5]:
                console.print(f"  - {a.get('name', '')}: {a.get('description', '')[:60]}...")

        if "practical_tips" in data:
            console.print("\n[bold]Practical Tips:[/bold]")
            for tip in data["practical_tips"][:5]:
                console.print(f"  - {tip}")

    if output:
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command()
def budget(
    destination: str = typer.Argument(..., help="Destination"),
    days: int = typer.Option(5, "--days", "-n", help="Trip duration"),
    travelers: int = typer.Option(1, "--travelers", "-t", help="Number of travelers"),
    level: str = typer.Option("moderate", "--level", "-l", help="Budget level"),
):
    """
    Estimate budget for a trip.

    Example:
        trippilot budget "Paris, France" --days 7 --level luxury
    """
    console.print(f"[bold]Estimating budget for {destination}...[/bold]\n")

    try:
        budget_level = BudgetLevel(level.lower())
    except ValueError:
        console.print(f"[red]Invalid budget level: {level}[/red]")
        raise typer.Exit(1)

    async def run_budget():
        orchestrator = TripPilotOrchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Calculating...", total=None)

            query = TravelQuery(
                destination=destination,
                duration_days=days,
                travelers=travelers,
                preferences=TravelPreferences(budget_level=budget_level),
            )
            result = await orchestrator.estimate_budget(query=query)
            progress.update(task, completed=True)

        return result

    result = asyncio.run(run_budget())

    if not result.success:
        console.print(f"[red]Budget estimation failed: {result.error}[/red]")
        raise typer.Exit(1)

    budget_data = result.data
    if budget_data:
        table = Table(title=f"Budget Estimate: {days} days in {destination}")
        table.add_column("Category")
        table.add_column("Amount (USD)", justify="right")

        breakdown = budget_data.breakdown
        table.add_row("Accommodation", f"${breakdown.accommodation:,.0f}")
        table.add_row("Food", f"${breakdown.food:,.0f}")
        table.add_row("Transportation", f"${breakdown.transportation:,.0f}")
        table.add_row("Activities", f"${breakdown.activities:,.0f}")
        table.add_row("Flights", f"${breakdown.flights:,.0f}")
        table.add_row("Miscellaneous", f"${breakdown.miscellaneous:,.0f}")
        table.add_row("[bold]Total[/bold]", f"[bold]${budget_data.total_estimated:,.0f}[/bold]")

        if budget_data.daily_average:
            table.add_row("Daily Average", f"${budget_data.daily_average:,.0f}")

        console.print(table)

        if budget_data.money_saving_tips:
            console.print("\n[bold]Money-Saving Tips:[/bold]")
            for tip in budget_data.money_saving_tips[:5]:
                console.print(f"  - {tip}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """
    Start the TripPilot API server.

    Example:
        trippilot serve --port 8000 --reload
    """
    import uvicorn

    console.print(Panel.fit(
        f"[bold green]Starting TripPilot API Server[/bold green]\n"
        f"Host: {host} | Port: {port}",
        title="TripPilot",
    ))

    uvicorn.run(
        "trippilot.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()
