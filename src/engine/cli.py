"""Typer CLI entrypoints. Each command maps to one pipeline stage; commands whose
phase isn't built yet exit with code 2 and say so, rather than pretending.
"""

from __future__ import annotations

import logging

import typer

from engine.config import load_settings
from engine.logging import configure_logging

app = typer.Typer(
    name="engine",
    help="NBA win-probability engine vs. Kalshi markets.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

_NOT_BUILT = 2


def _not_built(phase: str) -> None:
    typer.echo(f"not implemented yet — arrives in {phase}", err=True)
    raise typer.Exit(_NOT_BUILT)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v"), json_logs: bool = False) -> None:
    configure_logging(json_output=json_logs, level=logging.DEBUG if verbose else logging.INFO)


@app.command()
def ingest() -> None:
    """Backfill historical games and market data."""
    _not_built("Phase 1-2 (ingestion + storage)")


@app.command()
def backtest() -> None:
    """Run the walk-forward backtest and regenerate reports/."""
    _not_built("Phase 5 (backtest engine)")


@app.command()
def paper() -> None:
    """Run the live paper-trading loop (the default and only execution path)."""
    _not_built("Phase 6 (paper trading)")


@app.command()
def probe(date: str = typer.Option("2026-01-15", help="past date to probe, YYYY-MM-DD")) -> None:
    """Live health check: verify both upstream APIs still match our parsers.

    Fetches a real ESPN scoreboard, maps each game to its Kalshi event ticker,
    and confirms the event exists on Kalshi. Exits nonzero on schema drift.
    """
    import asyncio
    import datetime as dt

    import httpx

    from engine.ingestion.espn import EspnClient
    from engine.ingestion.http import RetryingClient
    from engine.ingestion.kalshi import KalshiPublicClient
    from engine.ingestion.mapping import game_event_ticker

    settings = load_settings()
    day = dt.date.fromisoformat(date)

    async def run() -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            http = RetryingClient(client)
            espn = EspnClient(http, settings.espn_api_base)
            kalshi = KalshiPublicClient(http, settings.kalshi_api_base)

            games = await espn.scoreboard(day)
            typer.echo(f"ESPN: {len(games)} games on {day}")
            checked = 0
            for game in games[:3]:
                ticker = game_event_ticker(
                    away_espn_name=game.away_team,
                    home_espn_name=game.home_team,
                    tipoff=game.start_time,
                )
                event = await kalshi.event(ticker)
                title = event.get("event", {}).get("title", "?")
                typer.echo(
                    f"  {game.away_team} @ {game.home_team} ({game.status}) "
                    f"-> {ticker} -> Kalshi: {title!r}"
                )
                checked += 1
            typer.echo(f"OK: {checked} games cross-verified against Kalshi events")

    asyncio.run(run())


@app.command()
def config() -> None:
    """Print effective (non-secret) configuration."""
    settings = load_settings()
    for key, value in settings.model_dump().items():
        # SecretStr renders as '**********'; never print raw secrets.
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()
