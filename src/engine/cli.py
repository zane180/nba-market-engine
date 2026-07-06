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
def config() -> None:
    """Print effective (non-secret) configuration."""
    settings = load_settings()
    for key, value in settings.model_dump().items():
        # SecretStr renders as '**********'; never print raw secrets.
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()
