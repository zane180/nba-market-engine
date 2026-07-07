"""Typer CLI entrypoints. Each command maps to one pipeline stage:
ingest (backfill + verify), report (calibration evaluations), backtest,
paper (the live loop), and config."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import typer

from engine.config import load_settings
from engine.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from engine.data.store import Store
    from engine.ingestion.espn import EspnClient
    from engine.ingestion.kalshi import KalshiPublicClient

app = typer.Typer(
    name="engine",
    help="NBA win-probability engine vs. Kalshi markets.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v"), json_logs: bool = False) -> None:
    configure_logging(json_output=json_logs, level=logging.DEBUG if verbose else logging.INFO)


ingest_app = typer.Typer(
    name="ingest",
    help="Backfill and verify the historical dataset.",
    no_args_is_help=True,
)
app.add_typer(ingest_app)


def _store() -> Store:
    from pathlib import Path

    from engine.data.store import Store

    settings = load_settings()
    if settings.database_url.startswith("sqlite:///"):
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    store = Store(settings.database_url)
    store.init_schema()
    return store


@asynccontextmanager
async def _clients() -> AsyncIterator[tuple[Store, EspnClient, KalshiPublicClient]]:
    from pathlib import Path

    import httpx

    from engine.ingestion.espn import EspnClient
    from engine.ingestion.http import FileCache, RetryingClient
    from engine.ingestion.kalshi import KalshiPublicClient

    settings = load_settings()
    store = _store()
    async with httpx.AsyncClient(timeout=30) as client:
        http = RetryingClient(client, cache=FileCache(Path("data/http_cache")))
        yield (
            store,
            EspnClient(http, settings.espn_api_base),
            KalshiPublicClient(http, settings.kalshi_api_base),
        )


@ingest_app.command("games")
def ingest_games(
    start: str = typer.Option("2022-10-01", help="first date, YYYY-MM-DD"),
    end: str = typer.Option(None, help="last date, YYYY-MM-DD (default: today)"),
) -> None:
    """Backfill NBA games (regular season + playoffs) from ESPN."""
    import asyncio
    import datetime as dt

    from engine.ingestion.backfill import backfill_games

    async def run() -> None:
        async with _clients() as (store, espn, _kalshi):
            stats = await backfill_games(
                espn,
                store,
                start=dt.date.fromisoformat(start),
                end=dt.date.fromisoformat(end) if end else dt.datetime.now(dt.UTC).date(),
            )
        typer.echo(
            f"games: {stats.games_upserted} upserted over {stats.dates_fetched} dates "
            f"({stats.games_skipped_preseason} preseason skipped)"
        )

    asyncio.run(run())


@ingest_app.command("markets")
def ingest_markets() -> None:
    """Backfill every NBA market Kalshi still serves; cross-check settlements."""
    import asyncio

    from engine.ingestion.backfill import backfill_markets

    async def run() -> None:
        async with _clients() as (store, _espn, kalshi):
            stats = await backfill_markets(kalshi, store)
        typer.echo(
            f"markets: {stats.markets_upserted} upserted, "
            f"{len(stats.markets_unmatched)} unmatched to games, "
            f"{len(stats.result_mismatches)} settlement mismatches"
        )
        if stats.result_mismatches:
            raise typer.Exit(1)

    asyncio.run(run())


@ingest_app.command("candles")
def ingest_candles(
    pre_tipoff_hours: int = typer.Option(24, help="history to keep before tipoff"),
) -> None:
    """Backfill 1-minute price candles for every market matched to a game."""
    import asyncio
    import datetime as dt

    from engine.ingestion.backfill import backfill_candles

    async def run() -> None:
        async with _clients() as (store, _espn, kalshi):
            stats = await backfill_candles(
                kalshi, store, pre_tipoff=dt.timedelta(hours=pre_tipoff_hours)
            )
        typer.echo(f"candles: {stats.candles_upserted} upserted")

    asyncio.run(run())


@ingest_app.command("snapshots")
def ingest_snapshots(
    all_games: bool = typer.Option(
        False, "--all-games", help="every stored final game, not just those with markets"
    ),
) -> None:
    """Backfill play-by-play snapshots for stored final games."""
    import asyncio

    from engine.ingestion.backfill import backfill_snapshots

    async def run() -> None:
        async with _clients() as (store, espn, _kalshi):
            stats = await backfill_snapshots(espn, store, only_with_markets=not all_games)
        typer.echo(
            f"snapshots: {stats.snapshots_upserted} upserted across "
            f"{stats.games_snapshotted} games ({stats.games_snapshot_failed} games failed)"
        )

    asyncio.run(run())


@ingest_app.command("verify")
def ingest_verify() -> None:
    """Integrity-check the dataset; exits 1 on settlement mismatches."""
    from engine.ingestion.backfill import verify_dataset

    store = _store()
    report = verify_dataset(store)
    for line in report.lines():
        typer.echo(line)
    if not report.ok:
        raise typer.Exit(1)


@ingest_app.command("all")
def ingest_all(
    start: str = typer.Option("2022-10-01", help="first game date, YYYY-MM-DD"),
) -> None:
    """Full pipeline: games -> markets -> candles -> snapshots -> verify."""
    import asyncio
    import datetime as dt

    from engine.ingestion.backfill import (
        backfill_candles,
        backfill_games,
        backfill_markets,
        backfill_snapshots,
        verify_dataset,
    )

    async def run() -> None:
        async with _clients() as (store, espn, kalshi):
            g = await backfill_games(
                espn, store, start=dt.date.fromisoformat(start), end=dt.datetime.now(dt.UTC).date()
            )
            typer.echo(f"games: {g.games_upserted}")
            m = await backfill_markets(kalshi, store)
            typer.echo(f"markets: {m.markets_upserted} ({len(m.markets_unmatched)} unmatched)")
            c = await backfill_candles(kalshi, store)
            typer.echo(f"candles: {c.candles_upserted}")
            s = await backfill_snapshots(espn, store)
            typer.echo(f"snapshots: {s.snapshots_upserted} across {s.games_snapshotted} games")
            report = verify_dataset(store)
            for line in report.lines():
                typer.echo(line)
            if not report.ok:
                raise typer.Exit(1)

    asyncio.run(run())


report_app = typer.Typer(
    name="report", help="Generate evaluation reports into reports/.", no_args_is_help=True
)
app.add_typer(report_app)


@report_app.command("pregame")
def report_pregame(
    first_test_season: int = typer.Option(
        2026, help="first season (end year) evaluated walk-forward"
    ),
) -> None:
    """Pre-game models vs. the de-vigged market: metrics + reliability diagram."""
    from pathlib import Path

    from engine.models.evaluation import run_pregame_evaluation

    settings = load_settings()
    evaluation = run_pregame_evaluation(
        _store(), first_test_season=first_test_season, seed=settings.random_seed
    )
    typer.echo(f"test seasons: {evaluation.test_seasons}")
    for block in evaluation.full_test:
        typer.echo(f"  [full]   {block.name}: brier={block.brier:.4f} n={block.n}")
    for block in evaluation.market_subset:
        typer.echo(f"  [market] {block.name}: brier={block.brier:.4f} n={block.n}")
    typer.echo(f"reports written to {Path('reports').resolve()}")


@report_app.command("live")
def report_live(
    first_test_season: int = typer.Option(
        2026, help="first season (end year) evaluated walk-forward"
    ),
) -> None:
    """Live WP model vs. the in-game de-vigged market: metrics + charts."""
    from engine.models.live_evaluation import run_live_evaluation

    settings = load_settings()
    ev = run_live_evaluation(
        _store(), first_test_season=first_test_season, seed=settings.random_seed
    )
    for name, n, brier, _ll, ece in ev.full_metrics:
        typer.echo(f"  [full]   {name}: brier={brier:.4f} ece={ece:.4f} n={n}")
    point, lo, hi = ev.diff_ci
    typer.echo(
        f"  [market] model {ev.model_brier:.4f} vs market {ev.market_brier:.4f} "
        f"on {ev.joined_n_snapshots} snapshots / {ev.joined_n_games} games; "
        f"diff {point:+.4f} CI [{lo:+.4f},{hi:+.4f}]"
    )


@app.command()
def backtest(
    first_test_season: int = typer.Option(2026, help="first test season (end year)"),
    bankroll: float = typer.Option(1000.0, help="initial bankroll in dollars"),
) -> None:
    """Walk-forward backtest with fees and sizing; writes reports/."""
    from engine.backtest.report import run_backtest

    settings = load_settings()
    report = run_backtest(
        _store(),
        first_test_season=first_test_season,
        initial_bankroll=bankroll,
        seed=settings.random_seed,
    )
    for name, result in (
        ("pregame        ", report.pregame),
        ("pregame no-fee ", report.pregame_no_fees),
        ("live           ", report.live),
        ("follow-market  ", report.follow_market),
    ):
        typer.echo(
            f"  {name} trades={len(result.trades):3d} pnl={result.total_pnl:+8.2f} "
            f"roi={result.roi:+7.2%} fees={result.total_fees:7.2f} maxDD={result.max_drawdown:.2%}"
        )


@app.command()
def paper(
    interval: float = typer.Option(60.0, help="seconds between ticks"),
    once: bool = typer.Option(False, "--once", help="run a single tick and exit"),
    bankroll: float = typer.Option(1000.0, help="paper bankroll (dollars)"),
) -> None:
    """Live paper-trading loop: ingest -> model -> edge -> logged would-be trades.

    The default and only enabled execution path. Records order-book snapshots
    on every tick; a tick with no NBA games is a healthy no-op.
    """
    import asyncio

    from engine.backtest.engine import StrategyParams
    from engine.execution.paper import PaperTrader
    from engine.pipeline.live_loop import LiveLoop, ModelBundle

    async def run() -> None:
        async with _clients() as (store, espn, kalshi):
            models = ModelBundle.fit_from_store(store)
            trader = PaperTrader(store, initial_bankroll=bankroll, params=StrategyParams())
            loop = LiveLoop(espn=espn, kalshi=kalshi, store=store, trader=trader, models=models)
            await loop.run(interval_seconds=interval, max_ticks=1 if once else None)
            typer.echo(
                f"paper bankroll: ${trader.bankroll:,.2f} "
                f"({len(store.paper_trades())} lifetime trades, "
                f"{len(store.paper_trades(open_only=True))} open)"
            )

    asyncio.run(run())


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
