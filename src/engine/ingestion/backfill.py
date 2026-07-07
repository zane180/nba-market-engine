"""Backfill orchestration: assemble the historical dataset from both APIs.

Ordering matters — games first (ESPN is the spine), then markets (joined to
games via deterministically constructed event tickers), then candles and
snapshots for exactly the markets/games we hold. Every step is idempotent:
re-running upserts the same rows.

Data-availability reality (verified 2026-07-06): Kalshi's public API retains
settled-market prices only ~2 months back, so the market side of the dataset
starts at whatever Kalshi still serves (currently the 2026 playoffs) and grows
forward as the recorder (Phase 6) captures live snapshots. ESPN history is
deep; the model-training side is not constrained.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import structlog

from engine.data.models import Game, GameStatus, MarketInfo, MarketResult
from engine.data.store import Store
from engine.ingestion.espn import EspnClient
from engine.ingestion.kalshi import KalshiPublicClient, KalshiSchemaError
from engine.ingestion.mapping import (
    UnmappedTeamError,
    game_event_ticker,
    team_from_kalshi_abbrev,
    teams_from_event_ticker,
)

logger = structlog.get_logger(__name__)

PRESEASON = 1


@dataclass
class BackfillStats:
    games_upserted: int = 0
    dates_fetched: int = 0
    games_skipped_preseason: int = 0
    markets_upserted: int = 0
    markets_unmatched: list[str] = field(default_factory=list)
    result_mismatches: list[str] = field(default_factory=list)
    candles_upserted: int = 0
    snapshots_upserted: int = 0
    games_snapshotted: int = 0
    games_snapshot_failed: int = 0


async def backfill_games(
    espn: EspnClient,
    store: Store,
    *,
    start: date,
    end: date,
    concurrency: int = 4,
) -> BackfillStats:
    """Upsert every NBA game (regular season + playoffs) in [start, end]."""
    stats = BackfillStats()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(day: date) -> list[Game]:
        async with semaphore:
            return await espn.scoreboard(day)

    for chunk_start in range(0, len(days), 50):
        chunk = days[chunk_start : chunk_start + 50]
        for games in await asyncio.gather(*(fetch(d) for d in chunk)):
            stats.dates_fetched += 1
            keep = [g for g in games if g.season_type != PRESEASON]
            stats.games_skipped_preseason += len(games) - len(keep)
            stats.games_upserted += store.upsert_games(keep)
        logger.info(
            "games backfill progress",
            through=chunk[-1].isoformat(),
            games=stats.games_upserted,
        )
    return stats


def _event_ticker_index(store: Store) -> dict[str, str]:
    """event ticker -> game_id for every stored game whose teams are mappable."""
    index: dict[str, str] = {}
    for game in store.games():
        try:
            ticker = game_event_ticker(
                away_espn_name=game.away_team,
                home_espn_name=game.home_team,
                tipoff=game.start_time,
            )
        except UnmappedTeamError:
            continue  # e.g. All-Star exhibition teams — no Kalshi market exists
        index[ticker] = game.game_id
    return index


async def backfill_markets(
    kalshi: KalshiPublicClient, store: Store, *, status: str = "settled"
) -> BackfillStats:
    """Pull every NBA game market Kalshi still serves and join it to our games.

    Also cross-checks Kalshi's settlement against ESPN's final score — a
    disagreement means one source is wrong about who won, which would poison
    every downstream evaluation, so it's surfaced instead of stored silently.
    """
    stats = BackfillStats()
    ticker_to_game = _event_ticker_index(store)
    infos: list[MarketInfo] = []

    for market in await kalshi.markets(status=status):
        ticker = market["ticker"]
        event_ticker = market["event_ticker"]
        try:
            teams_from_event_ticker(event_ticker)  # validates shape; raises on drift
            yes_team = team_from_kalshi_abbrev(ticker.rsplit("-", 1)[1]).espn_name
        except UnmappedTeamError as exc:
            raise KalshiSchemaError(f"market {ticker}: {exc}") from exc

        game_id = ticker_to_game.get(event_ticker)
        if game_id is None:
            stats.markets_unmatched.append(ticker)

        raw_result = market.get("result") or None
        result = MarketResult(raw_result) if raw_result in ("yes", "no") else None
        infos.append(
            MarketInfo(
                ticker=ticker,
                event_ticker=event_ticker,
                game_id=game_id,
                yes_team=yes_team,
                result=result,
                open_time=datetime.fromisoformat(market["open_time"]),
                close_time=datetime.fromisoformat(market["close_time"]),
                volume=float(market.get("volume_fp") or 0),
            )
        )

        if game_id is not None and result is not None:
            game = store.game(game_id)
            if game is not None and game.status is GameStatus.FINAL:
                yes_won = (yes_team == game.home_team) == game.home_won
                expected = MarketResult.YES if yes_won else MarketResult.NO
                if expected is not result:
                    stats.result_mismatches.append(
                        f"{ticker}: Kalshi settled {result.value}, ESPN says "
                        f"{game.home_team} {game.home_score}-{game.away_score} {game.away_team}"
                    )

    stats.markets_upserted = store.upsert_markets(infos)
    if stats.markets_unmatched:
        logger.warning("markets without a stored game", tickers=stats.markets_unmatched)
    for mismatch in stats.result_mismatches:
        logger.error("settlement mismatch", detail=mismatch)
    return stats


MAX_CANDLES_PER_REQUEST = 4800  # Kalshi caps at 5000 periods; stay under


async def backfill_candles(
    kalshi: KalshiPublicClient,
    store: Store,
    *,
    pre_tipoff: timedelta = timedelta(hours=24),
    period_seconds: int = 60,
) -> BackfillStats:
    """One-minute price history for every stored market that matched a game,
    from ``max(open_time, tipoff - pre_tipoff)`` through market close."""
    stats = BackfillStats()
    for info in store.markets(with_game_only=True):
        assert info.game_id is not None  # with_game_only
        game = store.game(info.game_id)
        if game is None:
            continue
        start = max(info.open_time, game.start_time - pre_tipoff)
        end = info.close_time
        window = timedelta(seconds=MAX_CANDLES_PER_REQUEST * period_seconds)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + window, end)
            candles = await kalshi.candlesticks(
                info.ticker,
                start=cursor,
                end=chunk_end,
                period_seconds=period_seconds,
                market_settled=info.result is not None,
            )
            stats.candles_upserted += store.upsert_candles(candles)
            cursor = chunk_end
        logger.info("candles backfilled", ticker=info.ticker, total=stats.candles_upserted)
    return stats


async def backfill_snapshots(
    espn: EspnClient,
    store: Store,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    only_with_markets: bool = True,
    skip_existing: bool = True,
    concurrency: int = 4,
) -> BackfillStats:
    """Play-by-play snapshots for stored FINAL games (the live model's training
    rows). Defaults to only games that have market data; pass
    ``only_with_markets=False`` for the deep-history training backfill.

    Bulk mode is failure-tolerant per game: one historical game with a broken
    summary shouldn't kill a 5,000-game run. Skipped games are counted and
    logged, never silently absorbed.
    """
    from engine.ingestion.espn import SchemaDriftError
    from engine.ingestion.http import UpstreamError

    stats = BackfillStats()
    final_games = store.games(start=start, end=end, statuses=[GameStatus.FINAL])
    if only_with_markets:
        with_markets = {m.game_id for m in store.markets(with_game_only=True)}
        final_games = [g for g in final_games if g.game_id in with_markets]
    if skip_existing:
        done = store.game_ids_with_snapshots()
        final_games = [g for g in final_games if g.game_id not in done]

    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(game: Game) -> int | None:
        async with semaphore:
            try:
                snapshots = await espn.game_snapshots(game.game_id)
            except (SchemaDriftError, UpstreamError) as exc:
                logger.warning(
                    "snapshot backfill skipped game", game_id=game.game_id, error=str(exc)
                )
                return None
        return store.upsert_snapshots(snapshots)

    for chunk_start in range(0, len(final_games), 200):
        chunk = final_games[chunk_start : chunk_start + 200]
        for count in await asyncio.gather(*(fetch(g) for g in chunk)):
            if count is None:
                stats.games_snapshot_failed += 1
            else:
                stats.snapshots_upserted += count
                stats.games_snapshotted += 1
        logger.info(
            "snapshot backfill progress",
            games=stats.games_snapshotted,
            snapshots=stats.snapshots_upserted,
            failed=stats.games_snapshot_failed,
        )
    return stats


@dataclass
class VerificationReport:
    counts: dict[str, int]
    markets_without_game: list[str]
    settlement_mismatches: list[str]
    markets_without_candles: list[str]
    unsettled_markets: list[str]

    @property
    def ok(self) -> bool:
        return not self.settlement_mismatches

    def lines(self) -> list[str]:
        out = [f"{name}: {count} rows" for name, count in self.counts.items()]
        out.append(f"markets unmatched to a game: {len(self.markets_without_game)}")
        out.append(f"markets with no candles: {len(self.markets_without_candles)}")
        out.append(f"markets not yet settled: {len(self.unsettled_markets)}")
        out.append(f"settlement mismatches vs ESPN: {len(self.settlement_mismatches)}")
        out.extend(f"  MISMATCH {m}" for m in self.settlement_mismatches)
        return out


def verify_dataset(store: Store) -> VerificationReport:
    """Integrity checks over the joined dataset. The hard failure is a
    settlement disagreeing with ESPN's final score; everything else is
    reported as context."""
    mismatches: list[str] = []
    without_game: list[str] = []
    unsettled: list[str] = []
    candle_counts = store.candle_count_by_ticker()
    without_candles: list[str] = []

    for info in store.markets():
        if info.game_id is None:
            without_game.append(info.ticker)
            continue
        if info.result is None:
            unsettled.append(info.ticker)
        if candle_counts.get(info.ticker, 0) == 0:
            without_candles.append(info.ticker)
        game = store.game(info.game_id)
        if game is None or game.status is not GameStatus.FINAL or info.result is None:
            continue
        yes_won = (info.yes_team == game.home_team) == game.home_won
        expected = MarketResult.YES if yes_won else MarketResult.NO
        if expected is not info.result:
            mismatches.append(
                f"{info.ticker}: Kalshi={info.result.value}, ESPN final "
                f"{game.home_team} {game.home_score}-{game.away_score} {game.away_team}"
            )

    return VerificationReport(
        counts=store.counts(),
        markets_without_game=without_game,
        settlement_mismatches=mismatches,
        markets_without_candles=without_candles,
        unsettled_markets=unsettled,
    )


__all__ = [
    "BackfillStats",
    "VerificationReport",
    "backfill_candles",
    "backfill_games",
    "backfill_markets",
    "backfill_snapshots",
    "verify_dataset",
]
