"""Backfill orchestration logic against fake clients: the market<->game join,
the settlement cross-check, preseason filtering, and idempotency."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from engine.data.models import Game, GameStatus, LiveGameState
from engine.data.store import Store
from engine.ingestion.backfill import (
    backfill_candles,
    backfill_games,
    backfill_markets,
    backfill_snapshots,
    verify_dataset,
)
from engine.ingestion.kalshi import KalshiSchemaError

TIP = datetime(2026, 6, 14, 0, 30, tzinfo=UTC)  # Jun 13 evening ET


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(f"sqlite:///{tmp_path}/test.db")
    s.init_schema()
    return s


def final_game(**overrides: object) -> Game:
    base: dict[str, object] = {
        "game_id": "g-finals-5",
        "home_team": "San Antonio Spurs",
        "away_team": "New York Knicks",
        "start_time": TIP,
        "status": GameStatus.FINAL,
        "home_score": 100,
        "away_score": 108,  # Knicks won -> SAS market settles NO
        "season_type": 3,
    }
    return Game(**{**base, **overrides})  # type: ignore[arg-type]


def kalshi_market(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ticker": "KXNBAGAME-26JUN13NYKSAS-SAS",
        "event_ticker": "KXNBAGAME-26JUN13NYKSAS",
        "result": "no",
        "open_time": "2026-06-09T03:55:00+00:00",
        "close_time": "2026-06-14T03:31:47+00:00",
        "volume_fp": "48467702.82",
    }
    return {**base, **overrides}


class FakeEspn:
    def __init__(
        self,
        by_date: dict[date, list[Game]],
        snapshots: dict[str, list[LiveGameState]] | None = None,
    ) -> None:
        self.by_date = by_date
        self.snaps = snapshots or {}
        self.snapshot_requests: list[str] = []

    async def scoreboard(self, day: date | None = None) -> list[Game]:
        assert day is not None
        return self.by_date.get(day, [])

    async def game_snapshots(self, game_id: str, *, final: bool = False) -> list[LiveGameState]:
        self.snapshot_requests.append(game_id)
        return self.snaps.get(game_id, [])


class FakeKalshi:
    def __init__(self, markets: list[dict[str, Any]], candles_per_call: int = 2) -> None:
        self._markets = markets
        self.candle_windows: list[tuple[str, datetime, datetime]] = []
        self._candles_per_call = candles_per_call

    async def markets(self, *, status: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return self._markets

    async def candlesticks(
        self,
        ticker: str,
        *,
        start: datetime,
        end: datetime,
        period_seconds: int = 60,
        market_settled: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        from engine.data.models import Candle

        self.candle_windows.append((ticker, start, end))
        return [
            Candle(
                ticker=ticker,
                end_time=start + timedelta(minutes=i + 1),
                period_seconds=period_seconds,
                yes_bid_close=40 + i,
                yes_ask_close=42 + i,
                trade_close=41,
                volume=10.0,
                open_interest=100.0,
            )
            for i in range(self._candles_per_call)
        ]


class TestBackfillGames:
    async def test_upserts_and_skips_preseason(self, store: Store) -> None:
        day = date(2026, 6, 13)
        espn = FakeEspn(
            {
                day: [
                    final_game(),
                    final_game(game_id="pre", season_type=1),  # preseason -> skipped
                ]
            }
        )
        stats = await backfill_games(espn, store, start=day, end=day)  # type: ignore[arg-type]
        assert stats.games_upserted == 1
        assert stats.games_skipped_preseason == 1
        assert store.counts()["games"] == 1

    async def test_rerun_is_idempotent(self, store: Store) -> None:
        day = date(2026, 6, 13)
        espn = FakeEspn({day: [final_game()]})
        await backfill_games(espn, store, start=day, end=day)  # type: ignore[arg-type]
        await backfill_games(espn, store, start=day, end=day)  # type: ignore[arg-type]
        assert store.counts()["games"] == 1


class TestBackfillMarkets:
    async def test_market_joins_to_game_by_constructed_ticker(self, store: Store) -> None:
        store.upsert_games([final_game()])
        stats = await backfill_markets(FakeKalshi([kalshi_market()]), store)  # type: ignore[arg-type]
        assert stats.markets_upserted == 1
        assert stats.markets_unmatched == []
        assert stats.result_mismatches == []
        info = store.markets()[0]
        assert info.game_id == "g-finals-5"
        assert info.yes_team == "San Antonio Spurs"

    async def test_settlement_mismatch_is_caught(self, store: Store) -> None:
        """Kalshi says SAS won (yes) but ESPN's final score says they lost —
        exactly the poison the cross-check exists for."""
        store.upsert_games([final_game()])
        stats = await backfill_markets(FakeKalshi([kalshi_market(result="yes")]), store)  # type: ignore[arg-type]
        assert len(stats.result_mismatches) == 1
        assert "KXNBAGAME-26JUN13NYKSAS-SAS" in stats.result_mismatches[0]

    async def test_market_without_stored_game_is_reported_not_guessed(self, store: Store) -> None:
        stats = await backfill_markets(FakeKalshi([kalshi_market()]), store)  # type: ignore[arg-type]
        assert stats.markets_unmatched == ["KXNBAGAME-26JUN13NYKSAS-SAS"]
        assert store.markets()[0].game_id is None

    async def test_unknown_team_in_ticker_raises(self, store: Store) -> None:
        bad = kalshi_market(
            ticker="KXNBAGAME-26JUN13NYKSEA-SEA", event_ticker="KXNBAGAME-26JUN13NYKSEA"
        )
        with pytest.raises(KalshiSchemaError, match="SEA"):
            await backfill_markets(FakeKalshi([bad]), store)  # type: ignore[arg-type]


class TestBackfillCandles:
    async def test_window_starts_at_tipoff_minus_pre_and_chunks(self, store: Store) -> None:
        store.upsert_games([final_game()])
        kalshi = FakeKalshi([kalshi_market()])
        await backfill_markets(kalshi, store)  # type: ignore[arg-type]
        stats = await backfill_candles(kalshi, store, pre_tipoff=timedelta(hours=24))  # type: ignore[arg-type]
        # market opened Jun 9 but window must start at tipoff-24h, not open_time
        assert kalshi.candle_windows[0][1] == TIP - timedelta(hours=24)
        # 24h + ~3h to close = ~27h of minutes < 4800 -> single chunk per market
        assert len(kalshi.candle_windows) == 1
        assert stats.candles_upserted == 2

    async def test_long_window_is_chunked(self, store: Store) -> None:
        store.upsert_games([final_game()])
        kalshi = FakeKalshi([kalshi_market(close_time=(TIP + timedelta(days=4)).isoformat())])
        await backfill_markets(kalshi, store)  # type: ignore[arg-type]
        kalshi.candle_windows.clear()
        await backfill_candles(kalshi, store, pre_tipoff=timedelta(hours=24))  # type: ignore[arg-type]
        assert len(kalshi.candle_windows) == 2  # 5 days of minutes / 4800 -> 2 chunks
        first, second = kalshi.candle_windows
        assert first[2] == second[1]  # contiguous


class TestBackfillSnapshots:
    def snapshot(self, game_id: str) -> LiveGameState:
        return LiveGameState(
            game_id=game_id,
            as_of=TIP + timedelta(hours=1),
            period=2,
            seconds_remaining_in_period=100.0,
            home_score=50,
            away_score=52,
            source_play_id="p1",
        )

    async def test_only_games_with_markets_by_default(self, store: Store) -> None:
        # distinct matchup/time so the two games can't collide on event ticker
        other = final_game(
            game_id="no-market",
            home_team="Boston Celtics",
            away_team="Miami Heat",
            start_time=TIP - timedelta(days=30),
        )
        store.upsert_games([final_game(), other])
        kalshi = FakeKalshi([kalshi_market()])
        await backfill_markets(kalshi, store)  # type: ignore[arg-type]
        espn = FakeEspn(
            {},
            snapshots={
                "g-finals-5": [self.snapshot("g-finals-5")],
                "no-market": [self.snapshot("no-market")],
            },
        )
        stats = await backfill_snapshots(espn, store)  # type: ignore[arg-type]
        assert espn.snapshot_requests == ["g-finals-5"]
        assert stats.games_snapshotted == 1
        assert store.snapshots("g-finals-5") != []
        assert store.snapshots("no-market") == []


class TestVerifyDataset:
    async def test_clean_dataset_verifies_ok(self, store: Store) -> None:
        store.upsert_games([final_game()])
        kalshi = FakeKalshi([kalshi_market()])
        await backfill_markets(kalshi, store)  # type: ignore[arg-type]
        await backfill_candles(kalshi, store)  # type: ignore[arg-type]
        report = verify_dataset(store)
        assert report.ok
        assert report.markets_without_game == []
        assert report.markets_without_candles == []

    async def test_mismatch_fails_verification(self, store: Store) -> None:
        store.upsert_games([final_game()])
        await backfill_markets(FakeKalshi([kalshi_market(result="yes")]), store)  # type: ignore[arg-type]
        report = verify_dataset(store)
        assert not report.ok
        assert len(report.settlement_mismatches) == 1
        assert any("MISMATCH" in line for line in report.lines())
