"""Storage-layer contracts: idempotent upserts, aware-UTC round-trips, and
schema-version guarding — the properties every backfill re-run relies on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.data.models import (
    Candle,
    Game,
    GameStatus,
    LiveGameState,
    MarketInfo,
    MarketResult,
)
from engine.data.store import SCHEMA_VERSION, SchemaVersionError, Store

TIP = datetime(2026, 6, 14, 0, 30, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(f"sqlite:///{tmp_path}/test.db")
    s.init_schema()
    return s


def make_game(**overrides: object) -> Game:
    base: dict[str, object] = {
        "game_id": "401810433",
        "home_team": "Orlando Magic",
        "away_team": "Memphis Grizzlies",
        "start_time": TIP,
        "status": GameStatus.SCHEDULED,
        "season_type": 2,
    }
    return Game(**{**base, **overrides})  # type: ignore[arg-type]


class TestGames:
    def test_roundtrip_preserves_everything(self, store: Store) -> None:
        game = make_game(status=GameStatus.FINAL, home_score=118, away_score=111)
        store.upsert_games([game])
        assert store.game("401810433") == game

    def test_upsert_is_idempotent(self, store: Store) -> None:
        store.upsert_games([make_game()])
        store.upsert_games([make_game()])
        assert store.counts()["games"] == 1

    def test_upsert_updates_in_place(self, store: Store) -> None:
        """A game re-ingested after finishing must overwrite, not duplicate —
        this is what makes re-running the backfill safe mid-season."""
        store.upsert_games([make_game()])
        store.upsert_games([make_game(status=GameStatus.FINAL, home_score=118, away_score=111)])
        stored = store.game("401810433")
        assert stored is not None
        assert stored.status is GameStatus.FINAL
        assert stored.home_score == 118
        assert store.counts()["games"] == 1

    def test_timestamps_come_back_aware_utc(self, store: Store) -> None:
        eastern_tip = TIP.astimezone(timezone(timedelta(hours=-4)))
        store.upsert_games([make_game(start_time=eastern_tip)])
        stored = store.game("401810433")
        assert stored is not None
        assert stored.start_time.tzinfo is UTC
        assert stored.start_time == TIP

    def test_range_and_status_filters(self, store: Store) -> None:
        store.upsert_games(
            [
                make_game(),
                make_game(
                    game_id="2",
                    start_time=TIP + timedelta(days=2),
                    status=GameStatus.FINAL,
                    home_score=1,
                    away_score=2,
                ),
            ]
        )
        assert [g.game_id for g in store.games(end=TIP + timedelta(days=1))] == ["401810433"]
        assert [g.game_id for g in store.games(statuses=[GameStatus.FINAL])] == ["2"]


class TestSnapshots:
    def make_snapshot(self, play_id: str | None = "p1", home_score: int = 10) -> LiveGameState:
        return LiveGameState(
            game_id="g1",
            as_of=TIP,
            period=1,
            seconds_remaining_in_period=300.0,
            home_score=home_score,
            away_score=8,
            source_play_id=play_id,
        )

    def test_roundtrip_and_dedupe_on_play_id(self, store: Store) -> None:
        store.upsert_snapshots([self.make_snapshot(), self.make_snapshot(home_score=12)])
        stored = store.snapshots("g1")
        assert len(stored) == 1
        assert stored[0].home_score == 12  # last write wins

    def test_snapshot_without_play_id_is_not_stored(self, store: Store) -> None:
        assert store.upsert_snapshots([self.make_snapshot(play_id=None)]) == 0
        assert store.snapshots("g1") == []


class TestMarketsAndCandles:
    def make_info(self, *, result: MarketResult | None = MarketResult.NO) -> MarketInfo:
        return MarketInfo(
            ticker="KXNBAGAME-26JUN13NYKSAS-SAS",
            event_ticker="KXNBAGAME-26JUN13NYKSAS",
            game_id="401810433",
            yes_team="San Antonio Spurs",
            result=result,
            open_time=TIP - timedelta(days=4),
            close_time=TIP + timedelta(hours=3),
            volume=48_467_702.82,
        )

    def test_market_roundtrip(self, store: Store) -> None:
        info = self.make_info()
        store.upsert_markets([info])
        assert store.markets() == [info]

    def test_unsettled_market_result_roundtrips_as_none(self, store: Store) -> None:
        store.upsert_markets([self.make_info(result=None)])
        assert store.markets()[0].result is None

    def test_with_game_only_filter(self, store: Store) -> None:
        unmatched = self.make_info().model_copy(update={"ticker": "X-Y", "game_id": None})
        store.upsert_markets([self.make_info(), unmatched])
        assert len(store.markets()) == 2
        assert [m.ticker for m in store.markets(with_game_only=True)] == [
            "KXNBAGAME-26JUN13NYKSAS-SAS"
        ]

    def test_candle_roundtrip_and_composite_key(self, store: Store) -> None:
        candle = Candle(
            ticker="T",
            end_time=TIP,
            period_seconds=60,
            yes_bid_close=64,
            yes_ask_close=65,
            trade_close=64,
            volume=95613.30,
            open_interest=2323845.14,
        )
        store.upsert_candles([candle, candle])  # same key twice in one batch is fine on re-run
        store.upsert_candles([candle.model_copy(update={"end_time": TIP + timedelta(minutes=1)})])
        stored = store.candles("T")
        assert len(stored) == 2
        assert stored[0] == candle
        assert store.candle_count_by_ticker() == {"T": 2}


class TestSchemaVersion:
    def test_reopening_same_version_is_fine(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path}/v.db"
        Store(url).init_schema()
        Store(url).init_schema()  # no error

    def test_unknown_version_refuses_to_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = f"sqlite:///{tmp_path}/v.db"
        Store(url).init_schema()
        monkeypatch.setattr("engine.data.store.SCHEMA_VERSION", SCHEMA_VERSION + 1)
        with pytest.raises(SchemaVersionError, match=f"expects v{SCHEMA_VERSION + 1}"):
            Store(url).init_schema()

    def test_naive_datetime_rejected_at_boundary(self, store: Store) -> None:
        from engine.data.store import _epoch

        with pytest.raises(ValueError, match="naive"):
            _epoch(datetime(2026, 1, 1))  # noqa: DTZ001 — naive on purpose
