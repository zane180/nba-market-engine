"""De-vig math and market-probability extraction from stored candles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine.data.models import Candle, Game, GameStatus, MarketInfo, MarketResult
from engine.data.store import Store
from engine.models.market_baseline import devig_two_way, market_home_probability

TIP = datetime(2026, 6, 14, 0, 30, tzinfo=UTC)


class TestDevig:
    def test_symmetric_vig_splits_evenly(self) -> None:
        # 52c/52c mids -> both raw 0.52 -> de-vigged 0.5 each
        a, b = devig_two_way(0.52, 0.52)
        assert a == pytest.approx(0.5)
        assert b == pytest.approx(0.5)

    def test_hand_computed(self) -> None:
        # favorite at 0.78 raw, dog at 0.26 raw (overround 0.04)
        home, away = devig_two_way(0.78, 0.26)
        assert home == pytest.approx(0.75)
        assert away == pytest.approx(0.25)
        assert home + away == pytest.approx(1.0)

    def test_underround_also_normalizes(self) -> None:
        # wide books can sum below 1; normalization still applies
        home, _away = devig_two_way(0.45, 0.45)
        assert home == pytest.approx(0.5)

    def test_nonpositive_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            devig_two_way(0.0, 0.5)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(f"sqlite:///{tmp_path}/t.db")
    s.init_schema()
    s.upsert_games(
        [
            Game(
                game_id="g1",
                home_team="San Antonio Spurs",
                away_team="New York Knicks",
                start_time=TIP,
                status=GameStatus.FINAL,
                home_score=90,
                away_score=94,
                season_type=3,
            )
        ]
    )
    for team, ticker in (
        ("San Antonio Spurs", "E-SAS"),
        ("New York Knicks", "E-NYK"),
    ):
        s.upsert_markets(
            [
                MarketInfo(
                    ticker=ticker,
                    event_ticker="E",
                    game_id="g1",
                    yes_team=team,
                    result=MarketResult.NO,
                    open_time=TIP - timedelta(days=4),
                    close_time=TIP + timedelta(hours=3),
                    volume=1000.0,
                )
            ]
        )
    return s


def candle(ticker: str, *, minutes_before_tip: int, bid: int | None, ask: int | None) -> Candle:
    return Candle(
        ticker=ticker,
        end_time=TIP - timedelta(minutes=minutes_before_tip),
        period_seconds=60,
        yes_bid_close=bid,
        yes_ask_close=ask,
        trade_close=None,
        volume=10.0,
        open_interest=100.0,
    )


class TestMarketHomeProbability:
    def test_devigs_freshest_pretip_quotes(self, store: Store) -> None:
        store.upsert_candles(
            [
                candle("E-SAS", minutes_before_tip=10, bid=55, ask=57),  # raw mid .56
                candle("E-SAS", minutes_before_tip=120, bid=50, ask=52),  # stale, ignored
                candle("E-NYK", minutes_before_tip=5, bid=45, ask=47),  # raw mid .46
            ]
        )
        result = market_home_probability(store, game_id="g1", home_team="San Antonio Spurs", at=TIP)
        assert result is not None
        assert result.home_prob == pytest.approx(0.56 / (0.56 + 0.46))
        assert result.overround == pytest.approx(0.02)
        assert result.as_of == TIP - timedelta(minutes=10)  # older of the two quotes

    def test_candles_after_cutoff_are_invisible(self, store: Store) -> None:
        """A quote printed after `at` must never inform the probability —
        the no-lookahead property on the market side."""
        store.upsert_candles(
            [
                candle("E-SAS", minutes_before_tip=30, bid=50, ask=52),
                candle("E-SAS", minutes_before_tip=-5, bid=90, ask=92),  # post-tip, in-game
                candle("E-NYK", minutes_before_tip=30, bid=48, ask=50),
                candle("E-NYK", minutes_before_tip=-5, bid=8, ask=10),
            ]
        )
        result = market_home_probability(store, game_id="g1", home_team="San Antonio Spurs", at=TIP)
        assert result is not None
        assert result.home_prob == pytest.approx(0.51 / (0.51 + 0.49))

    def test_one_sided_book_gives_none(self, store: Store) -> None:
        store.upsert_candles(
            [
                candle("E-SAS", minutes_before_tip=10, bid=55, ask=None),
                candle("E-NYK", minutes_before_tip=10, bid=45, ask=47),
            ]
        )
        assert (
            market_home_probability(store, game_id="g1", home_team="San Antonio Spurs", at=TIP)
            is None
        )

    def test_stale_quotes_give_none(self, store: Store) -> None:
        store.upsert_candles(
            [
                candle("E-SAS", minutes_before_tip=60 * 10, bid=55, ask=57),
                candle("E-NYK", minutes_before_tip=60 * 10, bid=45, ask=47),
            ]
        )
        assert (
            market_home_probability(store, game_id="g1", home_team="San Antonio Spurs", at=TIP)
            is None
        )

    def test_missing_market_gives_none(self, store: Store) -> None:
        assert (
            market_home_probability(store, game_id="nope", home_team="San Antonio Spurs", at=TIP)
            is None
        )
