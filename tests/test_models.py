"""Domain-model invariants. These are the boundary contracts every later phase
(ingestion, backtest, execution) relies on, so each rule gets an explicit test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from engine.data.models import (
    Game,
    GameStatus,
    LiveGameState,
    MarketQuote,
    OrderBook,
    OrderBookLevel,
    ensure_utc,
)

AWARE = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
NAIVE = datetime(2026, 1, 15, 23, 30)  # noqa: DTZ001 — naive on purpose: rejection under test


def make_game(**overrides: object) -> Game:
    base: dict[str, object] = {
        "game_id": "401585601",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "start_time": AWARE,
        "status": GameStatus.SCHEDULED,
    }
    return Game(**{**base, **overrides})  # type: ignore[arg-type]


class TestTimezoneDiscipline:
    """Naive datetimes cause silent misjoins between game state and market
    snapshots — they must be rejected at model construction."""

    def test_naive_start_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_game(start_time=NAIVE)

    def test_naive_quote_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MarketQuote(ticker="KXNBA-X", as_of=NAIVE, yes_bid=40, yes_ask=42)

    def test_ensure_utc_normalizes_offset(self) -> None:
        eastern = datetime(2026, 1, 15, 19, 0, tzinfo=timezone(timedelta(hours=-5)))
        assert ensure_utc(eastern) == datetime(2026, 1, 16, 0, 0, tzinfo=UTC)

    def test_ensure_utc_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="naive datetime"):
            ensure_utc(NAIVE)


class TestGame:
    def test_same_home_and_away_rejected(self) -> None:
        with pytest.raises(ValidationError, match="home and away"):
            make_game(away_team="Boston Celtics")

    def test_scheduled_game_with_score_rejected(self) -> None:
        with pytest.raises(ValidationError, match="scheduled"):
            make_game(home_score=2)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_game(status=GameStatus.FINAL, home_score=-1, away_score=100)

    def test_home_won_label(self) -> None:
        game = make_game(status=GameStatus.FINAL, home_score=110, away_score=104)
        assert game.home_won is True
        game = make_game(status=GameStatus.FINAL, home_score=99, away_score=104)
        assert game.home_won is False

    def test_home_won_unavailable_before_final(self) -> None:
        with pytest.raises(ValueError, match="not final"):
            _ = make_game(status=GameStatus.IN_PROGRESS, home_score=50, away_score=40).home_won

    def test_home_won_rejects_tie(self) -> None:
        # NBA games cannot end tied; a tied 'final' means an ingestion bug.
        with pytest.raises(ValueError, match="tied"):
            _ = make_game(status=GameStatus.FINAL, home_score=100, away_score=100).home_won

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            make_game().home_score = 10  # type: ignore[misc]


class TestMarketQuote:
    def test_crossed_book_rejected(self) -> None:
        with pytest.raises(ValidationError, match="crossed"):
            MarketQuote(ticker="KXNBA-X", as_of=AWARE, yes_bid=55, yes_ask=54)

    def test_touching_book_allowed(self) -> None:
        quote = MarketQuote(ticker="KXNBA-X", as_of=AWARE, yes_bid=55, yes_ask=55)
        assert quote.mid == 55.0

    @pytest.mark.parametrize("price", [0, 100, -3, 101])
    def test_untradable_prices_rejected(self, price: int) -> None:
        with pytest.raises(ValidationError):
            MarketQuote(ticker="KXNBA-X", as_of=AWARE, yes_bid=price, yes_ask=None)

    def test_empty_side_gives_no_mid(self) -> None:
        quote = MarketQuote(ticker="KXNBA-X", as_of=AWARE, yes_bid=40, yes_ask=None)
        assert quote.mid is None


class TestLiveGameState:
    def test_score_diff_is_home_minus_away(self) -> None:
        state = LiveGameState(
            game_id="401585601",
            as_of=AWARE,
            period=3,
            seconds_remaining_in_period=125.0,
            home_score=78,
            away_score=81,
        )
        assert state.score_diff == -3

    def test_seconds_remaining_bounded_by_period_length(self) -> None:
        with pytest.raises(ValidationError):
            LiveGameState(
                game_id="401585601",
                as_of=AWARE,
                period=1,
                seconds_remaining_in_period=721.0,
                home_score=0,
                away_score=0,
            )


class TestOrderBook:
    def test_levels_must_be_sorted_best_first(self) -> None:
        with pytest.raises(ValidationError, match="sorted"):
            OrderBook(
                ticker="KXNBA-X",
                as_of=AWARE,
                yes_bids=(
                    OrderBookLevel(price=40, quantity=10),
                    OrderBookLevel(price=45, quantity=5),
                ),
                no_bids=(),
            )

    def test_sorted_book_accepted(self) -> None:
        book = OrderBook(
            ticker="KXNBA-X",
            as_of=AWARE,
            yes_bids=(
                OrderBookLevel(price=45, quantity=5),
                OrderBookLevel(price=40, quantity=10),
            ),
            no_bids=(OrderBookLevel(price=54, quantity=7),),
        )
        assert book.yes_bids[0].price == 45
