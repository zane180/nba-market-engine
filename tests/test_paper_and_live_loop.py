"""Phase 6: paper-trade lifecycle, the live loop against fake clients, the
scoreboard live-state parser, and the real-money gates."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from engine.backtest.engine import Opportunity, StrategyParams
from engine.data.models import Game, GameStatus, LiveGameState, OrderBook, OrderBookLevel
from engine.data.store import Store
from engine.execution.paper import PaperTrader
from engine.pipeline.live_loop import LiveLoop, TickReport

NOW = datetime(2026, 7, 7, 1, 0, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(f"sqlite:///{tmp_path}/paper.db")
    s.init_schema()
    return s


def make_opportunity(*, model_prob: float = 0.7, ticker: str = "KXNBAGAME-X-BOS") -> Opportunity:
    return Opportunity(
        ticker=ticker,
        decision_time=NOW,
        model_prob=model_prob,
        yes_bid=48,
        yes_ask=50,
        settle_time=NOW + timedelta(hours=3),
        group="g1",
    )


def final_game(home_won: bool = True) -> Game:
    return Game(
        game_id="g1",
        home_team="Boston Celtics",
        away_team="Miami Heat",
        start_time=NOW - timedelta(hours=3),
        status=GameStatus.FINAL,
        home_score=110 if home_won else 100,
        away_score=100 if home_won else 110,
        season_type=2,
    )


class TestPaperTrader:
    def test_edge_creates_persisted_trade(self, store: Store) -> None:
        trader = PaperTrader(store, initial_bankroll=1000.0)
        trade = trader.consider(
            make_opportunity(), game_id="g1", yes_team="Boston Celtics", market_prob=0.49
        )
        assert trade is not None
        stored = store.paper_trades()
        assert len(stored) == 1 and stored[0].is_open
        assert stored[0].cost == pytest.approx(51.75)  # same math as backtest hand-check
        assert trader.bankroll == pytest.approx(1000.0 - 51.75)

    def test_no_edge_no_trade(self, store: Store) -> None:
        trader = PaperTrader(store)
        assert (
            trader.consider(
                make_opportunity(model_prob=0.51),
                game_id="g1",
                yes_team="Boston Celtics",
                market_prob=0.5,
            )
            is None
        )
        assert store.paper_trades() == []

    def test_one_position_per_game_even_across_ticks(self, store: Store) -> None:
        trader = PaperTrader(store)
        first = trader.consider(
            make_opportunity(), game_id="g1", yes_team="Boston Celtics", market_prob=0.49
        )
        second = trader.consider(
            make_opportunity(model_prob=0.9),
            game_id="g1",
            yes_team="Boston Celtics",
            market_prob=0.49,
        )
        assert first is not None and second is None

    def test_settlement_win_and_bankroll(self, store: Store) -> None:
        trader = PaperTrader(store, initial_bankroll=1000.0)
        trader.consider(
            make_opportunity(model_prob=0.7),  # buys YES on Boston market
            game_id="g1",
            yes_team="Boston Celtics",
            market_prob=0.49,
        )
        settled = trader.settle_final_game(final_game(home_won=True), settled_at=NOW)
        assert len(settled) == 1
        trades = store.paper_trades()
        assert not trades[0].is_open
        assert trades[0].payout == pytest.approx(trades[0].contracts)
        assert trader.bankroll == pytest.approx(1000.0 - 51.75 + 100.0)

    def test_settlement_loss(self, store: Store) -> None:
        trader = PaperTrader(store, initial_bankroll=1000.0)
        trader.consider(
            make_opportunity(model_prob=0.7),
            game_id="g1",
            yes_team="Boston Celtics",
            market_prob=0.49,
        )
        trader.settle_final_game(final_game(home_won=False), settled_at=NOW)
        assert store.paper_trades()[0].payout == 0.0
        assert trader.bankroll == pytest.approx(1000.0 - 51.75)

    def test_settling_unrelated_game_touches_nothing(self, store: Store) -> None:
        trader = PaperTrader(store)
        trader.consider(
            make_opportunity(), game_id="g1", yes_team="Boston Celtics", market_prob=0.49
        )
        other = final_game().model_copy(update={"game_id": "other"})
        assert trader.settle_final_game(other, settled_at=NOW) == []
        assert store.paper_trades()[0].is_open


class TestScoreboardLiveStates:
    def payload(self) -> dict[str, Any]:
        payload = json.loads((FIXTURES / "espn_scoreboard.json").read_text())
        event = payload["events"][0]
        event["status"] = {
            "clock": 245.0,
            "displayClock": "4:05",
            "period": 3,
            "type": {"id": "2", "name": "STATUS_IN_PROGRESS", "state": "in", "completed": False},
        }
        return payload  # type: ignore[no-any-return]

    def test_extracts_in_progress_games_only(self) -> None:
        from engine.ingestion.espn import parse_scoreboard_live_states

        states = parse_scoreboard_live_states(self.payload(), as_of=NOW)
        assert len(states) == 1
        state = states[0]
        assert state.period == 3
        assert state.seconds_remaining_in_period == 245.0
        assert state.as_of == NOW
        assert state.home_score > 0

    def test_naive_as_of_rejected(self) -> None:
        from engine.ingestion.espn import parse_scoreboard_live_states

        with pytest.raises(ValueError, match="aware"):
            parse_scoreboard_live_states(self.payload(), as_of=datetime(2026, 7, 7))  # noqa: DTZ001


def make_book(ticker: str, *, best_yes_bid: int, best_no_bid: int) -> OrderBook:
    return OrderBook(
        ticker=ticker,
        as_of=NOW,
        yes_bids=(OrderBookLevel(price=best_yes_bid, quantity=200.0),),
        no_bids=(OrderBookLevel(price=best_no_bid, quantity=200.0),),
    )


class FakeEspn:
    def __init__(self, games: list[Game], states: list[LiveGameState]) -> None:
        self._games = games
        self._states = states

    async def scoreboard_snapshot(self) -> tuple[list[Game], list[LiveGameState]]:
        return self._games, self._states


class FakeKalshi:
    def __init__(self, books: dict[str, OrderBook]) -> None:
        self._books = books
        self.requested: list[str] = []

    async def orderbook(self, ticker: str, *, depth: int = 32) -> OrderBook:
        self.requested.append(ticker)
        return self._books[ticker]


class FakeModels:
    """Model bundle stub with controllable outputs."""

    def __init__(self, pregame: float = 0.7, live: float = 0.8) -> None:
        self._pregame = pregame
        self._live = live

    def pregame_home_prob(self, game: Game) -> float:
        return self._pregame

    def live_home_prob(self, state: LiveGameState, *, prior_home_prob: float) -> float:
        return self._live


class TestLiveLoopTick:
    def scheduled_game(self) -> Game:
        return Game(
            game_id="401999001",
            home_team="Boston Celtics",
            away_team="Miami Heat",
            start_time=NOW + timedelta(hours=2),
            status=GameStatus.SCHEDULED,
            season_type=2,
        )

    def books_for(self, game: Game, *, home_cents: int) -> dict[str, OrderBook]:
        from engine.ingestion.mapping import game_event_ticker, market_ticker

        event = game_event_ticker(
            away_espn_name=game.away_team, home_espn_name=game.home_team, tipoff=game.start_time
        )
        home_t = market_ticker(event, winner_espn_name=game.home_team)
        away_t = market_ticker(event, winner_espn_name=game.away_team)
        return {
            home_t: make_book(home_t, best_yes_bid=home_cents - 1, best_no_bid=99 - home_cents),
            away_t: make_book(away_t, best_yes_bid=99 - home_cents, best_no_bid=home_cents - 1),
        }

    async def run_tick(
        self,
        store: Store,
        games: list[Game],
        states: list[LiveGameState],
        books: dict[str, OrderBook],
        models: FakeModels,
    ) -> tuple[TickReport, PaperTrader, FakeKalshi]:
        trader = PaperTrader(store, params=StrategyParams())
        kalshi = FakeKalshi(books)
        loop = LiveLoop(
            espn=FakeEspn(games, states),  # type: ignore[arg-type]
            kalshi=kalshi,  # type: ignore[arg-type]
            store=store,
            trader=trader,
            models=models,  # type: ignore[arg-type]
        )
        return await loop.tick(), trader, kalshi

    async def test_pregame_edge_produces_trade_and_records_books(self, store: Store) -> None:
        game = self.scheduled_game()
        books = self.books_for(game, home_cents=50)
        report, _trader, kalshi = await self.run_tick(
            store, [game], [], books, FakeModels(pregame=0.7)
        )
        assert report.trades == 1
        assert report.books_recorded == 2
        assert store.book_snapshot_count() > 0
        assert len(kalshi.requested) == 2
        trade = store.paper_trades()[0]
        assert trade.side == "yes" and trade.game_id == game.game_id
        assert trade.market_prob == pytest.approx(0.5, abs=0.02)

    async def test_live_game_uses_live_model(self, store: Store) -> None:
        game = self.scheduled_game().model_copy(
            update={"status": GameStatus.IN_PROGRESS, "home_score": 60, "away_score": 50}
        )
        state = LiveGameState(
            game_id=game.game_id,
            as_of=NOW,
            period=3,
            seconds_remaining_in_period=300.0,
            home_score=60,
            away_score=50,
        )
        books = self.books_for(game, home_cents=70)
        report, _, _ = await self.run_tick(
            store, [game], [state], books, FakeModels(pregame=0.5, live=0.9)
        )
        # live model says .9 vs market ~.7 -> trade fires; pregame .5 would not
        assert report.trades == 1
        assert store.paper_trades()[0].model_prob == pytest.approx(0.9)

    async def test_no_edge_records_books_but_no_trade(self, store: Store) -> None:
        game = self.scheduled_game()
        books = self.books_for(game, home_cents=70)
        report, _, _ = await self.run_tick(store, [game], [], books, FakeModels(pregame=0.7))
        assert report.trades == 0
        assert report.books_recorded == 2

    async def test_final_game_settles_open_position(self, store: Store) -> None:
        game = self.scheduled_game()
        books = self.books_for(game, home_cents=50)
        await self.run_tick(store, [game], [], books, FakeModels(pregame=0.7))
        assert store.paper_trades(open_only=True)
        final = game.model_copy(
            update={"status": GameStatus.FINAL, "home_score": 110, "away_score": 100}
        )
        report, trader, _ = await self.run_tick(store, [final], [], {}, FakeModels())
        assert report.settled == 1
        assert store.paper_trades(open_only=True) == []
        assert trader.bankroll > 1000.0  # YES home won

    async def test_one_bad_game_does_not_kill_the_tick(self, store: Store) -> None:
        good = self.scheduled_game()
        bad = good.model_copy(
            update={
                "game_id": "bad",
                "home_team": "Phoenix Suns",
                "away_team": "Denver Nuggets",
            }
        )
        books = self.books_for(good, home_cents=50)  # 'bad' game's tickers missing -> KeyError
        report, _, _ = await self.run_tick(store, [bad, good], [], books, FakeModels(pregame=0.7))
        assert report.errors == 1
        assert report.trades == 1  # good game still processed

    async def test_unmapped_team_fails_loudly(self, store: Store) -> None:
        from engine.ingestion.mapping import UnmappedTeamError

        seattle = self.scheduled_game().model_copy(update={"home_team": "Seattle SuperSonics"})
        with pytest.raises(UnmappedTeamError):
            await self.run_tick(store, [seattle], [], {}, FakeModels())


class TestLiveGates:
    def make_settings(self, **overrides: object) -> Any:
        from engine.config import Settings

        return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]

    def test_all_gates_closed_by_default(self) -> None:
        from engine.execution.live import gates_from

        gates = gates_from(self.make_settings(), cli_live_flag=False, confirmation=None)
        assert not gates.all_open
        assert len(gates.closed()) == 4

    @pytest.mark.parametrize(
        "missing",
        ["env", "flag", "credentials", "confirmation"],
    )
    def test_any_single_closed_gate_blocks(self, missing: str) -> None:
        from engine.execution.live import CONFIRMATION_PHRASE, gates_from

        settings = self.make_settings(
            live_trading_enabled=missing != "env",
            kalshi_api_key_id=None if missing == "credentials" else "k",
            kalshi_private_key_pem=None if missing == "credentials" else "pem",
        )
        gates = gates_from(
            settings,
            cli_live_flag=missing != "flag",
            confirmation=None if missing == "confirmation" else CONFIRMATION_PHRASE,
        )
        assert not gates.all_open

    async def test_dry_run_logs_but_never_sends(self) -> None:
        from engine.data.models import Side
        from engine.execution.live import LiveExecutor, LiveGates, OrderRequest

        gates = LiveGates(
            env_enabled=True,
            cli_flag=True,
            credentials_present=False,
            confirmed_interactively=True,
        )
        executor = LiveExecutor(self.make_settings(), gates)
        result = await executor.place_order(
            OrderRequest(ticker="T", side=Side.YES, contracts=1, limit_price_cents=50)
        )
        assert result["dry_run"] is True
        assert "Kalshi credentials" in result["closed_gates"]

    def test_wrong_confirmation_phrase_blocks(self) -> None:
        from engine.execution.live import gates_from

        settings = self.make_settings(
            live_trading_enabled=True, kalshi_api_key_id="k", kalshi_private_key_pem="pem"
        )
        gates = gates_from(settings, cli_live_flag=True, confirmation="yes please")
        assert not gates.all_open

    def test_no_side_order_payload_prices_correctly(self) -> None:
        from engine.data.models import Side
        from engine.execution.live import OrderRequest

        payload = OrderRequest(
            ticker="T", side=Side.NO, contracts=3, limit_price_cents=40
        ).payload()
        assert payload["side"] == "no"
        assert payload["yes_price"] == 60  # NO at 40c == YES at 60c
