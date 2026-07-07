"""The live loop: ingest -> model -> de-vig -> edge -> sized paper trade.

Each tick, independently and without carrying network state:

1. one ESPN scoreboard fetch -> today's games + in-game states;
2. for each non-final game, resolve its Kalshi event ticker and fetch both
   markets' order books (recorded to the store — this recorder is how we
   accumulate the depth history the public API never provides);
3. model probability: pre-game GBM before tipoff, live WP model in game;
4. de-vig the two books into a market probability, form the opportunity, and
   hand it to the paper trader (same decide() as the backtest);
5. settle open paper positions on games that went final.

A tick that finds no games is a healthy no-op — the loop runs year-round.
Failures inside one game's processing are logged and skipped; one bad payload
must not kill the loop mid-slate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import structlog

from engine.backtest.engine import Opportunity
from engine.data.models import Game, GameStatus, LiveGameState, OrderBook
from engine.data.store import Store
from engine.execution.paper import PaperTrader
from engine.features.live import compute_live_features, seconds_remaining_regulation
from engine.features.pregame import features_for_upcoming
from engine.ingestion.espn import EspnClient
from engine.ingestion.kalshi import KalshiPublicClient
from engine.ingestion.mapping import UnmappedTeamError, game_event_ticker, market_ticker
from engine.models.live_wp_model import LiveWpModel
from engine.models.market_baseline import devig_two_way
from engine.models.pregame_model import GbmModel

logger = structlog.get_logger(__name__)


class ModelBundle:
    """Both models fitted on the full stored history — the live configuration
    (walk-forward splits are for evaluation; live uses everything known)."""

    def __init__(self, pregame: GbmModel, live: LiveWpModel, history: list[Game]) -> None:
        self._pregame = pregame
        self._live = live
        self._history = history

    @classmethod
    def fit_from_store(cls, store: Store, *, seed: int = 1337) -> ModelBundle:
        from engine.features.live import build_live_dataset
        from engine.features.pregame import build_feature_rows

        history = store.games()
        rows = build_feature_rows(history)
        logger.info("fitting live-configuration models", pregame_rows=len(rows))
        pregame = GbmModel(seed=seed).fit(rows)
        live = LiveWpModel(seed=seed).fit(build_live_dataset(store))
        return cls(pregame, live, history)

    def pregame_home_prob(self, game: Game) -> float:
        features = features_for_upcoming(self._history, game)
        return float(self._pregame.predict_proba_features([features])[0])

    def live_home_prob(self, state: LiveGameState, *, prior_home_prob: float) -> float:
        features = compute_live_features(
            score_diff=np.array([float(state.score_diff)]),
            seconds_regulation=np.array(
                [seconds_remaining_regulation(state.period, state.seconds_remaining_in_period)]
            ),
            prior_home_prob=np.array([prior_home_prob]),
            period=np.array([state.period]),
            seconds_in_period=np.array([state.seconds_remaining_in_period]),
        )
        return float(self._live.predict_proba(features)[0])


@dataclass
class TickReport:
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    games_seen: int = 0
    games_live: int = 0
    books_recorded: int = 0
    opportunities: int = 0
    trades: int = 0
    settled: int = 0
    errors: int = 0


class LiveLoop:
    def __init__(
        self,
        *,
        espn: EspnClient,
        kalshi: KalshiPublicClient,
        store: Store,
        trader: PaperTrader,
        models: ModelBundle,
    ) -> None:
        self._espn = espn
        self._kalshi = kalshi
        self._store = store
        self._trader = trader
        self._models = models

    async def tick(self) -> TickReport:
        report = TickReport()
        games, live_states = await self._espn.scoreboard_snapshot()
        self._store.upsert_games([g for g in games if g.season_type != 1])
        states_by_game = {s.game_id: s for s in live_states}
        report.games_seen = len(games)
        report.games_live = len(live_states)

        for game in games:
            try:
                await self._process_game(game, states_by_game.get(game.game_id), report)
            except UnmappedTeamError:
                raise  # a wrong team mapping must never be papered over
            except Exception:
                report.errors += 1
                logger.exception("game processing failed", game_id=game.game_id)
        logger.info(
            "tick complete",
            games=report.games_seen,
            live=report.games_live,
            books=report.books_recorded,
            trades=report.trades,
            settled=report.settled,
            errors=report.errors,
            bankroll=round(self._trader.bankroll, 2),
        )
        return report

    async def _process_game(
        self, game: Game, state: LiveGameState | None, report: TickReport
    ) -> None:
        if game.status is GameStatus.FINAL:
            report.settled += len(
                self._trader.settle_final_game(game, settled_at=datetime.now(UTC))
            )
            return
        if game.status not in (GameStatus.SCHEDULED, GameStatus.IN_PROGRESS):
            return

        event_ticker = game_event_ticker(
            away_espn_name=game.away_team, home_espn_name=game.home_team, tipoff=game.start_time
        )
        home_ticker = market_ticker(event_ticker, winner_espn_name=game.home_team)
        away_ticker = market_ticker(event_ticker, winner_espn_name=game.away_team)
        home_book, away_book = await asyncio.gather(
            self._kalshi.orderbook(home_ticker), self._kalshi.orderbook(away_ticker)
        )
        report.books_recorded += self._record_books(home_book, away_book)

        market_prob = self._devig(home_book, away_book)
        if market_prob is None:
            return  # empty/one-sided books: nothing tradable, nothing comparable

        prior = self._models.pregame_home_prob(game)
        if game.status is GameStatus.IN_PROGRESS and state is not None:
            model_home_prob = self._models.live_home_prob(state, prior_home_prob=prior)
        else:
            model_home_prob = prior

        quote = home_book.to_quote()
        opportunity = Opportunity(
            ticker=home_ticker,
            decision_time=datetime.now(UTC),
            model_prob=model_home_prob,  # YES on the home market
            yes_bid=quote.yes_bid,
            yes_ask=quote.yes_ask,
            settle_time=game.start_time,  # informational; settlement is event-driven here
            group=game.game_id,
        )
        report.opportunities += 1
        trade = self._trader.consider(
            opportunity, game_id=game.game_id, yes_team=game.home_team, market_prob=market_prob
        )
        if trade is not None:
            report.trades += 1

    def _record_books(self, *books: OrderBook) -> int:
        recorded = 0
        for book in books:
            if book.yes_bids or book.no_bids:
                recorded += 1 if self._store.insert_book_snapshot(book) else 0
        return recorded

    @staticmethod
    def _devig(home_book: OrderBook, away_book: OrderBook) -> float | None:
        home_quote, away_quote = home_book.to_quote(), away_book.to_quote()
        if home_quote.mid is None or away_quote.mid is None:
            return None
        home_prob, _ = devig_two_way(home_quote.mid / 100.0, away_quote.mid / 100.0)
        return home_prob

    async def run(self, *, interval_seconds: float = 60.0, max_ticks: int | None = None) -> None:
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            started = asyncio.get_event_loop().time()
            try:
                await self.tick()
            except Exception:
                logger.exception("tick failed; continuing")
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return
            elapsed = asyncio.get_event_loop().time() - started
            await asyncio.sleep(max(0.0, interval_seconds - elapsed))
