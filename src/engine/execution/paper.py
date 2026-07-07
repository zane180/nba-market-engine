"""Paper trading: the default — and only enabled — execution path.

Decisions come from the exact same ``decide()`` the backtest uses, so the
paper track measures the same strategy the backtest simulated. Trades are
persisted (never just printed), settle when the game goes final, and the
bankroll is derived from the trade ledger — there is no mutable balance to
drift out of sync.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from engine.backtest.engine import Opportunity, StrategyParams, decide
from engine.backtest.fees import taker_fee_dollars
from engine.data.models import Game, Side
from engine.data.store import PaperTradeRecord, Store

logger = structlog.get_logger(__name__)


class PaperTrader:
    def __init__(
        self,
        store: Store,
        *,
        initial_bankroll: float = 1_000.0,
        params: StrategyParams | None = None,
    ) -> None:
        self._store = store
        self._initial = initial_bankroll
        self._params = params or StrategyParams()

    @property
    def bankroll(self) -> float:
        return self._store.paper_bankroll(self._initial)

    def games_with_positions(self) -> set[str]:
        return {t.game_id for t in self._store.paper_trades()}

    def consider(
        self,
        opportunity: Opportunity,
        *,
        game_id: str,
        yes_team: str,
        market_prob: float,
    ) -> PaperTradeRecord | None:
        """Run the strategy on one opportunity; persist and return the trade if
        it fires. One position per game, ever — same rule as the backtest."""
        if game_id in self.games_with_positions():
            return None
        decision = decide(opportunity, self.bankroll, self._params)
        if decision is None:
            return None
        side, contracts, price = decision
        fee = taker_fee_dollars(price, contracts)
        cost = contracts * price / 100.0 + fee
        if cost > self.bankroll:
            logger.warning("paper trade skipped: insufficient bankroll", ticker=opportunity.ticker)
            return None
        trade = PaperTradeRecord(
            trade_id=f"{opportunity.ticker}:{int(opportunity.decision_time.timestamp())}",
            ticker=opportunity.ticker,
            game_id=game_id,
            yes_team=yes_team,
            entered_at=opportunity.decision_time,
            side=side.value,
            contracts=contracts,
            price_cents=price,
            fee=fee,
            cost=cost,
            model_prob=opportunity.model_prob,
            market_prob=market_prob,
        )
        self._store.insert_paper_trade(trade)
        logger.info(
            "PAPER TRADE",
            ticker=trade.ticker,
            side=trade.side,
            contracts=round(contracts, 2),
            price_cents=price,
            cost=round(cost, 2),
            model_prob=round(trade.model_prob, 4),
            market_prob=round(market_prob, 4),
        )
        return trade

    def settle_final_game(self, game: Game, *, settled_at: datetime) -> list[PaperTradeRecord]:
        """Settle open trades on a game that just went final."""
        settled: list[PaperTradeRecord] = []
        for trade in self._store.paper_trades(open_only=True):
            if trade.game_id != game.game_id:
                continue
            yes_won = (trade.yes_team == game.home_team) == game.home_won
            trade_won = yes_won if trade.side == Side.YES.value else not yes_won
            payout = trade.contracts if trade_won else 0.0
            self._store.settle_paper_trade(trade.trade_id, settled_at=settled_at, payout=payout)
            logger.info(
                "paper trade settled",
                ticker=trade.ticker,
                won=trade_won,
                payout=round(payout, 2),
                pnl=round(payout - trade.cost, 2),
            )
            settled.append(trade)
        return settled
