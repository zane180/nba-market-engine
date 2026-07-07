"""Event-driven backtest over Kalshi game markets.

Discipline encoded here:

- An ``Opportunity`` carries only information available at its decision time:
  the model's calibrated probability (from walk-forward predictions) and the
  standing top-of-book. Outcomes live in a separate mapping the strategy never
  sees; they are consulted only at settlement.
- Fills are taker at the standing ask (YES) or implied ask (NO), pay Kalshi's
  real fee, and are size-capped because candle data carries no depth.
- At most one position per market, held to settlement — no exit modeling,
  because we have no historical depth to price exits honestly.
- Bankroll accounting is sequential in event time: stakes leave the bankroll at
  entry, payouts return at settlement, and sizing always uses the bankroll as
  of the decision moment.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import structlog

from engine.backtest.fees import fee_per_contract_dollars, taker_fee_dollars
from engine.backtest.metrics import FloatArray, max_drawdown
from engine.backtest.sizing import SizingParams, size_contracts
from engine.data.models import Side

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Opportunity:
    """One tradable moment in one market. ``model_prob`` is P(YES side wins).

    ``group`` identifies mutually-redundant markets (the two markets of one
    game are mirror images — YES home == NO away); the engine takes at most one
    position per group so the same directional bet can't be doubled.
    """

    ticker: str
    decision_time: datetime
    model_prob: float
    yes_bid: int | None
    yes_ask: int | None
    settle_time: datetime
    group: str | None = None

    @property
    def group_key(self) -> str:
        return self.group if self.group is not None else self.ticker

    def buy_price(self, side: Side) -> int | None:
        """Taker price in cents for entering ``side``, from the standing book."""
        if side is Side.YES:
            return self.yes_ask
        return 100 - self.yes_bid if self.yes_bid is not None else None


@dataclass(frozen=True)
class StrategyParams:
    edge_threshold: float = 0.02  # after-fee EV per contract (prob units) to act
    sizing: SizingParams = field(default_factory=SizingParams)
    # 1.0 = Kalshi's real fees; 0.0 exists only for the zero-fee diagnostic
    fee_multiplier: float = 1.0


@dataclass(frozen=True)
class ExecutedTrade:
    ticker: str
    time: datetime
    side: Side
    contracts: float
    price_cents: int
    fee: float
    cost: float  # premium + fee, dollars
    model_prob: float
    won: bool
    payout: float

    @property
    def pnl(self) -> float:
        return self.payout - self.cost


@dataclass(frozen=True)
class BacktestResult:
    initial_bankroll: float
    final_bankroll: float
    trades: list[ExecutedTrade]
    equity_times: list[datetime]
    equity: FloatArray

    @property
    def total_pnl(self) -> float:
        return self.final_bankroll - self.initial_bankroll

    @property
    def roi(self) -> float:
        return self.total_pnl / self.initial_bankroll

    @property
    def turnover(self) -> float:
        return sum(t.cost for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fee for t in self.trades)

    @property
    def win_rate(self) -> float | None:
        if not self.trades:
            return None
        return sum(t.won for t in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        return max_drawdown(self.equity)

    def sharpe_like(self) -> float | None:
        """Mean/std of per-trade returns on cost, scaled by sqrt(n). NOT an
        annualized Sharpe — a small-sample, per-trade analogue, labeled as such."""
        if len(self.trades) < 2:
            return None
        returns = np.array([t.pnl / t.cost for t in self.trades])
        std = float(returns.std(ddof=1))
        if std == 0:
            return None
        return float(returns.mean() / std * np.sqrt(len(returns)))


def decide(
    opportunity: Opportunity, bankroll: float, params: StrategyParams
) -> tuple[Side, float, int] | None:
    """Pure decision: (side, contracts, price_cents) or None.

    Considers both sides; requires after-fee expected value per contract to
    exceed the threshold, then sizes with capped fractional Kelly.
    """
    best: tuple[float, Side, int] | None = None
    for side in (Side.YES, Side.NO):
        price = opportunity.buy_price(side)
        if price is None:
            continue
        q = opportunity.model_prob if side is Side.YES else 1.0 - opportunity.model_prob
        ev = q - price / 100.0 - params.fee_multiplier * fee_per_contract_dollars(price)
        if ev > params.edge_threshold and (best is None or ev > best[0]):
            best = (ev, side, price)
    if best is None:
        return None
    _, side, price = best
    q = opportunity.model_prob if side is Side.YES else 1.0 - opportunity.model_prob
    contracts = size_contracts(
        q=q,
        price_cents=price,
        bankroll=bankroll,
        params=params.sizing,
        fee_multiplier=params.fee_multiplier,
    )
    if contracts <= 0:
        return None
    return side, contracts, price


def simulate(
    opportunities: list[Opportunity],
    outcomes: dict[str, bool],  # ticker -> YES settled true
    *,
    initial_bankroll: float = 1_000.0,
    params: StrategyParams | None = None,
) -> BacktestResult:
    """Run the strategy through opportunities in strict event-time order."""
    params = params or StrategyParams()
    ordered = sorted(opportunities, key=lambda o: (o.decision_time, o.ticker))

    bankroll = initial_bankroll
    equity_times: list[datetime] = []
    equity: list[float] = []
    trades: list[ExecutedTrade] = []
    open_tickers: set[str] = set()
    open_cost: float = 0.0  # open positions marked at cost on the equity curve
    settlement_heap: list[
        tuple[datetime, int, str, float, float]
    ] = []  # (t, seq, tkr, payout, cost)
    seq = 0

    def settle_due(now: datetime) -> None:
        nonlocal bankroll, open_cost
        while settlement_heap and settlement_heap[0][0] <= now:
            settle_time, _, ticker, payout, cost = heapq.heappop(settlement_heap)
            bankroll += payout
            open_cost -= cost
            open_tickers.discard(ticker)
            equity_times.append(settle_time)
            equity.append(bankroll + open_cost)

    # a "traded" marker outlives settlement: one position per group (= game), ever
    traded_groups: set[str] = set()

    for opp in ordered:
        settle_due(opp.decision_time)
        if opp.group_key in traded_groups:
            continue
        if opp.ticker not in outcomes:
            raise KeyError(f"no settlement outcome for {opp.ticker}")
        decision = decide(opp, bankroll, params)
        if decision is None:
            continue
        side, contracts, price = decision
        fee = params.fee_multiplier * taker_fee_dollars(price, contracts)
        cost = contracts * price / 100.0 + fee
        if cost > bankroll:  # never lever up
            continue
        bankroll -= cost
        yes_settled = outcomes[opp.ticker]
        won = yes_settled if side is Side.YES else not yes_settled
        payout = contracts * 1.0 if won else 0.0
        trades.append(
            ExecutedTrade(
                ticker=opp.ticker,
                time=opp.decision_time,
                side=side,
                contracts=contracts,
                price_cents=price,
                fee=fee,
                cost=cost,
                model_prob=opp.model_prob,
                won=won,
                payout=payout,
            )
        )
        seq += 1
        heapq.heappush(settlement_heap, (opp.settle_time, seq, opp.ticker, payout, cost))
        traded_groups.add(opp.group_key)
        open_tickers.add(opp.ticker)
        open_cost += cost
        equity_times.append(opp.decision_time)
        equity.append(bankroll + open_cost)

    # flush remaining settlements
    while settlement_heap:
        settle_time, _, ticker, payout, cost = heapq.heappop(settlement_heap)
        bankroll += payout
        open_cost -= cost
        open_tickers.discard(ticker)
        equity_times.append(settle_time)
        equity.append(bankroll + open_cost)

    if not equity:
        from datetime import UTC

        equity_times, equity = [datetime.min.replace(tzinfo=UTC)], [initial_bankroll]

    return BacktestResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        trades=trades,
        equity_times=equity_times,
        equity=np.array(equity),
    )
