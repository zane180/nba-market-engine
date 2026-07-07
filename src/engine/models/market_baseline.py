"""The market baseline: turn Kalshi quotes into a clean home-win probability.

A Kalshi game event has two YES/NO markets (one per team). Their raw implied
probabilities sum to more than 1 — the overround ("vig") — so quoted prices are
NOT probabilities until de-vigged. We use multiplicative normalization:

    p_home = raw_home / (raw_home + raw_away)

which is the standard two-outcome de-vig and exact when the vig is applied
proportionally. Every model-vs-market comparison uses this number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.data.models import Candle, MarketInfo
from engine.data.store import Store

MAX_QUOTE_STALENESS = timedelta(hours=6)


def devig_two_way(raw_a: float, raw_b: float) -> tuple[float, float]:
    """Normalize two raw implied probabilities to sum to 1.

    Raises if either input is non-positive or the pair carries no overround at
    all in the wrong direction is fine (sum < 1 happens with wide books).
    """
    if raw_a <= 0 or raw_b <= 0:
        raise ValueError(f"raw probabilities must be positive, got {raw_a}, {raw_b}")
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


@dataclass(frozen=True, slots=True)
class MarketProbability:
    game_id: str
    home_prob: float
    as_of: datetime  # timestamp of the older of the two quotes used
    overround: float  # raw_home + raw_away - 1; a liquidity/width diagnostic


def _last_quotable_candle(candles: list[Candle], at: datetime) -> Candle | None:
    """Most recent candle at-or-before ``at`` with both sides of the book."""
    for candle in reversed(candles):
        if candle.end_time > at:
            continue
        if candle.yes_bid_close is not None and candle.yes_ask_close is not None:
            return candle
    return None


def _mid_prob(candle: Candle) -> float:
    assert candle.yes_bid_close is not None and candle.yes_ask_close is not None
    return (candle.yes_bid_close + candle.yes_ask_close) / 200.0


def market_home_probability(
    store: Store,
    *,
    game_id: str,
    home_team: str,
    at: datetime,
    max_staleness: timedelta = MAX_QUOTE_STALENESS,
) -> MarketProbability | None:
    """De-vigged home-win probability implied by the freshest two-sided quotes
    at-or-before ``at``. None when either market lacks a usable quote — an
    absent market prob must never be silently replaced with a default.
    """
    markets = [m for m in store.markets(with_game_only=True) if m.game_id == game_id]
    home_market: MarketInfo | None = None
    away_market: MarketInfo | None = None
    for market in markets:
        if market.yes_team == home_team:
            home_market = market
        else:
            away_market = market
    if home_market is None or away_market is None:
        return None

    home_candle = _last_quotable_candle(store.candles(home_market.ticker), at)
    away_candle = _last_quotable_candle(store.candles(away_market.ticker), at)
    if home_candle is None or away_candle is None:
        return None
    oldest = min(home_candle.end_time, away_candle.end_time)
    if at - oldest > max_staleness:
        return None

    raw_home = _mid_prob(home_candle)
    raw_away = _mid_prob(away_candle)
    home_prob, _ = devig_two_way(raw_home, raw_away)
    return MarketProbability(
        game_id=game_id,
        home_prob=home_prob,
        as_of=oldest,
        overround=raw_home + raw_away - 1.0,
    )
