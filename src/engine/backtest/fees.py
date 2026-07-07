"""Kalshi's trading fee schedule.

The published formula for standard markets (which includes single-game sports):

    fee = ceil_to_cent( 0.07 * contracts * P * (1 - P) )

where P is the execution price in dollars. Fees are charged on taker executions;
settlement pays $1.00 per winning contract with no settlement fee. The ceiling
is to the next whole cent — Kalshi rounds fees UP, so we must too: rounding
down would flatter every backtest by a fraction of a cent per fill.
"""

from __future__ import annotations

import math

FEE_RATE = 0.07
_CENT = 100  # cents per dollar


def taker_fee_dollars(price_cents: int, contracts: float) -> float:
    """Fee in dollars for a taker fill of ``contracts`` at ``price_cents``."""
    if not 1 <= price_cents <= 99:
        raise ValueError(f"price {price_cents} outside tradable range 1..99")
    if contracts <= 0:
        raise ValueError(f"contracts must be positive, got {contracts}")
    p = price_cents / _CENT
    raw = FEE_RATE * contracts * p * (1.0 - p)
    return math.ceil(raw * _CENT - 1e-9) / _CENT  # epsilon: don't ceil exact cents up


def fee_per_contract_dollars(price_cents: int) -> float:
    """Marginal fee per contract before rounding — the right quantity for
    edge/EV math (rounding applies once per fill, not per contract)."""
    p = price_cents / _CENT
    return FEE_RATE * p * (1.0 - p)
