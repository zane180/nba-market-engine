"""Position sizing: fractional Kelly with a hard cap.

For a binary contract bought at price ``p`` (dollars, = probability-cost) with
believed win probability ``q``, the full-Kelly fraction of bankroll to stake is

    f* = (q - p) / (1 - p)

Full Kelly assumes ``q`` is exactly right; ours is an estimate, so we scale by
``kelly_multiplier`` (default 0.25) and cap at ``max_fraction`` of bankroll
(default 5%). Sizing uses the *fee-adjusted* effective price — an edge that
exists only before fees must size to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.backtest.fees import fee_per_contract_dollars


@dataclass(frozen=True)
class SizingParams:
    kelly_multiplier: float = 0.25
    max_fraction: float = 0.05
    max_contracts: float = 100.0  # depth guard: candles carry no book depth,
    # so never assume more than this fills at top-of-book

    def __post_init__(self) -> None:
        if not 0 < self.kelly_multiplier <= 1:
            raise ValueError("kelly_multiplier must be in (0, 1]")
        if not 0 < self.max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        if self.max_contracts <= 0:
            raise ValueError("max_contracts must be positive")


def kelly_fraction(q: float, price_cents: int, *, fee_multiplier: float = 1.0) -> float:
    """Full-Kelly bankroll fraction for buying at ``price_cents`` believing
    ``q``, with the per-contract fee folded into the effective price.
    Returns 0 when there is no positive after-fee edge.

    ``fee_multiplier=0`` exists only for the zero-fee diagnostic run.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q={q} outside [0, 1]")
    effective_price = price_cents / 100.0 + fee_multiplier * fee_per_contract_dollars(price_cents)
    if effective_price >= 1.0 or q <= effective_price:
        return 0.0
    return (q - effective_price) / (1.0 - effective_price)


def size_contracts(
    *,
    q: float,
    price_cents: int,
    bankroll: float,
    params: SizingParams,
    fee_multiplier: float = 1.0,
) -> float:
    """Contracts to buy (possibly fractional; Kalshi supports it). 0 = no trade."""
    if bankroll <= 0:
        return 0.0
    fraction = min(
        params.kelly_multiplier * kelly_fraction(q, price_cents, fee_multiplier=fee_multiplier),
        params.max_fraction,
    )
    if fraction <= 0:
        return 0.0
    stake_dollars = fraction * bankroll
    contracts = stake_dollars / (price_cents / 100.0)
    return min(contracts, params.max_contracts)
