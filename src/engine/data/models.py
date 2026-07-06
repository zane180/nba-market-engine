"""Typed domain models shared across ingestion, modeling, backtesting, and execution.

Conventions enforced here so downstream code never has to re-check them:

- All timestamps are timezone-aware UTC. Naive datetimes are rejected at the boundary
  because they are a classic source of lookahead bugs when joining game state to
  market snapshots.
- Kalshi prices are integer cents in [1, 99]; a price of ``c`` cents implies a raw
  (vigged) probability of ``c / 100``. De-vigging lives in ``models/market_baseline``,
  not here — a quote is a market observation, not a probability.
- Models are frozen. A snapshot of the world at time ``t`` must not be mutable.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
"""A calibrated-or-raw probability in [0, 1]."""

PriceCents = Annotated[int, Field(ge=1, le=99)]
"""A Kalshi contract price in cents. Tradable prices are 1..99 inclusive."""


def ensure_utc(ts: datetime) -> datetime:
    """Normalize an aware datetime to UTC. Naive input is a programming error upstream."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime reached a domain model; attach a timezone at ingestion")
    return ts.astimezone(UTC)


class GameStatus(enum.StrEnum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELED = "canceled"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Game(_Frozen):
    """A single NBA game as known from the schedule/scoreboard (ESPN identifiers)."""

    game_id: str
    home_team: str
    away_team: str
    start_time: AwareDatetime
    status: GameStatus
    home_score: int = Field(ge=0, default=0)
    away_score: int = Field(ge=0, default=0)
    # ESPN season.type: 1=preseason, 2=regular season, 3=postseason.
    # None when the payload omits it; preseason is excluded from training data.
    season_type: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> Game:
        if self.home_team == self.away_team:
            raise ValueError(f"home and away team are both {self.home_team!r}")
        if self.status is GameStatus.SCHEDULED and (self.home_score or self.away_score):
            raise ValueError("a scheduled game cannot have a nonzero score")
        return self

    @property
    def home_won(self) -> bool:
        """Outcome label. Only meaningful for final games — raises otherwise."""
        if self.status is not GameStatus.FINAL:
            raise ValueError(f"game {self.game_id} is {self.status}, not final")
        if self.home_score == self.away_score:
            raise ValueError(f"game {self.game_id} is final with a tied score")
        return self.home_score > self.away_score


class LiveGameState(_Frozen):
    """In-game snapshot used by the live win-probability model.

    ``seconds_remaining`` is regulation+OT time left in the *current* period;
    combined with ``period`` it fully orders snapshots within a game.
    """

    game_id: str
    as_of: AwareDatetime
    period: int = Field(ge=1)
    seconds_remaining_in_period: float = Field(ge=0.0, le=720.0)
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    home_has_possession: bool | None = None
    # upstream play id (ESPN) — the stable dedup key when persisting snapshots
    source_play_id: str | None = None

    @property
    def score_diff(self) -> int:
        """Home minus away — positive means the home team leads."""
        return self.home_score - self.away_score


class Side(enum.StrEnum):
    YES = "yes"
    NO = "no"


class MarketQuote(_Frozen):
    """Top-of-book snapshot for one Kalshi market at one instant.

    Prices are on the YES contract; the NO book is its mirror
    (``no_bid = 100 - yes_ask``). ``None`` means that side of the book is empty.
    """

    ticker: str
    as_of: AwareDatetime
    yes_bid: PriceCents | None
    yes_ask: PriceCents | None

    @model_validator(mode="after")
    def _validate(self) -> MarketQuote:
        if self.yes_bid is not None and self.yes_ask is not None and self.yes_bid > self.yes_ask:
            raise ValueError(
                f"{self.ticker}: crossed book (bid {self.yes_bid} > ask {self.yes_ask})"
            )
        return self

    @property
    def mid(self) -> float | None:
        """Midpoint in cents, or None if either side is empty. Still vigged — do not
        treat as a probability without de-vigging."""
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2


class OrderBookLevel(_Frozen):
    price: PriceCents
    # Kalshi supports fractional contracts; observed real book quantities like "36.75".
    quantity: float = Field(gt=0)


class OrderBook(_Frozen):
    """Full depth for one market side-pair at one instant (bids only, per Kalshi:
    the YES ask is implied by the NO bid)."""

    ticker: str
    as_of: AwareDatetime
    yes_bids: tuple[OrderBookLevel, ...]
    no_bids: tuple[OrderBookLevel, ...]

    @model_validator(mode="after")
    def _validate(self) -> OrderBook:
        for name, levels in (("yes_bids", self.yes_bids), ("no_bids", self.no_bids)):
            prices = [lvl.price for lvl in levels]
            if prices != sorted(prices, reverse=True):
                raise ValueError(f"{self.ticker}: {name} not sorted best-first")
        return self

    @property
    def best_yes_bid(self) -> OrderBookLevel | None:
        return self.yes_bids[0] if self.yes_bids else None

    @property
    def best_yes_ask(self) -> OrderBookLevel | None:
        """Implied from the best NO bid: selling NO at ``p`` == buying YES at ``100 - p``."""
        if not self.no_bids:
            return None
        best_no = self.no_bids[0]
        return OrderBookLevel(price=100 - best_no.price, quantity=best_no.quantity)

    def to_quote(self) -> MarketQuote:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        return MarketQuote(
            ticker=self.ticker,
            as_of=self.as_of,
            yes_bid=bid.price if bid else None,
            yes_ask=ask.price if ask else None,
        )


class MarketResult(enum.StrEnum):
    YES = "yes"
    NO = "no"


class MarketInfo(_Frozen):
    """Settlement-level metadata for one Kalshi game-winner market.

    ``yes_team`` is the ESPN display name of the team whose win resolves this
    market YES. ``game_id`` is the ESPN game this market was matched to — None
    when no stored game matched (surfaced by ``ingest verify``, never guessed).
    """

    ticker: str
    event_ticker: str
    game_id: str | None
    yes_team: str
    result: MarketResult | None
    open_time: AwareDatetime
    close_time: AwareDatetime
    volume: float = Field(ge=0)


class Candle(_Frozen):
    """One period of Kalshi market history (from the candlesticks endpoint).

    ``end_time`` is the period's end; treating the candle as known any earlier
    than that is lookahead. Price fields are None when that side had no
    quote/trade in the period.
    """

    ticker: str
    end_time: AwareDatetime
    period_seconds: int = Field(gt=0)
    yes_bid_close: PriceCents | None
    yes_ask_close: PriceCents | None
    trade_close: PriceCents | None
    volume: float = Field(ge=0)
    open_interest: float = Field(ge=0)
