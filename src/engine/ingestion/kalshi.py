"""Kalshi ingestion: public market data plus request signing for the
authenticated endpoints used later by execution.

Payload facts verified against the live API on 2026-07-06:

- Prices arrive as decimal-dollar *strings* ("0.6400") in ``*_dollars`` fields;
  the price grid is one cent (``price_level_structure: linear_cent``). We convert
  exactly via ``Decimal`` and reject off-grid values instead of rounding.
- A market payload with ``yes_bid_dollars == "0.0000"`` / ``yes_ask_dollars ==
  "1.0000"`` means that side of the book is empty, not a tradable 0/100 price.
- Order books come as ``orderbook_fp.{yes,no}_dollars``: ascending
  ``[price, quantity]`` string pairs (best bid LAST), fractional quantities.
- Candlesticks carry separate trade-price / yes_bid / yes_ask OHLC per period.
- History retention is limited: settled markets fall out of the public listing
  (and their candles/trades become unavailable) roughly two months after close.
  Event *metadata* remains addressable indefinitely.
"""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from engine.data.models import Candle, MarketQuote, OrderBook, OrderBookLevel
from engine.ingestion.http import RetryingClient

logger = structlog.get_logger(__name__)

NBA_GAME_SERIES = "KXNBAGAME"


class KalshiSchemaError(Exception):
    """Kalshi returned a shape or value we don't positively recognize."""


def dollars_to_cents(raw: str) -> int:
    """Exact conversion of a decimal-dollar string to integer cents.

    Raises on anything off the one-cent grid — silently rounding a price would
    corrupt every downstream probability by up to half a cent.
    """
    try:
        cents = Decimal(raw) * 100
    except InvalidOperation as exc:
        raise KalshiSchemaError(f"unparseable dollar amount {raw!r}") from exc
    if cents != cents.to_integral_value():
        raise KalshiSchemaError(f"price {raw!r} is not on the one-cent grid")
    return int(cents)


def _tradable_or_none(raw: str) -> int | None:
    """Book-edge price -> cents, mapping the 0/100 'empty side' sentinels to None."""
    cents = dollars_to_cents(raw)
    if cents in (0, 100):
        return None
    return cents


def parse_market_quote(market: dict[str, Any], *, as_of: datetime) -> MarketQuote:
    try:
        return MarketQuote(
            ticker=market["ticker"],
            as_of=as_of,
            yes_bid=_tradable_or_none(market["yes_bid_dollars"]),
            yes_ask=_tradable_or_none(market["yes_ask_dollars"]),
        )
    except KeyError as exc:
        raise KalshiSchemaError(f"market payload missing {exc}") from exc


def _parse_levels(raw: Any, *, context: str) -> tuple[OrderBookLevel, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise KalshiSchemaError(f"{context}: levels are {type(raw).__name__}, not list")
    levels = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise KalshiSchemaError(f"{context}: bad level {pair!r}")
        price, quantity = pair
        levels.append(OrderBookLevel(price=dollars_to_cents(price), quantity=float(quantity)))
    # API sends ascending (best bid last); domain model wants best-first.
    return tuple(reversed(levels))


def parse_orderbook(payload: dict[str, Any], *, ticker: str, as_of: datetime) -> OrderBook:
    book = payload.get("orderbook_fp")
    if not isinstance(book, dict):
        raise KalshiSchemaError(f"{ticker}: no 'orderbook_fp' in orderbook payload")
    return OrderBook(
        ticker=ticker,
        as_of=as_of,
        yes_bids=_parse_levels(book.get("yes_dollars"), context=f"{ticker} yes"),
        no_bids=_parse_levels(book.get("no_dollars"), context=f"{ticker} no"),
    )


def parse_candlesticks(payload: dict[str, Any], *, period_seconds: int) -> list[Candle]:
    ticker = payload.get("ticker")
    raw_candles = payload.get("candlesticks")
    if not isinstance(ticker, str) or not isinstance(raw_candles, list):
        raise KalshiSchemaError("candlesticks payload missing 'ticker' or 'candlesticks'")
    candles = []
    for raw in raw_candles:
        try:
            candles.append(
                Candle(
                    ticker=ticker,
                    end_time=datetime.fromtimestamp(int(raw["end_period_ts"]), tz=UTC),
                    period_seconds=period_seconds,
                    yes_bid_close=_tradable_or_none(raw["yes_bid"]["close_dollars"]),
                    yes_ask_close=_tradable_or_none(raw["yes_ask"]["close_dollars"]),
                    trade_close=(
                        dollars_to_cents(raw["price"]["close_dollars"])
                        if raw.get("price", {}).get("close_dollars") is not None
                        else None
                    ),
                    volume=float(raw.get("volume_fp") or 0),
                    open_interest=float(raw.get("open_interest_fp") or 0),
                )
            )
        except KeyError as exc:
            raise KalshiSchemaError(f"candle for {ticker} missing {exc}") from exc
    return candles


class KalshiSigner:
    """RSA-PSS request signing per Kalshi's API auth scheme.

    Signs ``{timestamp_ms}{METHOD}{path}`` with SHA-256/PSS and emits the three
    ``KALSHI-ACCESS-*`` headers. Only needed for portfolio/order endpoints; all
    of ingestion works unauthenticated.
    """

    def __init__(self, api_key_id: str, private_key_pem: str) -> None:
        self._api_key_id = api_key_id
        key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("Kalshi private key must be RSA")
        self._key = key

    def headers(self, method: str, path: str, *, timestamp_ms: int | None = None) -> dict[str, str]:
        ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode()
        signature = self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }


class KalshiPublicClient:
    """Unauthenticated market-data access. Candles for *settled* markets are
    immutable and cached; live resources never are."""

    def __init__(self, http: RetryingClient, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def markets(
        self,
        *,
        series_ticker: str = NBA_GAME_SERIES,
        status: str | None = None,
        max_pages: int = 30,
    ) -> list[dict[str, Any]]:
        """All markets for a series, following cursor pagination. Returns raw
        payloads — callers pick the fields they need (settlement vs. quotes)."""
        results: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(max_pages):
            params: dict[str, str | int] = {"series_ticker": series_ticker, "limit": 1000}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            page = await self._http.get_json(f"{self._base}/markets", params=params)
            results.extend(page["markets"])
            cursor = page.get("cursor") or ""
            if not cursor:
                return results
        raise KalshiSchemaError(f"pagination did not terminate after {max_pages} pages")

    async def event(self, event_ticker: str) -> dict[str, Any]:
        payload = await self._http.get_json(f"{self._base}/events/{event_ticker}")
        if not isinstance(payload, dict):
            raise KalshiSchemaError(f"event {event_ticker}: non-object response")
        return payload

    async def orderbook(self, ticker: str, *, depth: int = 32) -> OrderBook:
        payload = await self._http.get_json(
            f"{self._base}/markets/{ticker}/orderbook", params={"depth": depth}
        )
        return parse_orderbook(payload, ticker=ticker, as_of=datetime.now(UTC))

    async def candlesticks(
        self,
        ticker: str,
        *,
        series_ticker: str = NBA_GAME_SERIES,
        start: datetime,
        end: datetime,
        period_seconds: int = 60,
        market_settled: bool = False,
    ) -> list[Candle]:
        if period_seconds % 60:
            raise ValueError("Kalshi candle periods are whole minutes")
        params: dict[str, str | int] = {
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": period_seconds // 60,
        }
        payload = await self._http.get_json(
            f"{self._base}/series/{series_ticker}/markets/{ticker}/candlesticks",
            params=params,
            cache_key=(
                f"kalshi:candles:{ticker}:{params['start_ts']}:{params['end_ts']}:{period_seconds}"
                if market_settled
                else None
            ),
        )
        return parse_candlesticks(payload, period_seconds=period_seconds)
