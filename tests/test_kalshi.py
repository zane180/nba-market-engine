"""Kalshi parser + signer tests against real captured payloads (tests/fixtures/)."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from engine.ingestion.kalshi import (
    KalshiSchemaError,
    KalshiSigner,
    dollars_to_cents,
    parse_candlesticks,
    parse_market_quote,
    parse_orderbook,
)

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())  # type: ignore[no-any-return]


class TestDollarsToCents:
    @pytest.mark.parametrize(
        ("raw", "cents"),
        [("0.0100", 1), ("0.6400", 64), ("0.9900", 99), ("1.0000", 100), ("0.0000", 0)],
    )
    def test_exact_conversion(self, raw: str, cents: int) -> None:
        assert dollars_to_cents(raw) == cents

    @pytest.mark.parametrize("raw", ["0.015", "0.6450", "0.001"])
    def test_off_grid_price_raises_instead_of_rounding(self, raw: str) -> None:
        with pytest.raises(KalshiSchemaError, match="cent grid"):
            dollars_to_cents(raw)

    def test_garbage_raises(self) -> None:
        with pytest.raises(KalshiSchemaError, match="unparseable"):
            dollars_to_cents("N/A")


class TestParseMarketQuote:
    def test_settled_market_empty_book_maps_to_none(self) -> None:
        """Real settled payload has yes_bid=0.0000 / yes_ask=1.0000 — sentinels
        for 'nothing there', not tradable prices."""
        market = load("kalshi_markets.json")["markets"][0]
        quote = parse_market_quote(market, as_of=AS_OF)
        assert quote.ticker == "KXNBAGAME-26JUN13NYKSAS-SAS"
        assert quote.yes_bid is None
        assert quote.yes_ask is None
        assert quote.mid is None

    def test_live_prices_pass_through(self) -> None:
        market = dict(load("kalshi_markets.json")["markets"][0])
        market["yes_bid_dollars"] = "0.6400"
        market["yes_ask_dollars"] = "0.6500"
        quote = parse_market_quote(market, as_of=AS_OF)
        assert (quote.yes_bid, quote.yes_ask) == (64, 65)

    def test_missing_field_raises(self) -> None:
        with pytest.raises(KalshiSchemaError, match="missing"):
            parse_market_quote({"ticker": "X"}, as_of=AS_OF)


class TestParseOrderbook:
    def test_real_book_reversed_to_best_first(self) -> None:
        """API sends ascending prices (best bid last); domain model is best-first."""
        payload = load("kalshi_orderbook.json")
        book = parse_orderbook(payload, ticker="T", as_of=AS_OF)
        raw_yes = payload["orderbook_fp"]["yes_dollars"]
        assert book.yes_bids[0].price == dollars_to_cents(raw_yes[-1][0])  # best == raw last
        assert book.yes_bids[-1].price == dollars_to_cents(raw_yes[0][0])
        assert [lvl.price for lvl in book.yes_bids] == sorted(
            (lvl.price for lvl in book.yes_bids), reverse=True
        )

    def test_fractional_quantities_survive(self) -> None:
        book = parse_orderbook(load("kalshi_orderbook.json"), ticker="T", as_of=AS_OF)
        quantities = [lvl.quantity for lvl in book.yes_bids + book.no_bids]
        assert any(q != int(q) for q in quantities), "fixture should contain a fractional size"

    def test_implied_yes_ask_is_100_minus_best_no_bid(self) -> None:
        book = parse_orderbook(load("kalshi_orderbook.json"), ticker="T", as_of=AS_OF)
        assert book.no_bids and book.best_yes_ask is not None
        assert book.best_yes_ask.price == 100 - book.no_bids[0].price
        quote = book.to_quote()
        assert quote.yes_bid == book.yes_bids[0].price
        assert quote.yes_ask == book.best_yes_ask.price

    def test_empty_book_parses(self) -> None:
        book = parse_orderbook(
            {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}, ticker="T", as_of=AS_OF
        )
        assert book.yes_bids == () and book.to_quote().yes_bid is None

    def test_missing_orderbook_key_raises(self) -> None:
        with pytest.raises(KalshiSchemaError, match="orderbook_fp"):
            parse_orderbook({"orderbook": {}}, ticker="T", as_of=AS_OF)


class TestParseCandlesticks:
    def test_real_payload(self) -> None:
        payload = load("kalshi_candlesticks.json")
        candles = parse_candlesticks(payload, period_seconds=60)
        assert len(candles) == 3
        candle = candles[0]
        assert candle.ticker == payload["ticker"]
        assert candle.end_time.tzinfo is UTC
        assert candle.end_time == datetime.fromtimestamp(
            payload["candlesticks"][0]["end_period_ts"], tz=UTC
        )
        assert candle.yes_bid_close is not None and 1 <= candle.yes_bid_close <= 99

    def test_missing_price_block_raises(self) -> None:
        payload = load("kalshi_candlesticks.json")
        del payload["candlesticks"][0]["yes_bid"]
        with pytest.raises(KalshiSchemaError, match="missing"):
            parse_candlesticks(payload, period_seconds=60)


@pytest.fixture(scope="module")
def keypair() -> tuple[str, rsa.RSAPublicKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return pem, key.public_key()


class TestKalshiSigner:
    def test_headers_verify_against_public_key(self, keypair: tuple[str, rsa.RSAPublicKey]) -> None:
        pem, public = keypair
        signer = KalshiSigner("key-id-123", pem)
        headers = signer.headers(
            "get", "/trade-api/v2/portfolio/balance", timestamp_ms=1751800000000
        )

        assert headers["KALSHI-ACCESS-KEY"] == "key-id-123"
        assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1751800000000"
        message = b"1751800000000GET/trade-api/v2/portfolio/balance"
        public.verify(  # raises InvalidSignature on failure
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )

    def test_non_rsa_key_rejected(self) -> None:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        pem = (
            ed25519.Ed25519PrivateKey.generate()
            .private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            .decode()
        )
        with pytest.raises(ValueError, match="RSA"):
            KalshiSigner("key-id", pem)
