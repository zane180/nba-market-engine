"""Client-level behavior over MockTransport: pagination, cache policy (what is
and is not treated as immutable), and parameter validation."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from engine.ingestion.espn import EspnClient
from engine.ingestion.http import FileCache, RetryingClient
from engine.ingestion.kalshi import KalshiPublicClient, KalshiSchemaError

FIXTURES = Path(__file__).parent / "fixtures"


class Recorder:
    def __init__(self, handler: Any) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)  # type: ignore[no-any-return]


def make_http(recorder: Recorder, tmp_path: Path | None = None) -> RetryingClient:
    return RetryingClient(
        httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        cache=FileCache(tmp_path) if tmp_path else None,
        base_delay=0.001,
    )


class TestKalshiMarketsPagination:
    def pages(self, request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor", "")
        if not cursor:
            return httpx.Response(200, json={"markets": [{"ticker": "A"}], "cursor": "page2"})
        assert cursor == "page2"
        return httpx.Response(200, json={"markets": [{"ticker": "B"}], "cursor": ""})

    async def test_follows_cursor_to_the_end(self) -> None:
        recorder = Recorder(self.pages)
        client = KalshiPublicClient(make_http(recorder), "https://k.test/v2")
        markets = await client.markets(series_ticker="KXNBAGAME", status="settled")
        assert [m["ticker"] for m in markets] == ["A", "B"]
        assert len(recorder.requests) == 2
        assert recorder.requests[0].url.params["status"] == "settled"

    async def test_nonterminating_pagination_raises(self) -> None:
        recorder = Recorder(lambda _: httpx.Response(200, json={"markets": [], "cursor": "again"}))
        client = KalshiPublicClient(make_http(recorder), "https://k.test/v2")
        with pytest.raises(KalshiSchemaError, match="did not terminate"):
            await client.markets(max_pages=3)
        assert len(recorder.requests) == 3


class TestKalshiCandlesticks:
    @staticmethod
    def serve_fixture(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=json.loads((FIXTURES / "kalshi_candlesticks.json").read_text())
        )

    async def test_settled_market_candles_are_cached(self, tmp_path: Path) -> None:
        recorder = Recorder(self.serve_fixture)
        client = KalshiPublicClient(make_http(recorder, tmp_path), "https://k.test/v2")
        window = {
            "start": datetime(2026, 6, 13, 0, 0, tzinfo=UTC),
            "end": datetime(2026, 6, 14, 0, 0, tzinfo=UTC),
        }
        first = await client.candlesticks("T", market_settled=True, **window)  # type: ignore[arg-type]
        second = await client.candlesticks("T", market_settled=True, **window)  # type: ignore[arg-type]
        assert first == second and len(first) == 3
        assert len(recorder.requests) == 1

    async def test_live_market_candles_are_never_cached(self, tmp_path: Path) -> None:
        recorder = Recorder(self.serve_fixture)
        client = KalshiPublicClient(make_http(recorder, tmp_path), "https://k.test/v2")
        window = {
            "start": datetime(2026, 6, 13, 0, 0, tzinfo=UTC),
            "end": datetime(2026, 6, 14, 0, 0, tzinfo=UTC),
        }
        await client.candlesticks("T", market_settled=False, **window)  # type: ignore[arg-type]
        await client.candlesticks("T", market_settled=False, **window)  # type: ignore[arg-type]
        assert len(recorder.requests) == 2

    async def test_sub_minute_period_rejected(self) -> None:
        client = KalshiPublicClient(make_http(Recorder(self.serve_fixture)), "https://k.test/v2")
        with pytest.raises(ValueError, match="whole minutes"):
            await client.candlesticks(
                "T",
                start=datetime(2026, 6, 13, tzinfo=UTC),
                end=datetime(2026, 6, 14, tzinfo=UTC),
                period_seconds=30,
            )


class TestEspnCachePolicy:
    @staticmethod
    def serve_scoreboard(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json.loads((FIXTURES / "espn_scoreboard.json").read_text()))

    async def test_past_scoreboard_cached(self, tmp_path: Path) -> None:
        recorder = Recorder(self.serve_scoreboard)
        client = EspnClient(make_http(recorder, tmp_path), "https://e.test/nba")
        for _ in range(2):
            games = await client.scoreboard(date(2026, 1, 15))
            assert len(games) == 3
        assert len(recorder.requests) == 1

    async def test_today_scoreboard_not_cached(self, tmp_path: Path) -> None:
        """Today's scoreboard is live-changing state; caching it would freeze scores."""
        recorder = Recorder(self.serve_scoreboard)
        client = EspnClient(make_http(recorder, tmp_path), "https://e.test/nba")
        today = datetime.now(UTC).date()
        await client.scoreboard(today)
        await client.scoreboard(today)
        assert len(recorder.requests) == 2

    async def test_future_scoreboard_not_cached(self, tmp_path: Path) -> None:
        recorder = Recorder(self.serve_scoreboard)
        client = EspnClient(make_http(recorder, tmp_path), "https://e.test/nba")
        future = datetime.now(UTC).date() + timedelta(days=7)
        await client.scoreboard(future)
        await client.scoreboard(future)
        assert len(recorder.requests) == 2
