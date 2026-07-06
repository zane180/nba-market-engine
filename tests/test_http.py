"""Retry/backoff and cache behavior of the shared HTTP layer, using httpx
MockTransport — no real network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from engine.ingestion.http import FileCache, RetryingClient, UpstreamError


class Script:
    """Serves a scripted sequence of responses and records request count."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def make_client(
    script: Script, *, cache: FileCache | None = None, max_attempts: int = 4
) -> RetryingClient:
    return RetryingClient(
        httpx.AsyncClient(transport=httpx.MockTransport(script)),
        cache=cache,
        max_attempts=max_attempts,
        base_delay=0.001,  # keep tests fast; delay math is exercised, waiting isn't
    )


async def test_success_passthrough() -> None:
    script = Script([httpx.Response(200, json={"ok": True})])
    assert await make_client(script).get_json("https://x.test/a") == {"ok": True}
    assert script.calls == 1


async def test_retries_429_then_succeeds() -> None:
    script = Script(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(500),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    assert await make_client(script).get_json("https://x.test/a") == {"ok": 1}
    assert script.calls == 3


async def test_exhausted_retries_raise() -> None:
    script = Script([httpx.Response(503)])
    with pytest.raises(UpstreamError, match="giving up after 4 attempts"):
        await make_client(script).get_json("https://x.test/a")
    assert script.calls == 4


async def test_client_errors_are_not_retried() -> None:
    """404/401 mean the request itself is wrong; retrying burns rate limit."""
    script = Script([httpx.Response(404)])
    with pytest.raises(UpstreamError, match="not retryable"):
        await make_client(script).get_json("https://x.test/missing")
    assert script.calls == 1


async def test_transport_errors_are_retried() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": 1})

    client = RetryingClient(
        httpx.AsyncClient(transport=httpx.MockTransport(flaky)), base_delay=0.001
    )
    assert await client.get_json("https://x.test/a") == {"ok": 1}
    assert calls["n"] == 3


class TestFileCache:
    async def test_cache_hit_skips_network(self, tmp_path: Path) -> None:
        script = Script([httpx.Response(200, json={"v": 1})])
        client = make_client(script, cache=FileCache(tmp_path))
        first = await client.get_json("https://x.test/a", cache_key="k")
        second = await client.get_json("https://x.test/a", cache_key="k")
        assert first == second == {"v": 1}
        assert script.calls == 1

    async def test_no_cache_key_means_no_caching(self, tmp_path: Path) -> None:
        script = Script([httpx.Response(200, json={"v": 1})])
        client = make_client(script, cache=FileCache(tmp_path))
        await client.get_json("https://x.test/a")
        await client.get_json("https://x.test/a")
        assert script.calls == 2

    def test_roundtrip_preserves_json_types(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        value: dict[str, Any] = {"a": [1, 2.5, None, "s"], "b": {"nested": True}}
        cache.put("key", value)
        assert cache.get("key") == value
        assert cache.get("other") is None
