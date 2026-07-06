"""Shared async HTTP plumbing for ingestion clients.

Both upstream APIs are undocumented-or-rate-limited, so every client goes through
``RetryingClient``:

- retries transport errors, 429, and 5xx with exponential backoff + jitter,
  honoring ``Retry-After`` when the server sends one;
- other 4xx are *not* retried — they mean we asked a wrong question, and looping
  on them just burns rate limit;
- optionally caches GET responses on disk. Caching is opt-in per call and meant
  only for immutable resources (past scoreboards, settled-market candles):
  callers pass ``cache_key`` exactly when the response can never change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class UpstreamError(Exception):
    """A request failed after exhausting retries."""


class FileCache:
    """Content-addressed JSON cache. Keys are caller-chosen strings; values are
    whatever ``json.dumps`` accepts."""

    def __init__(self, root: Path) -> None:
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._root / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)  # atomic on POSIX


class RetryingClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        cache: FileCache | None = None,
        max_attempts: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
    ) -> None:
        self._client = client
        self._cache = cache
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        headers: dict[str, str] | None = None,
        cache_key: str | None = None,
    ) -> Any:
        if cache_key is not None and self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        payload = await self._request_with_retry(url, params=params, headers=headers)

        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, payload)
        return payload

    async def _request_with_retry(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None,
        headers: dict[str, str] | None,
    ) -> Any:
        last_error: str = "unknown"
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            try:
                response = await self._client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc!r}"
            else:
                if response.status_code < 400:
                    return response.json()
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise UpstreamError(f"GET {url}: {last_error} (not retryable)")
                header = response.headers.get("Retry-After")
                if header is not None:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        retry_after = None

            if attempt == self._max_attempts:
                break
            delay = retry_after
            if delay is None:
                delay = min(self._max_delay, self._base_delay * 2 ** (attempt - 1))
                delay *= 0.5 + random.random()
            logger.warning(
                "retrying upstream request",
                url=url,
                attempt=attempt,
                error=last_error,
                delay_seconds=round(delay, 2),
            )
            await asyncio.sleep(delay)

        raise UpstreamError(
            f"GET {url}: giving up after {self._max_attempts} attempts ({last_error})"
        )
