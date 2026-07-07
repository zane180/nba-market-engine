"""Real-money order placement. OFF BY DEFAULT, multiply gated, dry-run first.

Placing a real order requires ALL of:

1. ``ENGINE_LIVE_TRADING_ENABLED=true`` in the environment,
2. the ``--live`` CLI flag on the running command,
3. Kalshi API credentials present, and
4. an interactive confirmation phrase typed at startup.

Anything less logs the exact request that WOULD have been sent and returns.
This module deliberately has no import from the paper path — paper trading
can never accidentally reach it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog

from engine.config import Settings
from engine.data.models import Side
from engine.ingestion.kalshi import KalshiSigner

logger = structlog.get_logger(__name__)

CONFIRMATION_PHRASE = "trade real money"


class LiveTradingBlockedError(RuntimeError):
    """Raised when an order is attempted without every gate open."""


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    side: Side
    contracts: int  # whole contracts only for real orders
    limit_price_cents: int

    def payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "action": "buy",
            "side": self.side.value,
            "count": self.contracts,
            "type": "limit",
            "yes_price": self.limit_price_cents
            if self.side is Side.YES
            else 100 - self.limit_price_cents,
        }


@dataclass(frozen=True)
class LiveGates:
    """Explicit record of which gates are open. Immutable once constructed."""

    env_enabled: bool
    cli_flag: bool
    credentials_present: bool
    confirmed_interactively: bool

    @property
    def all_open(self) -> bool:
        return (
            self.env_enabled
            and self.cli_flag
            and self.credentials_present
            and self.confirmed_interactively
        )

    def closed(self) -> list[str]:
        gates = {
            "ENGINE_LIVE_TRADING_ENABLED": self.env_enabled,
            "--live flag": self.cli_flag,
            "Kalshi credentials": self.credentials_present,
            "interactive confirmation": self.confirmed_interactively,
        }
        return [name for name, is_open in gates.items() if not is_open]


def gates_from(settings: Settings, *, cli_live_flag: bool, confirmation: str | None) -> LiveGates:
    return LiveGates(
        env_enabled=settings.live_trading_enabled,
        cli_flag=cli_live_flag,
        credentials_present=(
            settings.kalshi_api_key_id is not None and settings.kalshi_private_key_pem is not None
        ),
        confirmed_interactively=confirmation == CONFIRMATION_PHRASE,
    )


class LiveExecutor:
    """Builds signed Kalshi order requests; sends them only if every gate is open."""

    def __init__(self, settings: Settings, gates: LiveGates) -> None:
        self._settings = settings
        self._gates = gates
        self._signer: KalshiSigner | None = None
        if gates.credentials_present:
            key_id, pem = settings.require_kalshi_credentials()
            self._signer = KalshiSigner(key_id, pem.get_secret_value())

    async def place_order(self, order: OrderRequest) -> dict[str, Any]:
        """Dry-run by default: logs the exact would-be request. Only sends when
        every gate is open; raises LiveTradingBlockedError if asked to send otherwise."""
        payload = order.payload()
        if not self._gates.all_open:
            logger.info(
                "DRY RUN — order not sent",
                closed_gates=self._gates.closed(),
                request=json.dumps(payload),
            )
            return {"dry_run": True, "request": payload, "closed_gates": self._gates.closed()}
        return await self._send(order, payload)

    async def _send(self, order: OrderRequest, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        if self._signer is None:  # pragma: no cover — gates guarantee credentials
            raise LiveTradingBlockedError("credentials missing despite open gates")
        path = "/trade-api/v2/portfolio/orders"
        headers = self._signer.headers("POST", path)
        base = self._settings.kalshi_api_base.removesuffix("/trade-api/v2")
        logger.warning("LIVE ORDER SENDING", ticker=order.ticker, payload=json.dumps(payload))
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{base}{path}", json=payload, headers=headers)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
        logger.warning("LIVE ORDER RESPONSE", response=json.dumps(result))
        return result
