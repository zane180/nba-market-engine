from __future__ import annotations

import pytest
from pydantic import SecretStr

from engine.config import Settings


def make_settings(**overrides: object) -> Settings:
    # _env_file=None keeps tests hermetic even if a developer has a local .env.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_live_trading_is_off_by_default() -> None:
    """Safety-critical default: a fresh environment must never be live-tradable."""
    assert make_settings().live_trading_enabled is False


def test_env_overrides_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_DATABASE_URL", "postgresql://x/y")
    assert make_settings().database_url == "postgresql://x/y"


def test_missing_kalshi_credentials_fail_loudly() -> None:
    with pytest.raises(RuntimeError, match="ENGINE_KALSHI_API_KEY_ID"):
        make_settings().require_kalshi_credentials()


def test_present_kalshi_credentials_are_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_KALSHI_API_KEY_ID", "key-123")
    monkeypatch.setenv("ENGINE_KALSHI_PRIVATE_KEY_PEM", "-----BEGIN PRIVATE KEY-----abc")
    key_id, pem = make_settings().require_kalshi_credentials()
    assert key_id == "key-123"
    assert pem.get_secret_value().startswith("-----BEGIN")


def test_private_key_never_appears_in_repr_or_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_KALSHI_PRIVATE_KEY_PEM", "-----BEGIN PRIVATE KEY-----abc")
    settings = make_settings()
    assert "BEGIN PRIVATE KEY" not in repr(settings)
    assert "BEGIN PRIVATE KEY" not in str(settings.model_dump())
    assert isinstance(settings.kalshi_private_key_pem, SecretStr)
