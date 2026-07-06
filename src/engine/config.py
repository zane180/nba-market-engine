"""Runtime configuration. Everything comes from the environment (or ``.env`` in dev);
nothing secret is ever read from the repo.

Kalshi credentials are optional because most of the system (historical ingestion,
modeling, backtesting) runs without them. Code paths that need auth must call
``Settings.require_kalshi_credentials`` and fail loudly rather than limping along.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- storage ---
    database_url: str = "sqlite:///data/engine.db"

    # --- Kalshi ---
    kalshi_api_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_api_key_id: str | None = None
    kalshi_private_key_pem: SecretStr | None = None

    # --- ESPN ---
    espn_api_base: str = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    # --- execution safety ---
    # Real-money order placement is additionally gated behind an explicit --live CLI
    # flag and an interactive confirmation; this setting alone can never place orders.
    live_trading_enabled: bool = False

    # --- reproducibility ---
    random_seed: int = 1337

    def require_kalshi_credentials(self) -> tuple[str, SecretStr]:
        if self.kalshi_api_key_id is None or self.kalshi_private_key_pem is None:
            raise RuntimeError(
                "Kalshi credentials missing: set ENGINE_KALSHI_API_KEY_ID and "
                "ENGINE_KALSHI_PRIVATE_KEY_PEM in the environment"
            )
        return self.kalshi_api_key_id, self.kalshi_private_key_pem


def load_settings() -> Settings:
    """Single construction point so tests can monkeypatch the environment cleanly."""
    return Settings()
