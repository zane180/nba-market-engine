from __future__ import annotations

import pytest
from typer.testing import CliRunner

from engine.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "backtest", "paper", "report", "config"):
        assert command in result.output


def test_ingest_group_lists_subcommands() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    for sub in ("games", "markets", "candles", "snapshots", "verify", "all"):
        assert sub in result.output


def test_config_prints_settings_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_KALSHI_PRIVATE_KEY_PEM", "-----BEGIN PRIVATE KEY-----abc")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "database_url=" in result.output
    assert "BEGIN PRIVATE KEY" not in result.output
