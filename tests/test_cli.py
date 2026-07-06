from __future__ import annotations

import pytest
from typer.testing import CliRunner

from engine.cli import _NOT_BUILT, app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "backtest", "paper", "config"):
        assert command in result.output


@pytest.mark.parametrize("command", ["ingest", "backtest", "paper"])
def test_unbuilt_commands_say_so_and_fail(command: str) -> None:
    """Stubs must exit nonzero — a silent success would let CI/scripts pass on a
    pipeline stage that doesn't exist."""
    result = runner.invoke(app, [command])
    assert result.exit_code == _NOT_BUILT
    assert "not implemented" in result.output


def test_config_prints_settings_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_KALSHI_PRIVATE_KEY_PEM", "-----BEGIN PRIVATE KEY-----abc")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "database_url=" in result.output
    assert "BEGIN PRIVATE KEY" not in result.output
