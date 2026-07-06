"""ESPN parser tests against real captured payloads (tests/fixtures/)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from engine.data.models import GameStatus
from engine.ingestion.espn import (
    SchemaDriftError,
    parse_clock,
    parse_scoreboard,
    parse_summary_snapshots,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def scoreboard() -> dict[str, Any]:
    return json.loads((FIXTURES / "espn_scoreboard.json").read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return json.loads((FIXTURES / "espn_summary.json").read_text())  # type: ignore[no-any-return]


class TestParseScoreboard:
    def test_parses_real_payload(self, scoreboard: dict[str, Any]) -> None:
        games = parse_scoreboard(scoreboard)
        assert len(games) == 3
        game = games[0]
        assert game.game_id == "401810433"
        assert game.home_team == "Orlando Magic"
        assert game.away_team == "Memphis Grizzlies"
        assert game.status is GameStatus.FINAL
        assert (game.home_score, game.away_score) == (118, 111)
        assert game.home_won is True

    def test_timestamps_are_utc_aware(self, scoreboard: dict[str, Any]) -> None:
        game = parse_scoreboard(scoreboard)[0]
        assert game.start_time == datetime(2026, 1, 15, 19, 0, tzinfo=UTC)
        assert game.start_time.tzinfo is UTC

    def test_unknown_status_raises_schema_drift(self, scoreboard: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(scoreboard))  # deep copy
        payload["events"][0]["status"]["type"] = {"name": "STATUS_RAIN_DELAY", "state": "??"}
        with pytest.raises(SchemaDriftError, match="unrecognized status"):
            parse_scoreboard(payload)

    def test_unknown_name_with_known_state_falls_back(self, scoreboard: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(scoreboard))
        payload["events"][0]["status"]["type"] = {"name": "STATUS_SOMETHING_NEW", "state": "in"}
        assert parse_scoreboard(payload)[0].status is GameStatus.IN_PROGRESS

    def test_missing_events_key_raises(self) -> None:
        with pytest.raises(SchemaDriftError, match="events"):
            parse_scoreboard({"leagues": []})

    def test_missing_home_side_raises(self, scoreboard: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(scoreboard))
        for competitor in payload["events"][0]["competitions"][0]["competitors"]:
            competitor["homeAway"] = "away"
        with pytest.raises(SchemaDriftError, match="home and one away"):
            parse_scoreboard(payload)


class TestParseClock:
    @pytest.mark.parametrize(
        ("display", "expected"),
        [("10:04", 604.0), ("12:00", 720.0), ("0:00", 0.0), ("45.3", 45.3), ("0.8", 0.8)],
    )
    def test_formats(self, display: str, expected: float) -> None:
        assert parse_clock(display) == expected

    def test_garbage_raises(self) -> None:
        with pytest.raises(SchemaDriftError):
            parse_clock("End of Period")


class TestParseSummarySnapshots:
    def test_parses_real_plays(self, summary: dict[str, Any]) -> None:
        snaps = parse_summary_snapshots(summary, game_id="401810433")
        assert len(snaps) == 5
        mid = snaps[2]
        assert mid.period == 3
        assert mid.as_of.tzinfo is UTC
        assert mid.home_score >= 0 and mid.away_score >= 0

    def test_snapshots_carry_wallclock_not_invented_time(self, summary: dict[str, Any]) -> None:
        """as_of must come from ESPN's wallclock — the join key that prevents
        lookahead when aligning with market candles."""
        first = parse_summary_snapshots(summary, game_id="401810433")[0]
        raw_first_wallclock = summary["plays"][0]["wallclock"]
        expected = datetime.fromisoformat(raw_first_wallclock.replace("Z", "+00:00"))
        assert first.as_of == expected

    def test_play_without_wallclock_is_dropped_not_guessed(self, summary: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(summary))
        del payload["plays"][0]["wallclock"]
        assert len(parse_summary_snapshots(payload, game_id="x")) == 4

    def test_malformed_play_raises(self, summary: dict[str, Any]) -> None:
        payload = json.loads(json.dumps(summary))
        payload["plays"][1]["period"] = {}
        with pytest.raises(SchemaDriftError, match="bad play"):
            parse_summary_snapshots(payload, game_id="x")
