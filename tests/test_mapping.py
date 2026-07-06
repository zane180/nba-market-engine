"""Team-mapping correctness. The reference points here are REAL Kalshi event
tickers (captured 2026-07-06), so a regression that would misroute a trade to
the wrong team's market fails against ground truth, not against our own table.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.ingestion.mapping import (
    TEAMS,
    Team,
    UnmappedTeamError,
    game_event_ticker,
    market_ticker,
    team_from_espn_name,
    team_from_kalshi_abbrev,
    teams_from_event_ticker,
)

# (away espn name, home espn name, tipoff UTC, real archived Kalshi event ticker)
REAL_EVENTS = [
    (
        "New York Knicks",
        "San Antonio Spurs",
        datetime(2026, 6, 14, 0, 30, tzinfo=UTC),  # ~8:30pm ET Jun 13 -> UTC rolls to Jun 14
        "KXNBAGAME-26JUN13NYKSAS",
    ),
    (
        "Memphis Grizzlies",
        "Orlando Magic",
        datetime(2026, 1, 15, 19, 0, tzinfo=UTC),
        "KXNBAGAME-26JAN15MEMORL",
    ),
    (
        "Atlanta Hawks",
        "Golden State Warriors",
        datetime(2026, 1, 12, 1, 0, tzinfo=UTC),  # 8pm ET Jan 11 game, UTC date Jan 12
        "KXNBAGAME-26JAN11ATLGSW",
    ),
    (
        "Dallas Mavericks",
        "Chicago Bulls",
        datetime(2026, 1, 11, 1, 0, tzinfo=UTC),
        "KXNBAGAME-26JAN10DALCHI",
    ),
]


class TestTable:
    def test_thirty_teams(self) -> None:
        assert len(TEAMS) == 30

    def test_keys_are_unique(self) -> None:
        for attr in ("espn_name", "espn_abbrev", "kalshi_abbrev"):
            values = [getattr(t, attr) for t in TEAMS]
            assert len(set(values)) == 30, f"duplicate {attr}"

    def test_no_tricode_is_a_prefix_of_another(self) -> None:
        """Required for the greedy event-ticker split to be unambiguous."""
        codes = [t.kalshi_abbrev for t in TEAMS]
        clashes = [(a, b) for a in codes for b in codes if a != b and b.startswith(a)]
        assert clashes == []

    def test_divergent_abbreviations_are_the_verified_ones(self) -> None:
        """The six teams where ESPN and Kalshi disagree — the whole reason this
        module exists. Values verified against live APIs 2026-07-06."""
        divergent = {
            t.espn_abbrev: t.kalshi_abbrev for t in TEAMS if t.espn_abbrev != t.kalshi_abbrev
        }
        assert divergent == {
            "GS": "GSW",
            "NO": "NOP",
            "NY": "NYK",
            "SA": "SAS",
            "UTAH": "UTA",
            "WSH": "WAS",
        }


class TestLookups:
    def test_roundtrip(self) -> None:
        for team in TEAMS:
            assert team_from_espn_name(team.espn_name) is team
            assert team_from_kalshi_abbrev(team.kalshi_abbrev) is team

    @pytest.mark.parametrize(
        ("fn", "value"),
        [
            (team_from_espn_name, "Seattle SuperSonics"),
            (team_from_espn_name, "Los Angeles Clippers"),  # ESPN says "LA Clippers"
            (team_from_kalshi_abbrev, "GS"),  # ESPN code, not Kalshi's
            (team_from_kalshi_abbrev, ""),
        ],
    )
    def test_unmapped_fails_loudly(self, fn: object, value: str) -> None:
        with pytest.raises(UnmappedTeamError):
            fn(value)  # type: ignore[operator]


class TestEventTickers:
    @pytest.mark.parametrize(("away", "home", "tipoff", "expected"), REAL_EVENTS)
    def test_construction_matches_real_archived_events(
        self, away: str, home: str, tipoff: datetime, expected: str
    ) -> None:
        assert (
            game_event_ticker(away_espn_name=away, home_espn_name=home, tipoff=tipoff) == expected
        )

    def test_utc_midnight_rollover_keeps_eastern_date(self) -> None:
        """A 9pm ET game is 02:00Z the next day; the ticker must use the ET date."""
        ticker = game_event_ticker(
            away_espn_name="Boston Celtics",
            home_espn_name="Los Angeles Lakers",
            tipoff=datetime(2026, 3, 2, 2, 0, tzinfo=UTC),
        )
        assert ticker == "KXNBAGAME-26MAR01BOSLAL"

    def test_naive_tipoff_rejected(self) -> None:
        with pytest.raises(ValueError, match="aware"):
            game_event_ticker(
                away_espn_name="Boston Celtics",
                home_espn_name="Miami Heat",
                tipoff=datetime(2026, 3, 1, 19, 0),  # noqa: DTZ001 — naive on purpose
            )

    def test_market_ticker_appends_winner_tricode(self) -> None:
        event = "KXNBAGAME-26JUN13NYKSAS"
        assert market_ticker(event, winner_espn_name="San Antonio Spurs") == f"{event}-SAS"
        assert market_ticker(event, winner_espn_name="New York Knicks") == f"{event}-NYK"

    @pytest.mark.parametrize(("away", "home", "tipoff", "ticker"), REAL_EVENTS)
    def test_parse_teams_back_out(
        self, away: str, home: str, tipoff: datetime, ticker: str
    ) -> None:
        parsed_away, parsed_home = teams_from_event_ticker(ticker)
        assert (parsed_away.espn_name, parsed_home.espn_name) == (away, home)

    @pytest.mark.parametrize(
        "bad",
        [
            "KXNBAGAME-26JUN13NYKXXX",  # unknown home tricode
            "KXNBAGAME-26JUN13NYK",  # missing home team
            "KXNBA-27-TOR",  # different series
            "garbage",
        ],
    )
    def test_unparseable_event_ticker_fails_loudly(self, bad: str) -> None:
        with pytest.raises(UnmappedTeamError):
            teams_from_event_ticker(bad)


def test_team_is_immutable() -> None:
    with pytest.raises(AttributeError):
        Team("A", "B", "C").espn_name = "X"  # type: ignore[misc]
