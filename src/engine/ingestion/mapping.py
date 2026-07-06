"""Deterministic ESPN <-> Kalshi team resolution.

ESPN and Kalshi do not share identifiers: ESPN uses full display names and its
own abbreviations (``GS``, ``NY``, ``SA``, ``UTAH``, ``WSH``); Kalshi identifies
teams by city name and uses standard NBA tricodes in tickers (``GSW``, ``NYK``,
``SAS``, ``UTA``, ``WAS``). Guessing the join is how a Lakers trade lands on a
Clippers market, so:

- every row below was verified against live data on 2026-07-06 — ESPN's
  ``/teams`` endpoint on one side, real Kalshi event tickers (playoff markets
  plus archived regular-season events, e.g. ``KXNBAGAME-26JAN11ATLGSW``) on the
  other;
- lookups on anything unmapped raise ``UnmappedTeamError`` — never a default.

Kalshi game tickers embed the game's *US-Eastern* calendar date and the
away+home tricodes: ``KXNBAGAME-26JUN13NYKSAS`` is New York at San Antonio on
Jun 13 2026 (ET), and its two markets append the team that must win:
``...-SAS`` / ``...-NYK``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from engine.ingestion.kalshi import NBA_GAME_SERIES

_EASTERN = ZoneInfo("America/New_York")


class UnmappedTeamError(Exception):
    def __init__(self, kind: str, value: str) -> None:
        super().__init__(
            f"no NBA team with {kind} {value!r} — if this is a real team identifier, "
            "the mapping table in engine/ingestion/mapping.py needs re-verification"
        )


@dataclass(frozen=True, slots=True)
class Team:
    espn_name: str  # ESPN team.displayName
    espn_abbrev: str  # ESPN team.abbreviation
    kalshi_abbrev: str  # tricode used in Kalshi tickers


TEAMS: tuple[Team, ...] = (
    Team("Atlanta Hawks", "ATL", "ATL"),
    Team("Boston Celtics", "BOS", "BOS"),
    Team("Brooklyn Nets", "BKN", "BKN"),
    Team("Charlotte Hornets", "CHA", "CHA"),
    Team("Chicago Bulls", "CHI", "CHI"),
    Team("Cleveland Cavaliers", "CLE", "CLE"),
    Team("Dallas Mavericks", "DAL", "DAL"),
    Team("Denver Nuggets", "DEN", "DEN"),
    Team("Detroit Pistons", "DET", "DET"),
    Team("Golden State Warriors", "GS", "GSW"),
    Team("Houston Rockets", "HOU", "HOU"),
    Team("Indiana Pacers", "IND", "IND"),
    Team("LA Clippers", "LAC", "LAC"),
    Team("Los Angeles Lakers", "LAL", "LAL"),
    Team("Memphis Grizzlies", "MEM", "MEM"),
    Team("Miami Heat", "MIA", "MIA"),
    Team("Milwaukee Bucks", "MIL", "MIL"),
    Team("Minnesota Timberwolves", "MIN", "MIN"),
    Team("New Orleans Pelicans", "NO", "NOP"),
    Team("New York Knicks", "NY", "NYK"),
    Team("Oklahoma City Thunder", "OKC", "OKC"),
    Team("Orlando Magic", "ORL", "ORL"),
    Team("Philadelphia 76ers", "PHI", "PHI"),
    Team("Phoenix Suns", "PHX", "PHX"),
    Team("Portland Trail Blazers", "POR", "POR"),
    Team("Sacramento Kings", "SAC", "SAC"),
    Team("San Antonio Spurs", "SA", "SAS"),
    Team("Toronto Raptors", "TOR", "TOR"),
    Team("Utah Jazz", "UTAH", "UTA"),
    Team("Washington Wizards", "WSH", "WAS"),
)

_BY_ESPN_NAME = {t.espn_name: t for t in TEAMS}
_BY_KALSHI_ABBREV = {t.kalshi_abbrev: t for t in TEAMS}


def team_from_espn_name(espn_name: str) -> Team:
    team = _BY_ESPN_NAME.get(espn_name)
    if team is None:
        raise UnmappedTeamError("ESPN name", espn_name)
    return team


def team_from_kalshi_abbrev(abbrev: str) -> Team:
    team = _BY_KALSHI_ABBREV.get(abbrev)
    if team is None:
        raise UnmappedTeamError("Kalshi abbreviation", abbrev)
    return team


def game_event_ticker(*, away_espn_name: str, home_espn_name: str, tipoff: datetime) -> str:
    """Kalshi event ticker for a game, from ESPN identifiers and the UTC tipoff.

    The embedded date is the tipoff's US-Eastern calendar date (verified: a
    19:00Z Jan 15 tipoff -> ``26JAN15``; an evening ET game whose UTC time has
    rolled past midnight keeps the ET date).
    """
    if tipoff.tzinfo is None:
        raise ValueError("tipoff must be timezone-aware")
    away = team_from_espn_name(away_espn_name)
    home = team_from_espn_name(home_espn_name)
    local = tipoff.astimezone(_EASTERN)
    date_part = local.strftime("%y%b%d").upper()
    return f"{NBA_GAME_SERIES}-{date_part}{away.kalshi_abbrev}{home.kalshi_abbrev}"


def market_ticker(event_ticker: str, *, winner_espn_name: str) -> str:
    """The market within an event that pays out if ``winner_espn_name`` wins."""
    return f"{event_ticker}-{team_from_espn_name(winner_espn_name).kalshi_abbrev}"


_EVENT_TICKER_RE = re.compile(r"^KXNBAGAME-\d{2}[A-Z]{3}\d{2}([A-Z]+)$")


def teams_from_event_ticker(event_ticker: str) -> tuple[Team, Team]:
    """Parse ``(away, home)`` back out of an event ticker's matchup segment.

    The matchup segment is the two tricodes concatenated. A greedy prefix split
    is safe: no verified tricode is a prefix of another, and both halves must be
    known tricodes for the split to be accepted.
    """
    match = _EVENT_TICKER_RE.match(event_ticker)
    if match is None:
        raise UnmappedTeamError("event ticker", event_ticker)
    teams_segment = match.group(1)
    for away_abbrev in _BY_KALSHI_ABBREV:
        if teams_segment.startswith(away_abbrev):
            home_abbrev = teams_segment[len(away_abbrev) :]
            if home_abbrev in _BY_KALSHI_ABBREV:
                return _BY_KALSHI_ABBREV[away_abbrev], _BY_KALSHI_ABBREV[home_abbrev]
    raise UnmappedTeamError("event ticker matchup", teams_segment)
