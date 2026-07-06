"""ESPN NBA ingestion.

ESPN's endpoints are undocumented and can drift, so this module is split into:

- pure parsers (``parse_scoreboard``, ``parse_summary_snapshots``) that are unit-
  tested against real captured payloads and raise ``SchemaDriftError`` on any
  shape we don't positively recognize — never a silent None; and
- ``EspnClient``, a thin async fetch layer over ``RetryingClient`` that caches
  immutable resources (past scoreboards, summaries of final games).

Payload facts verified against the live API on 2026-07-06:
- scoreboard: ``events[].competitions[0].competitors[]`` with ``homeAway``,
  ``team.displayName``, string ``score``; event ``date`` like ``2026-01-15T19:00Z``.
- summary: ``plays[]`` with ``period.number``, ``clock.displayValue`` ("10:04"),
  ``homeScore``/``awayScore`` (ints), and ``wallclock`` (ISO, Z) — the wallclock is
  what lets historical in-game snapshots be joined to market data without lookahead.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

import structlog

from engine.data.models import Game, GameStatus, LiveGameState
from engine.ingestion.http import RetryingClient

logger = structlog.get_logger(__name__)


class SchemaDriftError(Exception):
    """ESPN returned a shape we don't recognize. Fail loudly: a guessed parse of
    market-adjacent data is worse than no data."""


_STATUS_BY_NAME = {
    "STATUS_SCHEDULED": GameStatus.SCHEDULED,
    "STATUS_IN_PROGRESS": GameStatus.IN_PROGRESS,
    "STATUS_HALFTIME": GameStatus.IN_PROGRESS,
    "STATUS_END_PERIOD": GameStatus.IN_PROGRESS,
    "STATUS_FINAL": GameStatus.FINAL,
    "STATUS_POSTPONED": GameStatus.POSTPONED,
    "STATUS_CANCELED": GameStatus.CANCELED,
}
_STATUS_BY_STATE = {
    "pre": GameStatus.SCHEDULED,
    "in": GameStatus.IN_PROGRESS,
    "post": GameStatus.FINAL,
}

_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d+)(?:\.(\d+))?$")


def parse_clock(display: str) -> float:
    """ESPN clock display -> seconds remaining. Handles "10:04", "45.3", "0.8"."""
    match = _CLOCK_RE.match(display.strip())
    if match is None:
        raise SchemaDriftError(f"unparseable clock display {display!r}")
    minutes, seconds, fraction = match.groups()
    total = float(int(seconds) + 60 * int(minutes or 0))
    if fraction:
        total += float(f"0.{fraction}")
    return total


def _parse_espn_datetime(raw: str) -> datetime:
    """ESPN uses minute-precision ISO with a literal Z, e.g. '2026-01-15T19:00Z'."""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaDriftError(f"unparseable datetime {raw!r}") from exc
    if parsed.tzinfo is None:
        raise SchemaDriftError(f"ESPN datetime without timezone: {raw!r}")
    return parsed.astimezone(UTC)


def _parse_status(status_type: dict[str, Any], *, context: str) -> GameStatus:
    name = status_type.get("name")
    if isinstance(name, str) and name in _STATUS_BY_NAME:
        return _STATUS_BY_NAME[name]
    state = status_type.get("state")
    if isinstance(state, str) and state in _STATUS_BY_STATE:
        logger.warning("unknown ESPN status name, using state", name=name, state=state)
        return _STATUS_BY_STATE[state]
    raise SchemaDriftError(f"{context}: unrecognized status {status_type!r}")


def parse_scoreboard(payload: dict[str, Any]) -> list[Game]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise SchemaDriftError("scoreboard payload has no 'events' list")

    games: list[Game] = []
    for event in events:
        game_id = str(event["id"])
        context = f"event {game_id}"
        try:
            competition = event["competitions"][0]
        except (KeyError, IndexError) as exc:
            raise SchemaDriftError(f"{context}: no competitions") from exc

        sides: dict[str, tuple[str, int]] = {}
        for competitor in competition["competitors"]:
            side = competitor.get("homeAway")
            if side not in ("home", "away"):
                raise SchemaDriftError(f"{context}: competitor homeAway={side!r}")
            name = competitor.get("team", {}).get("displayName")
            if not isinstance(name, str) or not name:
                raise SchemaDriftError(f"{context}: competitor without team.displayName")
            score = int(competitor.get("score") or 0)
            sides[side] = (name, score)
        if set(sides) != {"home", "away"}:
            raise SchemaDriftError(f"{context}: expected exactly one home and one away side")

        status = _parse_status(event["status"]["type"], context=context)
        raw_season_type = event.get("season", {}).get("type")
        games.append(
            Game(
                game_id=game_id,
                home_team=sides["home"][0],
                away_team=sides["away"][0],
                start_time=_parse_espn_datetime(event["date"]),
                status=status,
                # a scheduled game legitimately has no score yet
                home_score=0 if status is GameStatus.SCHEDULED else sides["home"][1],
                away_score=0 if status is GameStatus.SCHEDULED else sides["away"][1],
                season_type=int(raw_season_type) if raw_season_type is not None else None,
            )
        )
    return games


def parse_summary_snapshots(payload: dict[str, Any], *, game_id: str) -> list[LiveGameState]:
    """Play-by-play -> in-game snapshots, one per play that carries a wallclock.

    Plays without a wallclock are dropped (and counted) rather than guessed: a
    snapshot with an invented timestamp is a lookahead bug waiting to happen.
    """
    plays = payload.get("plays")
    if not isinstance(plays, list):
        raise SchemaDriftError(f"summary for {game_id} has no 'plays' list")

    snapshots: list[LiveGameState] = []
    dropped = 0
    for play in plays:
        wallclock = play.get("wallclock")
        if not wallclock:
            dropped += 1
            continue
        try:
            period = int(play["period"]["number"])
            clock_display = play["clock"]["displayValue"]
            home_score = int(play["homeScore"])
            away_score = int(play["awayScore"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaDriftError(f"summary for {game_id}: bad play {play.get('id')}") from exc
        play_id = play.get("id")
        snapshots.append(
            LiveGameState(
                game_id=game_id,
                as_of=_parse_espn_datetime(wallclock),
                period=period,
                seconds_remaining_in_period=parse_clock(clock_display),
                home_score=home_score,
                away_score=away_score,
                source_play_id=str(play_id) if play_id is not None else None,
            )
        )
    if dropped:
        logger.info("dropped plays without wallclock", game_id=game_id, dropped=dropped)
    return snapshots


class EspnClient:
    """Fetch layer. Past dates and final games are immutable, so they're cached."""

    def __init__(self, http: RetryingClient, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def scoreboard(self, day: date | None = None) -> list[Game]:
        params: dict[str, str | int] = {}
        cache_key: str | None = None
        if day is not None:
            params["dates"] = day.strftime("%Y%m%d")
            if day < datetime.now(UTC).date():  # yesterday-or-older can't change
                cache_key = f"espn:scoreboard:{params['dates']}"
        payload = await self._http.get_json(
            f"{self._base}/scoreboard", params=params, cache_key=cache_key
        )
        return parse_scoreboard(payload)

    async def game_snapshots(self, game_id: str, *, final: bool = False) -> list[LiveGameState]:
        """Historical (or in-progress) play-by-play snapshots for one game.
        Pass ``final=True`` to cache — only a finished game's plays are immutable."""
        payload = await self._http.get_json(
            f"{self._base}/summary",
            params={"event": game_id},
            cache_key=f"espn:summary:{game_id}" if final else None,
        )
        return parse_summary_snapshots(payload, game_id=game_id)
