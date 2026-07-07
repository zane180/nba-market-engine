"""Pre-game features, leakage-safe by construction.

``build_feature_rows`` walks games in strict chronological order; each game's
row is emitted *before* that game's result touches any running state (Elo,
rest, form). Nothing here reads a score until the game it belongs to has
already produced its feature row — the property pinned by the explicit
no-leakage test in tests/test_pregame_features.py.

Elo follows the well-known FiveThirtyEight NBA construction: K=20 with a
margin-of-victory multiplier, ~100 rating points of home-court advantage, and
25% regression toward the mean across season boundaries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from engine.data.models import Game, GameStatus

ELO_INITIAL = 1500.0
ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
SEASON_CARRYOVER = 0.75
FORM_WINDOW = 10
REST_CAP_DAYS = 10.0

FEATURE_NAMES: tuple[str, ...] = (
    "elo_diff",
    "elo_home_prob",
    "rest_days_home",
    "rest_days_away",
    "back_to_back_home",
    "back_to_back_away",
    "form_home",
    "form_away",
    "games_played_home",
    "games_played_away",
    "is_playoffs",
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Everything the pre-game models see about one game, plus the label."""

    game_id: str
    start_time: datetime
    season: int  # season end year: 2025-26 -> 2026
    home_team: str
    away_team: str
    label_home_won: bool
    features: dict[str, float]

    @property
    def elo_home_prob(self) -> float:
        return self.features["elo_home_prob"]


def season_of(start_time: datetime) -> int:
    """Season end year; NBA seasons span Oct-Jun (Aug+ counts toward next)."""
    return start_time.year + 1 if start_time.month >= 8 else start_time.year


def elo_expected_home(home_elo: float, away_elo: float) -> float:
    diff = home_elo + ELO_HOME_ADVANTAGE - away_elo
    return float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier: bigger blowouts move
    ratings more, damped when the winner was already favored."""
    return float(((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner))


class _TeamState:
    __slots__ = ("elo", "games_played", "last_game", "recent")

    def __init__(self) -> None:
        self.elo = ELO_INITIAL
        self.last_game: datetime | None = None
        self.recent: deque[bool] = deque(maxlen=FORM_WINDOW)
        self.games_played = 0


def _rest_days(state: _TeamState, tipoff: datetime) -> float:
    if state.last_game is None:
        return REST_CAP_DAYS
    return min(REST_CAP_DAYS, (tipoff - state.last_game).total_seconds() / 86_400.0)


def build_feature_rows(games: list[Game]) -> list[FeatureRow]:
    """Feature rows for every FINAL, decidable game, in chronological order.

    Input may be any order/status; non-final games are skipped (they carry no
    label and must not touch Elo state either).
    """
    finals = sorted(
        (g for g in games if g.status is GameStatus.FINAL and g.home_score != g.away_score),
        key=lambda g: (g.start_time, g.game_id),
    )
    states: dict[str, _TeamState] = {}
    current_season: int | None = None
    rows: list[FeatureRow] = []

    for game in finals:
        season = season_of(game.start_time)
        if current_season is not None and season != current_season:
            for state in states.values():  # new season: regress toward mean
                state.elo = SEASON_CARRYOVER * state.elo + (1 - SEASON_CARRYOVER) * ELO_MEAN
                state.recent.clear()
                state.last_game = None
                state.games_played = 0
        current_season = season

        home = states.setdefault(game.home_team, _TeamState())
        away = states.setdefault(game.away_team, _TeamState())

        # ---- emit features BEFORE this game's result updates any state ----
        rest_home = _rest_days(home, game.start_time)
        rest_away = _rest_days(away, game.start_time)
        features = {
            "elo_diff": home.elo - away.elo,
            "elo_home_prob": elo_expected_home(home.elo, away.elo),
            "rest_days_home": rest_home,
            "rest_days_away": rest_away,
            "back_to_back_home": float(rest_home <= 1.25),
            "back_to_back_away": float(rest_away <= 1.25),
            "form_home": sum(home.recent) / len(home.recent) if home.recent else 0.5,
            "form_away": sum(away.recent) / len(away.recent) if away.recent else 0.5,
            "games_played_home": float(home.games_played),
            "games_played_away": float(away.games_played),
            "is_playoffs": float(game.season_type == 3),
        }
        rows.append(
            FeatureRow(
                game_id=game.game_id,
                start_time=game.start_time,
                season=season,
                home_team=game.home_team,
                away_team=game.away_team,
                label_home_won=game.home_won,
                features=features,
            )
        )

        # ---- now update state with the result ----
        expected = features["elo_home_prob"]
        outcome = 1.0 if game.home_won else 0.0
        margin = game.home_score - game.away_score
        winner_elo_diff = (
            home.elo + ELO_HOME_ADVANTAGE - away.elo
            if game.home_won
            else away.elo - (home.elo + ELO_HOME_ADVANTAGE)
        )
        shift = ELO_K * _mov_multiplier(margin, winner_elo_diff) * (outcome - expected)
        home.elo += shift
        away.elo -= shift
        for state, won in ((home, game.home_won), (away, not game.home_won)):
            state.recent.append(won)
            state.last_game = game.start_time
            state.games_played += 1

    return rows
