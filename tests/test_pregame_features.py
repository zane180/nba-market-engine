"""Feature-builder correctness, headlined by the explicit no-leakage guard."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from engine.data.models import Game, GameStatus
from engine.features.pregame import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    FEATURE_NAMES,
    build_feature_rows,
    elo_expected_home,
    season_of,
)

T0 = datetime(2024, 10, 22, 0, 0, tzinfo=UTC)
TEAMS = ["Boston Celtics", "Miami Heat", "Denver Nuggets", "Phoenix Suns"]


def make_game(
    idx: int,
    home: str,
    away: str,
    *,
    home_score: int,
    away_score: int,
    start: datetime | None = None,
    season_type: int = 2,
) -> Game:
    return Game(
        game_id=f"g{idx}",
        home_team=home,
        away_team=away,
        start_time=start if start is not None else T0 + timedelta(days=idx),
        status=GameStatus.FINAL,
        home_score=home_score,
        away_score=away_score,
        season_type=season_type,
    )


def random_schedule(n: int, *, start: datetime, seed: int = 7) -> list[Game]:
    rng = random.Random(seed)
    games = []
    for i in range(n):
        home, away = rng.sample(TEAMS, 2)
        winner_is_home = rng.random() < 0.55
        hs = rng.randint(95, 130)
        as_ = hs - rng.randint(1, 20) if winner_is_home else hs + rng.randint(1, 20)
        games.append(
            make_game(i, home, away, home_score=hs, away_score=as_, start=start + timedelta(days=i))
        )
    return games


class TestNoLeakage:
    """THE guard: a game's features must be identical whether or not any
    later game (including its own result) exists in the input. If someone
    accidentally lets a result update state before the feature row is
    emitted, or sorts wrong, this fails."""

    def test_features_unchanged_by_future_games(self) -> None:
        games = random_schedule(60, start=T0)
        full = {r.game_id: r for r in build_feature_rows(games)}
        for cut in (1, 15, 37, 59):
            truncated = build_feature_rows(games[: cut + 1])
            target = truncated[-1]  # row for games[cut], with no future data present
            assert target.features == full[target.game_id].features, (
                f"features for game {target.game_id} changed when future games were added — "
                "lookahead leakage"
            )
            assert target.label_home_won == full[target.game_id].label_home_won

    def test_first_meeting_is_uninformed(self) -> None:
        rows = build_feature_rows(random_schedule(10, start=T0))
        first = rows[0]
        assert first.features["elo_diff"] == 0.0
        assert first.features["elo_home_prob"] == pytest.approx(
            elo_expected_home(ELO_INITIAL, ELO_INITIAL)
        )
        assert first.features["form_home"] == 0.5  # no history -> neutral prior, not a peek

    def test_non_final_games_contribute_nothing(self) -> None:
        games = random_schedule(20, start=T0)
        scheduled = Game(
            game_id="future",
            home_team=TEAMS[0],
            away_team=TEAMS[1],
            start_time=T0 + timedelta(days=5, hours=3),
            status=GameStatus.SCHEDULED,
        )
        with_scheduled = build_feature_rows([*games, scheduled])
        without = build_feature_rows(games)
        assert [r.features for r in with_scheduled] == [r.features for r in without]


class TestEloDynamics:
    def test_winner_gains_loser_drops(self) -> None:
        games = [
            make_game(0, TEAMS[0], TEAMS[1], home_score=120, away_score=100),
            make_game(1, TEAMS[0], TEAMS[1], home_score=110, away_score=105),
        ]
        rows = build_feature_rows(games)
        # after game 0 (home win), home team's elo must exceed away's
        assert rows[1].features["elo_diff"] > 0

    def test_home_advantage_present(self) -> None:
        assert elo_expected_home(1500, 1500) > 0.5
        assert elo_expected_home(1500 - ELO_HOME_ADVANTAGE, 1500) == pytest.approx(0.5)

    def test_season_boundary_regresses_and_resets(self) -> None:
        season_1 = random_schedule(40, start=T0)
        cross = make_game(
            99,
            TEAMS[0],
            TEAMS[1],
            home_score=100,
            away_score=90,
            start=datetime(2025, 10, 21, 0, 0, tzinfo=UTC),  # next season
        )
        rows = build_feature_rows([*season_1, cross])
        last = rows[-1]
        # rest/form/games_played reset at the boundary
        assert last.features["games_played_home"] == 0.0
        assert last.features["form_home"] == 0.5
        assert last.features["rest_days_home"] == 10.0
        # elo gap shrinks by exactly the 25% regression: compare against the
        # same matchup replayed just before the boundary
        replay = make_game(
            98, TEAMS[0], TEAMS[1], home_score=100, away_score=90, start=T0 + timedelta(days=41)
        )
        same_season_rows = build_feature_rows([*season_1, replay])
        pre_boundary_diff = same_season_rows[-1].features["elo_diff"]
        assert abs(last.features["elo_diff"]) == pytest.approx(abs(pre_boundary_diff) * 0.75)

    def test_rest_days_and_back_to_back(self) -> None:
        games = [
            make_game(0, TEAMS[0], TEAMS[1], home_score=100, away_score=90),
            make_game(
                1,
                TEAMS[0],
                TEAMS[2],
                home_score=100,
                away_score=90,
                start=T0 + timedelta(days=1),
            ),
        ]
        rows = build_feature_rows(games)
        assert rows[1].features["rest_days_home"] == pytest.approx(1.0)
        assert rows[1].features["back_to_back_home"] == 1.0
        assert rows[1].features["rest_days_away"] == 10.0  # capped first appearance


class TestShape:
    def test_every_row_has_every_feature(self) -> None:
        rows = build_feature_rows(random_schedule(25, start=T0))
        for row in rows:
            assert set(row.features) == set(FEATURE_NAMES)

    def test_chronological_output(self) -> None:
        games = random_schedule(25, start=T0)
        rows = build_feature_rows(list(reversed(games)))  # input order must not matter
        assert [r.start_time for r in rows] == sorted(r.start_time for r in rows)

    def test_season_of(self) -> None:
        assert season_of(datetime(2025, 10, 22, tzinfo=UTC)) == 2026
        assert season_of(datetime(2026, 6, 14, tzinfo=UTC)) == 2026
        assert season_of(datetime(2026, 7, 1, tzinfo=UTC)) == 2026
