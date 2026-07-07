"""Model-layer discipline: walk-forward boundaries, calibration hygiene,
determinism, and refusal to train on too little data."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engine.backtest.metrics import expected_calibration_error
from engine.data.models import Game, GameStatus
from engine.features.pregame import FeatureRow, build_feature_rows
from engine.models.calibration import IsotonicCalibrator
from engine.models.pregame_model import (
    GbmModel,
    elo_probabilities,
    walk_forward_by_season,
)


def synthetic_rows(n_per_season: int, seasons: list[int], seed: int = 11) -> list[FeatureRow]:
    """Two seasons of synthetic games between 8 fake teams, built through the
    real feature builder so rows are internally consistent."""
    rng = random.Random(seed)
    teams = [f"Team {c}" for c in "ABCDEFGH"]
    # Elo-scale team strengths (+-200 points) and the same 100-point home
    # advantage the real model assumes, so Elo can converge to the truth here.
    strength = {t: rng.uniform(-200, 200) for t in teams}
    games: list[Game] = []
    idx = 0
    for season in seasons:
        start = datetime(season - 1, 10, 20, tzinfo=UTC)
        for i in range(n_per_season):
            home, away = rng.sample(teams, 2)
            p_home = 1 / (1 + 10 ** (-(strength[home] - strength[away] + 100) / 400))
            home_wins = rng.random() < p_home
            margin = rng.randint(1, 18)
            hs = 110
            games.append(
                Game(
                    game_id=f"s{season}g{idx}",
                    home_team=home,
                    away_team=away,
                    start_time=start + timedelta(hours=6 * i),
                    status=GameStatus.FINAL,
                    home_score=hs + (margin if home_wins else 0),
                    away_score=hs + (0 if home_wins else margin),
                    season_type=2,
                )
            )
            idx += 1
    return build_feature_rows(games)


class TestWalkForward:
    def test_trains_only_on_earlier_seasons(self) -> None:
        rows = synthetic_rows(600, [2024, 2025, 2026])
        preds = walk_forward_by_season(rows, first_test_season=2025)
        assert [p.season for p in preds] == [2025, 2026]
        for p in preds:
            assert all(r.season == p.season for r in p.rows)
            assert len(p.gbm) == len(p.rows) == len(p.labels)

    def test_refuses_tiny_train_set(self) -> None:
        rows = synthetic_rows(100, [2025, 2026])  # 100 < MIN_TRAIN_ROWS
        with pytest.raises(ValueError, match="refusing to fit"):
            walk_forward_by_season(rows, first_test_season=2026)

    def test_deterministic_under_seed(self) -> None:
        rows = synthetic_rows(600, [2025, 2026])
        a = walk_forward_by_season(rows, first_test_season=2026, seed=99)
        b = walk_forward_by_season(rows, first_test_season=2026, seed=99)
        assert np.array_equal(a[0].gbm, b[0].gbm)

    def test_beats_coin_flip_on_learnable_synthetic_data(self) -> None:
        """Sanity floor: on data with real signal the model must extract some."""
        from engine.backtest.metrics import brier_score

        rows = synthetic_rows(700, [2024, 2025, 2026])
        preds = walk_forward_by_season(rows, first_test_season=2026)[0]
        assert brier_score(preds.gbm, preds.labels) < 0.25
        assert brier_score(preds.elo, preds.labels) < 0.25


class TestGbmModel:
    def test_probabilities_in_range(self) -> None:
        rows = synthetic_rows(600, [2025, 2026])
        train = [r for r in rows if r.season == 2025]
        test = [r for r in rows if r.season == 2026]
        probs = GbmModel().fit(train).predict_proba(test)
        assert np.all((probs > 0) & (probs < 1))

    def test_calibration_improves_or_matches_ece(self) -> None:
        rows = synthetic_rows(900, [2025, 2026])
        train = [r for r in rows if r.season == 2025]
        test = [r for r in rows if r.season == 2026]
        labels = np.array([1.0 if r.label_home_won else 0.0 for r in test])
        raw = GbmModel(calibrate=False).fit(train).predict_proba(test)
        cal = GbmModel(calibrate=True).fit(train).predict_proba(test)
        assert expected_calibration_error(cal, labels) <= (
            expected_calibration_error(raw, labels) + 0.02  # small tolerance: different seeds/bins
        )

    def test_elo_probabilities_passthrough(self) -> None:
        rows = synthetic_rows(50, [2025])
        assert np.array_equal(
            elo_probabilities(rows), np.array([r.features["elo_home_prob"] for r in rows])
        )


class TestIsotonicCalibrator:
    def test_fixes_known_overconfidence(self) -> None:
        rng = np.random.default_rng(5)
        true_p = rng.uniform(0.1, 0.9, 4000)
        labels = (rng.random(4000) < true_p).astype(np.float64)
        overconfident = np.clip(true_p * 1.6 - 0.3, 0.01, 0.99)
        cal = IsotonicCalibrator().fit(overconfident, labels)
        fixed = cal.transform(overconfident)
        assert expected_calibration_error(fixed, labels) < expected_calibration_error(
            overconfident, labels
        )

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            IsotonicCalibrator().transform(np.array([0.5]))

    def test_output_clipped_from_certainty(self) -> None:
        probs = np.array([0.0, 0.5, 1.0])
        labels = np.array([0.0, 1.0, 1.0])
        out = IsotonicCalibrator().fit(probs, labels).transform(np.array([0.0, 1.0]))
        assert out.min() >= 0.01 and out.max() <= 0.99
