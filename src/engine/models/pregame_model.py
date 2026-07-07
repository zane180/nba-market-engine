"""Pre-game win-probability models.

Two tiers, always reported side by side:

- ``elo_probabilities`` — the transparent baseline; no fitting at all.
- ``GbmModel`` — gradient boosting over the pre-game features, with an
  isotonic calibrator fitted on cross-validated (out-of-sample) training
  predictions and frozen before it ever sees test data.

Walk-forward discipline lives in ``walk_forward_by_season``: predictions for
season S come from a model fitted on strictly earlier seasons. The model is
not refit within a season (features still update game by game via Elo/form);
that staleness is a documented modeling choice, not an accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from engine.backtest.metrics import FloatArray
from engine.features.pregame import FEATURE_NAMES, FeatureRow
from engine.models.calibration import IsotonicCalibrator

MIN_TRAIN_ROWS = 500


def feature_matrix(rows: list[FeatureRow]) -> FloatArray:
    return np.array([[r.features[name] for name in FEATURE_NAMES] for r in rows])


def labels_vector(rows: list[FeatureRow]) -> FloatArray:
    return np.array([1.0 if r.label_home_won else 0.0 for r in rows])


def elo_probabilities(rows: list[FeatureRow]) -> FloatArray:
    return np.array([r.elo_home_prob for r in rows])


class GbmModel:
    def __init__(self, *, seed: int = 1337, calibrate: bool = True) -> None:
        self._seed = seed
        self._calibrate = calibrate
        self._clf = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
        )
        self._calibrator: IsotonicCalibrator | None = None

    def fit(self, rows: list[FeatureRow]) -> GbmModel:
        if len(rows) < MIN_TRAIN_ROWS:
            raise ValueError(f"refusing to fit on {len(rows)} rows (< {MIN_TRAIN_ROWS})")
        x, y = feature_matrix(rows), labels_vector(rows)

        if self._calibrate:
            # Out-of-fold predictions on TRAIN ONLY feed the calibrator; the
            # test season never touches it.
            oof = np.zeros_like(y)
            for train_idx, val_idx in KFold(5, shuffle=True, random_state=self._seed).split(x):
                fold = HistGradientBoostingClassifier(**self._clf.get_params())
                fold.fit(x[train_idx], y[train_idx])
                oof[val_idx] = fold.predict_proba(x[val_idx])[:, 1]
            self._calibrator = IsotonicCalibrator().fit(oof, y)

        self._clf.fit(x, y)
        return self

    def predict_proba(self, rows: list[FeatureRow]) -> FloatArray:
        raw = np.asarray(self._clf.predict_proba(feature_matrix(rows))[:, 1], dtype=np.float64)
        if self._calibrator is not None:
            return self._calibrator.transform(raw)
        return raw


@dataclass(frozen=True, slots=True)
class SeasonPredictions:
    season: int
    rows: list[FeatureRow]
    elo: FloatArray
    gbm: FloatArray
    labels: FloatArray


def walk_forward_by_season(
    rows: list[FeatureRow], *, first_test_season: int, seed: int = 1337
) -> list[SeasonPredictions]:
    """For each season >= first_test_season, predict it with a model trained on
    all strictly earlier rows. Raises rather than quietly training on less
    data than ``MIN_TRAIN_ROWS`` — a tiny train set would produce numbers that
    look like results."""
    out: list[SeasonPredictions] = []
    seasons = sorted({r.season for r in rows if r.season >= first_test_season})
    for season in seasons:
        train = [r for r in rows if r.season < season]
        test = [r for r in rows if r.season == season]
        model = GbmModel(seed=seed).fit(train)
        out.append(
            SeasonPredictions(
                season=season,
                rows=test,
                elo=elo_probabilities(test),
                gbm=model.predict_proba(test),
                labels=labels_vector(test),
            )
        )
    return out
