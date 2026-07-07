"""Live in-game win-probability model.

Gradient boosting over the live features, isotonic-calibrated on *game-grouped*
out-of-fold training predictions — snapshots within a game share an outcome, so
folds that split a game across train/val would leak the label through its
sibling snapshots. GroupKFold makes that impossible.

Walk-forward mirrors the pre-game model: seasons before the test season train;
the test season is only ever predicted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from engine.backtest.metrics import FloatArray
from engine.features.live import LiveDataset
from engine.models.calibration import IsotonicCalibrator

MIN_TRAIN_SNAPSHOTS = 10_000


class LiveWpModel:
    def __init__(self, *, seed: int = 1337, calibrate: bool = True) -> None:
        self._seed = seed
        self._calibrate = calibrate
        self._clf = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
        )
        self._calibrator: IsotonicCalibrator | None = None

    def fit(self, data: LiveDataset) -> LiveWpModel:
        if len(data) < MIN_TRAIN_SNAPSHOTS:
            raise ValueError(f"refusing to fit on {len(data)} snapshots (< {MIN_TRAIN_SNAPSHOTS})")
        x, y = data.features, data.labels
        if self._calibrate:
            # deterministic group ids (str hash() is salted per process)
            _, groups = np.unique(np.array(data.game_ids), return_inverse=True)
            oof = np.zeros_like(y)
            for train_idx, val_idx in GroupKFold(n_splits=5).split(x, y, groups=groups):
                fold = HistGradientBoostingClassifier(**self._clf.get_params())
                fold.fit(x[train_idx], y[train_idx])
                oof[val_idx] = fold.predict_proba(x[val_idx])[:, 1]
            self._calibrator = IsotonicCalibrator().fit(oof, y)
        self._clf.fit(x, y)
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        raw = np.asarray(self._clf.predict_proba(features)[:, 1], dtype=np.float64)
        if self._calibrator is not None:
            return self._calibrator.transform(raw)
        return raw


@dataclass(frozen=True)
class LiveSeasonPredictions:
    season: int
    data: LiveDataset
    model: FloatArray


def walk_forward_live(
    data: LiveDataset, *, first_test_season: int, seed: int = 1337
) -> list[LiveSeasonPredictions]:
    out: list[LiveSeasonPredictions] = []
    test_seasons = sorted({int(s) for s in np.unique(data.seasons) if s >= first_test_season})
    for season in test_seasons:
        train = data.mask(data.seasons < season)
        test = data.mask(data.seasons == season)
        model = LiveWpModel(seed=seed).fit(train)
        out.append(
            LiveSeasonPredictions(
                season=season, data=test, model=model.predict_proba(test.features)
            )
        )
    return out
