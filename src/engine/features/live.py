"""In-game features for the live win-probability model.

A live feature vector is a pure function of (snapshot, pre-game prior) — no
running state, so leakage-safety reduces to two things this module enforces:

- the prior comes from the leakage-safe pre-game Elo stream (computed from
  strictly earlier games), and
- nothing about the game's future — including its final score — enters the
  features. The label is attached separately from the games table.

Time is normalized to *seconds remaining in regulation* (0 at the end of the
4th); overtime snapshots have 0 regulation seconds plus an ``ot_seconds``
feature for the current OT period. This keeps the dominant feature
(score-lead-per-remaining-time) on one consistent scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engine.backtest.metrics import FloatArray
from engine.data.models import Game, GameStatus
from engine.features.pregame import build_feature_rows, season_of

if TYPE_CHECKING:
    from engine.data.store import SnapshotColumns, Store

REGULATION_PERIODS = 4
PERIOD_SECONDS = 720.0
OT_SECONDS = 300.0
# score_diff / sqrt(remaining + tau): tau keeps the ratio finite at the buzzer
TIME_SMOOTHING_TAU = 30.0

LIVE_FEATURE_NAMES: tuple[str, ...] = (
    "score_diff",
    "seconds_remaining_regulation",
    "diff_per_sqrt_time",
    "prior_home_prob",
    "is_overtime",
    "ot_seconds_remaining",
)


def seconds_remaining_regulation(period: int, seconds_in_period: float) -> float:
    """Regulation seconds left; 0 for any overtime snapshot."""
    if period > REGULATION_PERIODS:
        return 0.0
    return (REGULATION_PERIODS - period) * PERIOD_SECONDS + seconds_in_period


@dataclass(frozen=True)
class LiveDataset:
    """Column-oriented live training/eval set. Row i belongs to game
    ``game_ids[i]``; grouping by game is mandatory for any resampling."""

    game_ids: list[str]
    as_of_epoch: np.ndarray[tuple[int], np.dtype[np.int64]]
    features: FloatArray  # shape (n, len(LIVE_FEATURE_NAMES))
    labels: FloatArray
    seasons: np.ndarray[tuple[int], np.dtype[np.int64]]

    def __len__(self) -> int:
        return len(self.game_ids)

    def mask(self, keep: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> LiveDataset:
        return LiveDataset(
            game_ids=[g for g, k in zip(self.game_ids, keep, strict=True) if k],
            as_of_epoch=self.as_of_epoch[keep],
            features=self.features[keep],
            labels=self.labels[keep],
            seasons=self.seasons[keep],
        )


def compute_live_features(
    score_diff: FloatArray,
    seconds_regulation: FloatArray,
    prior_home_prob: FloatArray,
    period: np.ndarray[tuple[int], np.dtype[np.int64]],
    seconds_in_period: FloatArray,
) -> FloatArray:
    """Vectorized feature matrix; single source of truth for training AND the
    live paper-trading path so the two can never drift apart."""
    is_ot = (period > REGULATION_PERIODS).astype(np.float64)
    ot_seconds = np.where(period > REGULATION_PERIODS, seconds_in_period, 0.0)
    diff_per_sqrt_time = score_diff / np.sqrt(seconds_regulation + ot_seconds + TIME_SMOOTHING_TAU)
    return np.column_stack(
        [
            score_diff,
            seconds_regulation,
            diff_per_sqrt_time,
            prior_home_prob,
            is_ot,
            ot_seconds,
        ]
    )


def build_live_dataset(store: Store) -> LiveDataset:
    """Join every stored snapshot to its game's label and pre-game Elo prior.

    Games without a prior (first-ever appearance) or without a decidable final
    result are excluded — exclusions are structural, never label-dependent.
    """
    games: dict[str, Game] = {g.game_id: g for g in store.games(statuses=[GameStatus.FINAL])}
    priors: dict[str, float] = {
        row.game_id: row.elo_home_prob for row in build_feature_rows(list(games.values()))
    }

    cols: SnapshotColumns = store.snapshot_columns()
    keep_idx: list[int] = []
    labels: list[float] = []
    prior_col: list[float] = []
    seasons: list[int] = []
    for i, game_id in enumerate(cols.game_id):
        prior = priors.get(game_id)
        game = games.get(game_id)
        if prior is None or game is None:
            continue
        keep_idx.append(i)
        labels.append(1.0 if game.home_won else 0.0)
        prior_col.append(prior)
        seasons.append(season_of(game.start_time))

    idx = np.array(keep_idx, dtype=np.int64)
    period = cols.period[idx]
    seconds_in_period = cols.seconds_remaining_in_period[idx]
    score_diff = (cols.home_score[idx] - cols.away_score[idx]).astype(np.float64)
    seconds_reg = np.array(
        [
            seconds_remaining_regulation(int(p), float(s))
            for p, s in zip(period, seconds_in_period, strict=True)
        ]
    )
    features = compute_live_features(
        score_diff=score_diff,
        seconds_regulation=seconds_reg,
        prior_home_prob=np.array(prior_col),
        period=period,
        seconds_in_period=seconds_in_period,
    )
    return LiveDataset(
        game_ids=[cols.game_id[i] for i in keep_idx],
        as_of_epoch=cols.as_of_epoch[idx],
        features=features,
        labels=np.array(labels),
        seasons=np.array(seasons, dtype=np.int64),
    )
