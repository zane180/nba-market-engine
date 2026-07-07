"""Proper scoring rules and comparison statistics.

These are the numbers every claim in the README traces back to, so they're
implemented here once, from scratch (no metric imported from a library that
might silently change clipping or binning behavior between versions).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_EPS = 1e-15


def _validate(probs: FloatArray, labels: FloatArray) -> tuple[FloatArray, FloatArray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape or probs.ndim != 1:
        raise ValueError(f"shape mismatch: probs {probs.shape}, labels {labels.shape}")
    if probs.size == 0:
        raise ValueError("empty inputs")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities outside [0, 1]")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("labels must be 0 or 1")
    return probs, labels


def brier_score(probs: FloatArray, labels: FloatArray) -> float:
    """Mean squared error of probabilities. 0 is perfect; 0.25 is the score of
    a constant 0.5 forecast."""
    probs, labels = _validate(probs, labels)
    return float(np.mean((probs - labels) ** 2))


def log_loss(probs: FloatArray, labels: FloatArray) -> float:
    """Mean negative log-likelihood in nats, clipped at 1e-15 so a (wrongly)
    certain forecast is heavily punished but finite."""
    probs, labels = _validate(probs, labels)
    clipped = np.clip(probs, _EPS, 1 - _EPS)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def expected_calibration_error(probs: FloatArray, labels: FloatArray, *, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum over bins of |mean prob - hit rate|
    weighted by bin occupancy."""
    probs, labels = _validate(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive last bin so p=1.0 lands in a bin
    bin_index = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


class _Metric(Protocol):
    def __call__(self, probs: FloatArray, labels: FloatArray) -> float: ...


def paired_bootstrap_diff(
    probs_a: FloatArray,
    probs_b: FloatArray,
    labels: FloatArray,
    *,
    metric: _Metric = brier_score,
    n_resamples: int = 10_000,
    seed: int = 1337,
) -> tuple[float, float, float]:
    """(point diff, 2.5%, 97.5%) for ``metric(a) - metric(b)`` under paired
    resampling of games. Negative means A scores better (lower).

    This is what keeps small-sample comparisons honest: with ~42 games, the
    interval matters far more than the point estimate.
    """
    probs_a, labels = _validate(probs_a, labels)
    probs_b, _ = _validate(probs_b, labels)
    point = metric(probs_a, labels) - metric(probs_b, labels)
    rng = np.random.default_rng(seed)
    n = labels.size
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric(probs_a[idx], labels[idx]) - metric(probs_b[idx], labels[idx])
    lo, hi = np.quantile(diffs, (0.025, 0.975))
    return point, float(lo), float(hi)
