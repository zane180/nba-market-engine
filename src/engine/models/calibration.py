"""Probability calibration and reliability diagrams.

Isotonic regression is the workhorse: monotone, non-parametric, and safe for
the sample sizes here (thousands of games). The calibrator must only ever be
fitted on predictions the underlying model produced out-of-sample — fitting it
on in-sample predictions launders overfitting into fake confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from engine.backtest.metrics import FloatArray


class IsotonicCalibrator:
    """Thin typed wrapper around sklearn's isotonic regression, clipped away
    from hard 0/1 so log loss stays finite."""

    def __init__(self, *, floor: float = 0.01) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._floor = floor
        self._fitted = False

    def fit(self, probs: FloatArray, labels: FloatArray) -> IsotonicCalibrator:
        if probs.shape != labels.shape or probs.ndim != 1:
            raise ValueError("probs and labels must be 1-D and same length")
        self._iso.fit(probs, labels)
        self._fitted = True
        return self

    def transform(self, probs: FloatArray) -> FloatArray:
        if not self._fitted:
            raise RuntimeError("calibrator used before fit")
        out = np.asarray(self._iso.predict(probs), dtype=np.float64)
        return np.clip(out, self._floor, 1.0 - self._floor)


@dataclass(frozen=True, slots=True)
class ReliabilitySeries:
    name: str
    probs: FloatArray
    labels: FloatArray


def reliability_diagram(
    series: list[ReliabilitySeries],
    path: Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> None:
    """Calibration curves (top) + forecast histogram (bottom) -> PNG.

    Bins with fewer than 5 games are dropped from the curve rather than drawn
    as noise.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; never require a display
    import matplotlib.pyplot as plt

    fig, (ax_curve, ax_hist) = plt.subplots(2, 1, figsize=(7, 8), height_ratios=[3, 1], sharex=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ax_curve.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    for s in series:
        bin_index = np.clip(np.digitize(s.probs, edges[1:-1]), 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            mask = bin_index == b
            if mask.sum() < 5:
                continue
            xs.append(float(s.probs[mask].mean()))
            ys.append(float(s.labels[mask].mean()))
        ax_curve.plot(xs, ys, marker="o", linewidth=1.5, label=f"{s.name} (n={s.probs.size})")
        ax_hist.hist(s.probs, bins=list(edges), histtype="step", linewidth=1.5, label=s.name)
    ax_curve.set_ylabel("empirical home-win rate")
    ax_curve.set_title(title)
    ax_curve.legend(loc="upper left", fontsize=9)
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_hist.set_xlabel("forecast home-win probability")
    ax_hist.set_ylabel("count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
