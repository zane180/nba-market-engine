"""Scoring-rule correctness against hand-computed values and known properties."""

from __future__ import annotations

import numpy as np
import pytest

from engine.backtest.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    paired_bootstrap_diff,
)


class TestBrier:
    def test_hand_computed(self) -> None:
        probs = np.array([0.8, 0.3, 0.5])
        labels = np.array([1.0, 0.0, 1.0])
        # (0.04 + 0.09 + 0.25) / 3
        assert brier_score(probs, labels) == pytest.approx(0.38 / 3)

    def test_perfect_and_worst(self) -> None:
        labels = np.array([1.0, 0.0])
        assert brier_score(np.array([1.0, 0.0]), labels) == 0.0
        assert brier_score(np.array([0.0, 1.0]), labels) == 1.0

    def test_constant_half_scores_quarter(self) -> None:
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(np.full(4, 0.5), labels) == pytest.approx(0.25)


class TestLogLoss:
    def test_hand_computed(self) -> None:
        probs = np.array([0.8, 0.3])
        labels = np.array([1.0, 0.0])
        expected = -(np.log(0.8) + np.log(0.7)) / 2
        assert log_loss(probs, labels) == pytest.approx(float(expected))

    def test_certain_and_wrong_is_finite(self) -> None:
        assert np.isfinite(log_loss(np.array([1.0]), np.array([0.0])))

    def test_proper_scoring(self) -> None:
        """The truth minimizes expected log loss — the property that makes it a
        proper scoring rule (and any bugged variant fail here)."""
        rng = np.random.default_rng(0)
        true_p = 0.7
        labels = (rng.random(20_000) < true_p).astype(np.float64)
        truth = log_loss(np.full_like(labels, true_p), labels)
        for other in (0.5, 0.6, 0.8, 0.9):
            assert truth < log_loss(np.full_like(labels, other), labels)


class TestEce:
    def test_perfectly_calibrated_bins(self) -> None:
        # 10 forecasts of 0.2 with 2 hits; 10 of 0.8 with 8 hits
        probs = np.array([0.2] * 10 + [0.8] * 10)
        labels = np.array([1.0] * 2 + [0.0] * 8 + [1.0] * 8 + [0.0] * 2)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_known_miscalibration(self) -> None:
        # all forecasts 0.9, hit rate 0.5 -> ECE = 0.4
        probs = np.full(10, 0.9)
        labels = np.array([1.0, 0.0] * 5)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.4)

    def test_edge_probs_binned(self) -> None:
        # p=0.0 and p=1.0 must land in bins, not crash or vanish
        assert expected_calibration_error(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        ) == pytest.approx(0.0)


class TestValidation:
    @pytest.mark.parametrize(
        ("probs", "labels"),
        [
            (np.array([0.5, 0.5]), np.array([1.0])),  # length mismatch
            (np.array([1.5]), np.array([1.0])),  # prob out of range
            (np.array([0.5]), np.array([2.0])),  # non-binary label
            (np.array([]), np.array([])),  # empty
        ],
    )
    def test_bad_inputs_raise(self, probs: object, labels: object) -> None:
        with pytest.raises(ValueError):
            brier_score(probs, labels)  # type: ignore[arg-type]


class TestPairedBootstrap:
    def test_clearly_better_model_gets_negative_interval(self) -> None:
        rng = np.random.default_rng(1)
        true_p = rng.uniform(0.2, 0.8, size=400)
        labels = (rng.random(400) < true_p).astype(np.float64)
        good = np.clip(true_p, 0.01, 0.99)
        bad = np.clip(true_p + rng.normal(0, 0.25, size=400), 0.01, 0.99)
        point, _lo, hi = paired_bootstrap_diff(good, bad, labels, n_resamples=2000)
        assert point < 0 and hi < 0  # good model better, significantly

    def test_identical_models_straddle_zero(self) -> None:
        rng = np.random.default_rng(2)
        probs = rng.uniform(0.3, 0.7, 100)
        labels = (rng.random(100) < probs).astype(np.float64)
        point, lo, hi = paired_bootstrap_diff(probs, probs.copy(), labels, n_resamples=500)
        assert point == pytest.approx(0.0)
        assert lo <= 0.0 <= hi

    def test_deterministic_under_seed(self) -> None:
        rng = np.random.default_rng(3)
        a, b = rng.uniform(0.2, 0.8, 50), rng.uniform(0.2, 0.8, 50)
        labels = (rng.random(50) < 0.5).astype(np.float64)
        r1 = paired_bootstrap_diff(a, b, labels, n_resamples=200, seed=42)
        r2 = paired_bootstrap_diff(a, b, labels, n_resamples=200, seed=42)
        assert r1 == r2
