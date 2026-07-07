"""Live model evaluation vs. the in-game market, written to reports/.

The in-game join is the delicate part: each snapshot (ESPN wallclock) is
matched to the latest market candle that CLOSED at or before that wallclock,
for both team markets, then de-vigged. A candle that closes after the snapshot
is information from the future and is never used.

Uncertainty is bootstrapped BY GAME: snapshots within a game share one outcome
and are wildly correlated — resampling snapshots would fake ~500x the sample
size actually available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from engine.backtest.metrics import (
    FloatArray,
    brier_score,
    expected_calibration_error,
    log_loss,
)
from engine.data.models import Candle
from engine.data.store import Store
from engine.features.live import build_live_dataset
from engine.models.calibration import ReliabilitySeries, reliability_diagram
from engine.models.live_wp_model import LiveSeasonPredictions, walk_forward_live
from engine.models.market_baseline import devig_two_way

logger = structlog.get_logger(__name__)

MAX_INGAME_QUOTE_AGE_SECONDS = 15 * 60


@dataclass(frozen=True)
class InGameMarketSeries:
    """Model, market, and label aligned per snapshot for one test set."""

    game_ids: list[str]
    as_of_epoch: np.ndarray[tuple[int], np.dtype[np.int64]]
    model: FloatArray
    market: FloatArray
    labels: FloatArray

    def __len__(self) -> int:
        return len(self.game_ids)


def _mid_series(candles: list[Candle]) -> tuple[list[int], list[float]]:
    """(epoch seconds, raw mid prob) for candles with a two-sided book."""
    times: list[int] = []
    mids: list[float] = []
    for c in candles:
        if c.yes_bid_close is None or c.yes_ask_close is None:
            continue
        times.append(int(c.end_time.timestamp()))
        mids.append((c.yes_bid_close + c.yes_ask_close) / 200.0)
    return times, mids


def _latest_at_or_before(
    times: list[int], values: list[float], at: int
) -> tuple[int, float] | None:
    """Binary search for the freshest (time, value) with time <= at."""
    import bisect

    pos = bisect.bisect_right(times, at) - 1
    if pos < 0:
        return None
    return times[pos], values[pos]


def join_market_ingame(
    store: Store,
    predictions: list[LiveSeasonPredictions],
    *,
    max_quote_age_seconds: int = MAX_INGAME_QUOTE_AGE_SECONDS,
) -> InGameMarketSeries:
    """Align each test snapshot with the de-vigged market prob at that instant."""
    # market tickers per game, split home/away by yes_team
    games = {g.game_id: g for g in store.games()}
    tickers_by_game: dict[str, dict[str, str]] = {}
    for info in store.markets(with_game_only=True):
        assert info.game_id is not None
        game = games[info.game_id]
        side = "home" if info.yes_team == game.home_team else "away"
        tickers_by_game.setdefault(info.game_id, {})[side] = info.ticker

    series_cache: dict[str, tuple[list[int], list[float]]] = {}

    def mids_for(ticker: str) -> tuple[list[int], list[float]]:
        if ticker not in series_cache:
            series_cache[ticker] = _mid_series(store.candles(ticker))
        return series_cache[ticker]

    game_ids: list[str] = []
    as_of: list[int] = []
    model: list[float] = []
    market: list[float] = []
    labels: list[float] = []
    for sp in predictions:
        for i, game_id in enumerate(sp.data.game_ids):
            sides = tickers_by_game.get(game_id)
            if sides is None or "home" not in sides or "away" not in sides:
                continue
            at = int(sp.data.as_of_epoch[i])
            home_hit = _latest_at_or_before(*mids_for(sides["home"]), at)
            away_hit = _latest_at_or_before(*mids_for(sides["away"]), at)
            if home_hit is None or away_hit is None:
                continue
            if at - min(home_hit[0], away_hit[0]) > max_quote_age_seconds:
                continue
            home_prob, _ = devig_two_way(home_hit[1], away_hit[1])
            game_ids.append(game_id)
            as_of.append(at)
            model.append(float(sp.model[i]))
            market.append(home_prob)
            labels.append(float(sp.data.labels[i]))
    return InGameMarketSeries(
        game_ids=game_ids,
        as_of_epoch=np.array(as_of, dtype=np.int64),
        model=np.array(model),
        market=np.array(market),
        labels=np.array(labels),
    )


def game_grouped_bootstrap_brier_diff(
    series: InGameMarketSeries, *, n_resamples: int = 5_000, seed: int = 1337
) -> tuple[float, float, float]:
    """Brier(model) - Brier(market), resampling GAMES with replacement."""
    unique_games = sorted(set(series.game_ids))
    indices_by_game: dict[str, list[int]] = {g: [] for g in unique_games}
    for i, g in enumerate(series.game_ids):
        indices_by_game[g].append(i)
    point = brier_score(series.model, series.labels) - brier_score(series.market, series.labels)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    n_games = len(unique_games)
    for r in range(n_resamples):
        chosen = rng.integers(0, n_games, size=n_games)
        idx = np.concatenate([np.array(indices_by_game[unique_games[c]]) for c in chosen])
        diffs[r] = brier_score(series.model[idx], series.labels[idx]) - brier_score(
            series.market[idx], series.labels[idx]
        )
    lo, hi = np.quantile(diffs, (0.025, 0.975))
    return point, float(lo), float(hi)


def wp_chart(
    store: Store,
    series: InGameMarketSeries,
    game_id: str,
    path: Path,
) -> None:
    """Model-vs-market win-probability traces through one game."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    game = store.game(game_id)
    assert game is not None
    mask = np.array([g == game_id for g in series.game_ids])
    times = (series.as_of_epoch[mask] - series.as_of_epoch[mask].min()) / 60.0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, series.model[mask], label="model", linewidth=1.6)
    ax.plot(times, series.market[mask], label="market (de-vigged)", linewidth=1.6, alpha=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    outcome = "home won" if game.home_won else "home lost"
    ax.set_title(
        f"{game.away_team} @ {game.home_team} — "
        f"final {game.away_score}-{game.home_score} ({outcome})"
    )
    ax.set_xlabel("minutes since first snapshot (wallclock)")
    ax.set_ylabel("P(home win)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@dataclass(frozen=True)
class LiveEvaluation:
    test_seasons: list[int]
    n_test_snapshots: int
    full_metrics: list[tuple[str, int, float, float, float]]  # name, n, brier, ll, ece
    joined_n_snapshots: int
    joined_n_games: int
    model_brier: float
    market_brier: float
    diff_ci: tuple[float, float, float]


def run_live_evaluation(
    store: Store,
    *,
    first_test_season: int = 2026,
    reports_dir: Path = Path("reports"),
    seed: int = 1337,
) -> LiveEvaluation:
    data = build_live_dataset(store)
    logger.info("live dataset built", snapshots=len(data))
    predictions = walk_forward_live(data, first_test_season=first_test_season, seed=seed)

    all_model = np.concatenate([sp.model for sp in predictions])
    all_labels = np.concatenate([sp.data.labels for sp in predictions])
    all_prior = np.concatenate([sp.data.features[:, 3] for sp in predictions])

    def block(name: str, probs: FloatArray) -> tuple[str, int, float, float, float]:
        return (
            name,
            int(all_labels.size),
            brier_score(probs, all_labels),
            log_loss(probs, all_labels),
            expected_calibration_error(probs, all_labels),
        )

    full_metrics = [
        block("Live WP model (calibrated)", all_model),
        block("Pre-game prior only (Elo)", all_prior),
    ]

    joined = join_market_ingame(store, predictions)
    diff_ci = game_grouped_bootstrap_brier_diff(joined, seed=seed)

    reliability_diagram(
        [
            ReliabilitySeries("Live WP model", all_model, all_labels),
            ReliabilitySeries("Pre-game prior", all_prior, all_labels),
        ],
        reports_dir / "calibration_live.png",
        title=f"Live WP reliability — test season(s) {[sp.season for sp in predictions]}",
    )

    # example chart: the joined game with the most snapshots
    if len(joined):
        counts: dict[str, int] = {}
        for g in joined.game_ids:
            counts[g] = counts.get(g, 0) + 1
        example = max(counts, key=lambda g: counts[g])
        wp_chart(store, joined, example, reports_dir / "live_game_example.png")

    evaluation = LiveEvaluation(
        test_seasons=[sp.season for sp in predictions],
        n_test_snapshots=int(all_labels.size),
        full_metrics=full_metrics,
        joined_n_snapshots=len(joined),
        joined_n_games=len(set(joined.game_ids)),
        model_brier=brier_score(joined.model, joined.labels),
        market_brier=brier_score(joined.market, joined.labels),
        diff_ci=diff_ci,
    )
    _write_summary(evaluation, reports_dir / "live_summary.md")
    return evaluation


def _write_summary(ev: LiveEvaluation, path: Path) -> None:
    point, lo, hi = ev.diff_ci
    verdict = "model better" if hi < 0 else ("market better" if lo > 0 else "not distinguishable")
    lines = [
        "# Live in-game win-probability evaluation",
        "",
        f"Walk-forward test season(s): **{', '.join(map(str, ev.test_seasons))}**; "
        f"{ev.n_test_snapshots:,} test snapshots. Snapshots within a game are correlated — "
        "all uncertainty below is bootstrapped **by game**, not by snapshot.",
        "",
        "## Full test set",
        "",
        "| model | n | Brier | log loss | ECE |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {n:,} | {b:.4f} | {ll:.4f} | {e:.4f} |"
            for name, n, b, ll, e in ev.full_metrics
        ),
        "",
        "## In-game market comparison",
        "",
        f"Snapshots joinable to a fresh two-sided Kalshi quote (< 15 min old): "
        f"**{ev.joined_n_snapshots:,}** across **{ev.joined_n_games} games** "
        "(playoffs only — see data-retention note in the README).",
        "",
        f"- model Brier: **{ev.model_brier:.4f}**",
        f"- market Brier: **{ev.market_brier:.4f}**",
        f"- difference (model - market): {point:+.4f}, 95% game-bootstrap CI "
        f"[{lo:+.4f}, {hi:+.4f}] — **{verdict}**",
        "",
        "![reliability](calibration_live.png)",
        "",
        "![example game](live_game_example.png)",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("live summary written", path=str(path))
