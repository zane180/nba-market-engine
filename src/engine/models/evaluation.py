"""Pre-game evaluation: models vs. the de-vigged market, written to reports/.

Every number the README cites about the pre-game model comes out of
``run_pregame_evaluation`` — regenerable end-to-end from the ingested dataset
with fixed seeds.
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
    paired_bootstrap_diff,
)
from engine.data.store import Store
from engine.features.pregame import FeatureRow, build_feature_rows
from engine.models.calibration import ReliabilitySeries, reliability_diagram
from engine.models.market_baseline import market_home_probability
from engine.models.pregame_model import SeasonPredictions, walk_forward_by_season

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MetricBlock:
    name: str
    n: int
    brier: float
    logloss: float
    ece: float

    @classmethod
    def compute(cls, name: str, probs: FloatArray, labels: FloatArray) -> MetricBlock:
        return cls(
            name=name,
            n=int(labels.size),
            brier=brier_score(probs, labels),
            logloss=log_loss(probs, labels),
            ece=expected_calibration_error(probs, labels),
        )

    def row(self) -> str:
        return (
            f"| {self.name} | {self.n} | {self.brier:.4f} | {self.logloss:.4f} | {self.ece:.4f} |"
        )


@dataclass(frozen=True, slots=True)
class PregameEvaluation:
    test_seasons: list[int]
    full_test: list[MetricBlock]
    market_subset: list[MetricBlock]
    market_subset_n: int
    gbm_vs_market_brier: tuple[float, float, float]  # point, ci_lo, ci_hi
    elo_vs_market_brier: tuple[float, float, float]
    mean_overround: float


@dataclass(frozen=True, slots=True)
class MarketJoined:
    rows: list[FeatureRow]
    market: FloatArray
    elo: FloatArray
    gbm: FloatArray
    labels: FloatArray
    overrounds: FloatArray


def _market_joined(store: Store, predictions: list[SeasonPredictions]) -> MarketJoined:
    """Rows of the test seasons that have a usable pre-tipoff market quote.

    The market prob is taken strictly at-or-before tipoff — the same instant
    the model's features freeze — so both forecasters answer the same question
    with the same information cutoff.
    """
    rows: list[FeatureRow] = []
    market: list[float] = []
    elo: list[float] = []
    gbm: list[float] = []
    labels: list[float] = []
    overrounds: list[float] = []
    for sp in predictions:
        for i, row in enumerate(sp.rows):
            quote = market_home_probability(
                store, game_id=row.game_id, home_team=row.home_team, at=row.start_time
            )
            if quote is None:
                continue
            rows.append(row)
            market.append(quote.home_prob)
            overrounds.append(quote.overround)
            elo.append(float(sp.elo[i]))
            gbm.append(float(sp.gbm[i]))
            labels.append(float(sp.labels[i]))
    if not rows:
        raise RuntimeError("no test games have market data — run 'engine ingest all' first")
    return MarketJoined(
        rows=rows,
        market=np.array(market),
        elo=np.array(elo),
        gbm=np.array(gbm),
        labels=np.array(labels),
        overrounds=np.array(overrounds),
    )


def run_pregame_evaluation(
    store: Store,
    *,
    first_test_season: int = 2026,
    reports_dir: Path = Path("reports"),
    seed: int = 1337,
) -> PregameEvaluation:
    games = store.games()
    rows = build_feature_rows(games)
    logger.info("feature rows built", total=len(rows))

    predictions = walk_forward_by_season(rows, first_test_season=first_test_season, seed=seed)
    test_seasons = [sp.season for sp in predictions]

    all_elo = np.concatenate([sp.elo for sp in predictions])
    all_gbm = np.concatenate([sp.gbm for sp in predictions])
    all_labels = np.concatenate([sp.labels for sp in predictions])
    home_rate = float(all_labels.mean())
    constant = np.full_like(all_labels, home_rate)

    full_test = [
        MetricBlock.compute("Elo", all_elo, all_labels),
        MetricBlock.compute("GBM (isotonic-calibrated)", all_gbm, all_labels),
        MetricBlock.compute(f"constant p={home_rate:.3f}", constant, all_labels),
    ]

    joined = _market_joined(store, predictions)
    market_subset = [
        MetricBlock.compute("Market (de-vigged mid)", joined.market, joined.labels),
        MetricBlock.compute("GBM (isotonic-calibrated)", joined.gbm, joined.labels),
        MetricBlock.compute("Elo", joined.elo, joined.labels),
    ]
    gbm_vs_market = paired_bootstrap_diff(joined.gbm, joined.market, joined.labels, seed=seed)
    elo_vs_market = paired_bootstrap_diff(joined.elo, joined.market, joined.labels, seed=seed)

    reliability_diagram(
        [
            ReliabilitySeries("Elo", all_elo, all_labels),
            ReliabilitySeries("GBM (calibrated)", all_gbm, all_labels),
        ],
        reports_dir / "calibration_pregame.png",
        title=f"Pre-game reliability — test season(s) {', '.join(map(str, test_seasons))}",
    )

    evaluation = PregameEvaluation(
        test_seasons=test_seasons,
        full_test=full_test,
        market_subset=market_subset,
        market_subset_n=int(joined.labels.size),
        gbm_vs_market_brier=gbm_vs_market,
        elo_vs_market_brier=elo_vs_market,
        mean_overround=float(joined.overrounds.mean()),
    )
    _write_summary(evaluation, reports_dir / "pregame_summary.md")
    return evaluation


def _fmt_diff(diff: tuple[float, float, float]) -> str:
    point, lo, hi = diff
    verdict = "model better" if hi < 0 else ("market better" if lo > 0 else "not distinguishable")
    return f"{point:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]) — {verdict}"


def _write_summary(ev: PregameEvaluation, path: Path) -> None:
    header = "| model | n | Brier | log loss | ECE |\n|---|---|---|---|---|"
    lines = [
        "# Pre-game model evaluation",
        "",
        f"Walk-forward test season(s): **{', '.join(map(str, ev.test_seasons))}** "
        "(trained on strictly earlier seasons; no in-season refits).",
        "",
        "## Full test set (all final games)",
        "",
        header,
        *(b.row() for b in ev.full_test),
        "",
        "## Market-covered subset",
        "",
        f"Games with a usable de-vigged Kalshi quote at tipoff: **{ev.market_subset_n}** "
        f"(mean overround {ev.mean_overround:.3f}). Kalshi's API only retains recent market "
        "history, so this subset is small and currently playoffs-only — read the intervals, "
        "not the point estimates.",
        "",
        header,
        *(b.row() for b in ev.market_subset),
        "",
        "### Paired bootstrap, Brier difference vs. market (negative = beats market)",
        "",
        f"- GBM - market: {_fmt_diff(ev.gbm_vs_market_brier)}",
        f"- Elo - market: {_fmt_diff(ev.elo_vs_market_brier)}",
        "",
        "![reliability diagram](calibration_pregame.png)",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("summary written", path=str(path))
