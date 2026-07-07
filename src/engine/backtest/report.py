"""Build opportunities from the stored dataset, run the backtest, and write
reports/backtest_summary.md + equity_curve.png.

Two strategies are simulated with identical machinery:

- **pregame**: one decision per market at the last standing quote before tipoff,
  using the walk-forward pre-game GBM probability.
- **live**: one decision per market-minute using the walk-forward live WP model;
  the decision at candle-close T uses the model state as of T and executes at
  the first candle ending >= T + execution_delay — a deliberate handicap that
  absorbs ESPN wallclock lag and our own reaction time.

Baselines: "never trade" (implicit: ROI 0) and "follow the market" (buy the
market favorite pre-game with the same sizing cap, believing the market's own
de-vigged probability... which sizes to zero under Kelly, so it uses a fixed
1% stake — it exists to show fee drag on a no-edge strategy).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import structlog

from engine.backtest.engine import (
    BacktestResult,
    Opportunity,
    StrategyParams,
    simulate,
)
from engine.backtest.fees import taker_fee_dollars
from engine.data.models import MarketResult, Side
from engine.data.store import Store
from engine.features.pregame import build_feature_rows
from engine.models.live_wp_model import walk_forward_live
from engine.models.pregame_model import walk_forward_by_season

logger = structlog.get_logger(__name__)

PREGAME_STALENESS = timedelta(hours=6)
DEFAULT_EXECUTION_DELAY = timedelta(seconds=60)


@dataclass(frozen=True)
class MarketContext:
    ticker: str
    game_id: str
    yes_is_home: bool
    yes_settled: bool
    tipoff: datetime
    settle_time: datetime
    candle_times: list[int]  # epoch, two-sided candles only
    candle_bids: list[int]
    candle_asks: list[int]


def _market_contexts(store: Store) -> dict[str, MarketContext]:
    games = {g.game_id: g for g in store.games()}
    contexts: dict[str, MarketContext] = {}
    for info in store.markets(with_game_only=True):
        assert info.game_id is not None
        if info.result is None:
            continue
        game = games[info.game_id]
        times: list[int] = []
        bids: list[int] = []
        asks: list[int] = []
        for c in store.candles(info.ticker):
            if c.yes_bid_close is None or c.yes_ask_close is None:
                continue
            times.append(int(c.end_time.timestamp()))
            bids.append(c.yes_bid_close)
            asks.append(c.yes_ask_close)
        contexts[info.ticker] = MarketContext(
            ticker=info.ticker,
            game_id=info.game_id,
            yes_is_home=info.yes_team == game.home_team,
            yes_settled=info.result is MarketResult.YES,
            tipoff=game.start_time,
            settle_time=info.close_time,
            candle_times=times,
            candle_bids=bids,
            candle_asks=asks,
        )
    return contexts


def _quote_at_or_before(ctx: MarketContext, at: int) -> tuple[int, int, int] | None:
    pos = bisect.bisect_right(ctx.candle_times, at) - 1
    if pos < 0:
        return None
    return ctx.candle_times[pos], ctx.candle_bids[pos], ctx.candle_asks[pos]


def _quote_at_or_after(ctx: MarketContext, at: int) -> tuple[int, int, int] | None:
    pos = bisect.bisect_left(ctx.candle_times, at)
    if pos >= len(ctx.candle_times):
        return None
    return ctx.candle_times[pos], ctx.candle_bids[pos], ctx.candle_asks[pos]


def build_pregame_opportunities(
    store: Store, *, first_test_season: int, seed: int
) -> tuple[list[Opportunity], dict[str, bool]]:
    rows = build_feature_rows(store.games())
    predictions = walk_forward_by_season(rows, first_test_season=first_test_season, seed=seed)
    home_prob: dict[str, float] = {}
    for sp in predictions:
        for i, row in enumerate(sp.rows):
            home_prob[row.game_id] = float(sp.gbm[i])

    opportunities: list[Opportunity] = []
    outcomes: dict[str, bool] = {}
    for ctx in _market_contexts(store).values():
        prob_home = home_prob.get(ctx.game_id)
        if prob_home is None:
            continue
        tip_epoch = int(ctx.tipoff.timestamp())
        quote = _quote_at_or_before(ctx, tip_epoch)
        if quote is None or tip_epoch - quote[0] > PREGAME_STALENESS.total_seconds():
            continue
        _, bid, ask = quote
        opportunities.append(
            Opportunity(
                ticker=ctx.ticker,
                decision_time=ctx.tipoff,
                model_prob=prob_home if ctx.yes_is_home else 1.0 - prob_home,
                yes_bid=bid,
                yes_ask=ask,
                settle_time=ctx.settle_time,
                group=ctx.game_id,
            )
        )
        outcomes[ctx.ticker] = ctx.yes_settled
    return opportunities, outcomes


def build_live_opportunities(
    store: Store,
    *,
    first_test_season: int,
    seed: int,
    execution_delay: timedelta = DEFAULT_EXECUTION_DELAY,
    max_model_staleness: timedelta = timedelta(minutes=5),
) -> tuple[list[Opportunity], dict[str, bool]]:
    from engine.features.live import build_live_dataset

    data = build_live_dataset(store)
    predictions = walk_forward_live(data, first_test_season=first_test_season, seed=seed)

    # model prob series per game (sorted by construction: snapshot_columns
    # orders by game_id, as_of)
    prob_series: dict[str, tuple[list[int], list[float]]] = {}
    for sp in predictions:
        for i, game_id in enumerate(sp.data.game_ids):
            times, probs = prob_series.setdefault(game_id, ([], []))
            times.append(int(sp.data.as_of_epoch[i]))
            probs.append(float(sp.model[i]))

    opportunities: list[Opportunity] = []
    outcomes: dict[str, bool] = {}
    delay_s = int(execution_delay.total_seconds())
    staleness_s = int(max_model_staleness.total_seconds())
    for ctx in _market_contexts(store).values():
        series = prob_series.get(ctx.game_id)
        if series is None:
            continue
        snap_times, snap_probs = series
        tip_epoch = int(ctx.tipoff.timestamp())
        outcomes[ctx.ticker] = ctx.yes_settled
        for snap_pos, snap_time in enumerate(snap_times):
            if snap_time < tip_epoch:
                continue
            exec_quote = _quote_at_or_after(ctx, snap_time + delay_s)
            if exec_quote is None:
                continue
            exec_time, bid, ask = exec_quote
            if exec_time - snap_time > staleness_s + delay_s:
                continue  # model view too old by execution time
            prob_home = snap_probs[snap_pos]
            opportunities.append(
                Opportunity(
                    ticker=ctx.ticker,
                    decision_time=datetime.fromtimestamp(exec_time, tz=UTC),
                    model_prob=prob_home if ctx.yes_is_home else 1.0 - prob_home,
                    yes_bid=bid,
                    yes_ask=ask,
                    settle_time=ctx.settle_time,
                    group=ctx.game_id,
                )
            )
    return opportunities, outcomes


def follow_market_baseline(
    opportunities: list[Opportunity], outcomes: dict[str, bool], *, initial_bankroll: float
) -> BacktestResult:
    """Buy the market favorite at every pre-game opportunity with a flat 1%
    stake. No model, no edge — isolates spread + fee drag."""
    bankroll = initial_bankroll
    from engine.backtest.engine import ExecutedTrade

    trades: list[ExecutedTrade] = []
    equity = [initial_bankroll]
    times = [opportunities[0].decision_time if opportunities else datetime.now(UTC)]
    for opp in sorted(opportunities, key=lambda o: (o.decision_time, o.ticker)):
        if opp.yes_bid is None or opp.yes_ask is None:
            continue
        mid = (opp.yes_bid + opp.yes_ask) / 2
        side = Side.YES if mid >= 50 else Side.NO
        price = opp.buy_price(side)
        if price is None:
            continue
        stake = 0.01 * bankroll
        contracts = stake / (price / 100.0)
        fee = taker_fee_dollars(price, contracts)
        cost = stake + fee
        if cost > bankroll:
            continue
        yes_settled = outcomes[opp.ticker]
        won = yes_settled if side is Side.YES else not yes_settled
        payout = contracts if won else 0.0
        bankroll += payout - cost
        trades.append(
            ExecutedTrade(
                ticker=opp.ticker,
                time=opp.decision_time,
                side=side,
                contracts=contracts,
                price_cents=price,
                fee=fee,
                cost=cost,
                model_prob=mid / 100.0,
                won=won,
                payout=payout,
            )
        )
        times.append(opp.decision_time)
        equity.append(bankroll)
    return BacktestResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        trades=trades,
        equity_times=times,
        equity=np.array(equity),
    )


def pnl_bootstrap_ci(
    result: BacktestResult,
    ticker_to_game: dict[str, str],
    *,
    n_resamples: int = 10_000,
    seed: int = 1337,
) -> tuple[float, float]:
    """95% CI on total PnL from resampling GAMES (trades within a game are one
    draw). This is the number that keeps a lucky small sample honest."""
    by_game: dict[str, float] = {}
    for t in result.trades:
        game = ticker_to_game.get(t.ticker, t.ticker)
        by_game[game] = by_game.get(game, 0.0) + t.pnl
    pnls = np.array(list(by_game.values()))
    if pnls.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    totals = np.array(
        [pnls[rng.integers(0, pnls.size, size=pnls.size)].sum() for _ in range(n_resamples)]
    )
    lo, hi = np.quantile(totals, (0.025, 0.975))
    return float(lo), float(hi)


@dataclass(frozen=True)
class BacktestReport:
    pregame: BacktestResult
    pregame_no_fees: BacktestResult
    live: BacktestResult
    follow_market: BacktestResult
    n_markets: int
    pregame_pnl_ci: tuple[float, float]
    live_pnl_ci: tuple[float, float]


def run_backtest(
    store: Store,
    *,
    first_test_season: int = 2026,
    initial_bankroll: float = 1_000.0,
    seed: int = 1337,
    reports_dir: Path = Path("reports"),
) -> BacktestReport:
    params = StrategyParams()
    pre_opps, pre_outcomes = build_pregame_opportunities(
        store, first_test_season=first_test_season, seed=seed
    )
    live_opps, live_outcomes = build_live_opportunities(
        store, first_test_season=first_test_season, seed=seed
    )
    logger.info("opportunities built", pregame=len(pre_opps), live=len(live_opps))

    pregame = simulate(pre_opps, pre_outcomes, initial_bankroll=initial_bankroll, params=params)
    # diagnostic: identical strategy in a fee-free world
    from dataclasses import replace

    pregame_no_fees = simulate(
        pre_opps,
        pre_outcomes,
        initial_bankroll=initial_bankroll,
        params=replace(params, fee_multiplier=0.0),
    )
    live = simulate(live_opps, live_outcomes, initial_bankroll=initial_bankroll, params=params)
    follow = follow_market_baseline(pre_opps, pre_outcomes, initial_bankroll=initial_bankroll)

    ticker_to_game = {t: ctx.game_id for t, ctx in _market_contexts(store).items()}
    report = BacktestReport(
        pregame=pregame,
        pregame_no_fees=pregame_no_fees,
        live=live,
        follow_market=follow,
        n_markets=len(pre_outcomes),
        pregame_pnl_ci=pnl_bootstrap_ci(pregame, ticker_to_game, seed=seed),
        live_pnl_ci=pnl_bootstrap_ci(live, ticker_to_game, seed=seed),
    )
    _write_outputs(report, reports_dir)
    return report


def _result_row(name: str, r: BacktestResult) -> str:
    sharpe = r.sharpe_like()
    return (
        f"| {name} | {len(r.trades)} | {r.turnover:.2f} | {r.total_fees:.2f} | "
        f"{r.total_pnl:+.2f} | {r.roi:+.2%} | {r.max_drawdown:.2%} | "
        f"{'-' if sharpe is None else f'{sharpe:+.2f}'} |"
    )


def _write_outputs(report: BacktestReport, reports_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in (
        ("pregame", report.pregame),
        ("live", report.live),
        ("follow-market baseline", report.follow_market),
    ):
        if len(result.equity) > 1:
            ax.plot(np.array(result.equity_times), result.equity, label=name, linewidth=1.5)
    ax.axhline(
        report.pregame.initial_bankroll,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="never trade",
    )
    ax.set_ylabel("bankroll ($)")
    ax.set_title("Walk-forward backtest equity — 2026 test season (fees included)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(reports_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Walk-forward backtest",
        "",
        f"Universe: **{report.n_markets} settled Kalshi markets** (2026 playoffs — all the "
        "API retains; see data-retention note). Initial bankroll $1,000; taker fills at the "
        "standing top-of-book; Kalshi fee formula `ceil(0.07*C*P*(1-P))`; fractional Kelly "
        "(x0.25, 5% cap, 100-contract depth guard); **one position per game** (a game's two "
        "markets are mirror images — trading both would silently double exposure), held to "
        "settlement; live entries delayed 60s after the model's information time.",
        "",
        "| strategy | trades | turnover $ | fees $ | PnL $ | ROI | max DD | sharpe-like |",
        "|---|---|---|---|---|---|---|---|",
        _result_row("pregame (after fees)", report.pregame),
        _result_row("pregame (zero-fee diagnostic)", report.pregame_no_fees),
        _result_row("live (after fees)", report.live),
        _result_row("follow-market baseline", report.follow_market),
        "| never trade | 0 | 0.00 | 0.00 | +0.00 | +0.00% | 0.00% | - |",
        "",
        "### Is the PnL distinguishable from luck? (game-level bootstrap, 95% CI)",
        "",
        f"- pregame total PnL CI: [{report.pregame_pnl_ci[0]:+.2f}, "
        f"{report.pregame_pnl_ci[1]:+.2f}] $",
        f"- live total PnL CI: [{report.live_pnl_ci[0]:+.2f}, {report.live_pnl_ci[1]:+.2f}] $",
        "",
        "**Read this with the calibration reports, not instead of them.** With this few "
        "independent games, ROI is dominated by variance; the paired-bootstrap probability "
        "comparisons in `pregame_summary.md` / `live_summary.md` are the honest estimate of "
        "edge. A PnL interval that excludes zero on 42 games is still one lucky playoff run, "
        "not a validated strategy. The sharpe-like column is a per-trade mean/std*sqrt(n) "
        "analogue, not an annualized Sharpe. Fills assume top-of-book depth up to 100 "
        "contracts with no market impact — optimistic for real size.",
        "",
        "![equity curve](equity_curve.png)",
        "",
    ]
    (reports_dir / "backtest_summary.md").write_text("\n".join(lines))
    logger.info("backtest report written", path=str(reports_dir / "backtest_summary.md"))
