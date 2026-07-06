# nba-market-engine

[![CI](https://github.com/zane180/nba-market-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/zane180/nba-market-engine/actions/workflows/ci.yml)

Estimating NBA win probabilities — pre-game and live in-game — and evaluating them
against the probabilities implied by [Kalshi](https://kalshi.com) prediction-market
prices: calibration analysis, a walk-forward backtest with real fees, and a live
paper-trading loop.

> **Status: under construction.** This README will become the full case study
> (problem framing → architecture → methodology → results → limitations) as the
> phases below land. Until then, no performance claims are made here — and when
> they are, every number will be reproducible via `make repro` and traceable to a
> file in `reports/`.

## The premise

The NBA game-winner market is liquid and roughly efficient, which makes it a good
*measuring stick* rather than an easy profit source. The interesting questions are:

1. How well-calibrated can an open-data model get (Brier score, log loss, ECE,
   reliability diagrams)?
2. How does it compare to the de-vigged market-implied probability — overall, and
   on any identifiable subset?
3. If a residual edge exists anywhere, does it survive Kalshi's actual fee
   schedule, realistic order-book liquidity, and conservative position sizing in a
   walk-forward backtest with strict no-lookahead guards?

If the answer to (3) is "no" — the likely answer for an efficient market — this
project will say so plainly. The deliverable is honest measurement, not a
"profitable bot" claim.

## Quick start

```bash
make setup      # uv sync (Python 3.12, pinned deps)
make check      # ruff + mypy --strict + pytest — same as CI
uv run engine --help
```

Configuration is environment-only (see `.env.example`). Paper trading is the
default and only enabled execution path; real-money order placement is off by
default and multiply gated.

## Roadmap

- [x] Phase 0 — scaffold: tooling, CI, typed domain models, CLI skeleton
- [ ] Phase 1 — ingestion: ESPN + Kalshi clients, team-mapping layer
- [ ] Phase 2 — storage + historical backfill
- [ ] Phase 3 — pre-game model (Elo → GBM) + calibration vs. market
- [ ] Phase 4 — live in-game win-probability model
- [ ] Phase 5 — walk-forward backtest engine (fees, sizing, leakage guards)
- [ ] Phase 6 — live paper-trading loop
- [ ] Phase 7 — case-study writeup, notebooks, polish

## License

MIT
