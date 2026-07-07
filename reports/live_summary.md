# Live in-game win-probability evaluation

Walk-forward test season(s): **2026**; 642,889 test snapshots. Snapshots within a game are correlated — all uncertainty below is bootstrapped **by game**, not by snapshot.

## Full test set

| model | n | Brier | log loss | ECE |
|---|---|---|---|---|
| Live WP model (calibrated) | 642,889 | 0.1514 | 0.4531 | 0.0106 |
| Pre-game prior only (Elo) | 642,889 | 0.2119 | 0.6145 | 0.0655 |

## In-game market comparison

Snapshots joinable to a fresh two-sided Kalshi quote (< 15 min old): **19,350** across **42 games** (playoffs only — see data-retention note in the README).

- model Brier: **0.1855**
- market Brier: **0.1776**
- difference (model - market): +0.0079, 95% game-bootstrap CI [-0.0138, +0.0293] — **not distinguishable**

![reliability](calibration_live.png)

![example game](live_game_example.png)
