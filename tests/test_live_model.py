"""Live feature/model/join correctness: time math incl. OT, label isolation,
grouped calibration, and the no-lookahead in-game market join."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from engine.data.models import Candle, Game, GameStatus, LiveGameState, MarketInfo, MarketResult
from engine.data.store import Store
from engine.features.live import (
    LIVE_FEATURE_NAMES,
    build_live_dataset,
    compute_live_features,
    seconds_remaining_regulation,
)
from engine.models.live_evaluation import (
    game_grouped_bootstrap_brier_diff,
    join_market_ingame,
)
from engine.models.live_wp_model import LiveSeasonPredictions, LiveWpModel, walk_forward_live

TIP = datetime(2026, 6, 14, 0, 30, tzinfo=UTC)


class TestTimeMath:
    @pytest.mark.parametrize(
        ("period", "secs_in_period", "expected"),
        [
            (1, 720.0, 2880.0),  # opening tip
            (2, 300.0, 1740.0),
            (4, 0.0, 0.0),  # regulation buzzer
            (5, 300.0, 0.0),  # OT start: no regulation time left
            (6, 12.5, 0.0),
        ],
    )
    def test_seconds_remaining_regulation(
        self, period: int, secs_in_period: float, expected: float
    ) -> None:
        assert seconds_remaining_regulation(period, secs_in_period) == expected

    def test_ot_features(self) -> None:
        features = compute_live_features(
            score_diff=np.array([3.0]),
            seconds_regulation=np.array([0.0]),
            prior_home_prob=np.array([0.6]),
            period=np.array([5]),
            seconds_in_period=np.array([120.0]),
        )
        row = dict(zip(LIVE_FEATURE_NAMES, features[0], strict=True))
        assert row["is_overtime"] == 1.0
        assert row["ot_seconds_remaining"] == 120.0
        # denominator uses OT time, not zero
        assert row["diff_per_sqrt_time"] == pytest.approx(3.0 / np.sqrt(150.0))

    def test_buzzer_ratio_is_finite(self) -> None:
        features = compute_live_features(
            score_diff=np.array([1.0]),
            seconds_regulation=np.array([0.0]),
            prior_home_prob=np.array([0.5]),
            period=np.array([4]),
            seconds_in_period=np.array([0.0]),
        )
        assert np.isfinite(features).all()


def _store_with_game(tmp_path: Path) -> Store:
    store = Store(f"sqlite:///{tmp_path}/live.db")
    store.init_schema()
    return store


def _snapshot(
    game_id: str, *, minute: int, period: int = 2, home: int = 50, away: int = 48
) -> LiveGameState:
    return LiveGameState(
        game_id=game_id,
        as_of=TIP + timedelta(minutes=minute),
        period=period,
        seconds_remaining_in_period=300.0,
        home_score=home,
        away_score=away,
        source_play_id=f"{game_id}-p{minute}",
    )


class TestBuildLiveDataset:
    def make_games(self, n: int = 4) -> list[Game]:
        teams = ["Boston Celtics", "Miami Heat", "Denver Nuggets", "Phoenix Suns"]
        games = []
        for i in range(n):
            home, away = teams[i % 4], teams[(i + 1) % 4]
            games.append(
                Game(
                    game_id=f"g{i}",
                    home_team=home,
                    away_team=away,
                    start_time=TIP - timedelta(days=n - i),
                    status=GameStatus.FINAL,
                    home_score=100 + (i % 2),
                    away_score=100 - (i % 2) + (0 if i % 2 else 1),
                    season_type=2,
                )
            )
        return games

    def test_labels_come_from_game_not_snapshot(self, tmp_path: Path) -> None:
        """A snapshot where home trails must still carry label=1 if home
        eventually won — the label is the game outcome, never current score."""
        store = _store_with_game(tmp_path)
        store.upsert_games(self.make_games())
        # g0: home won (score 100 vs 99... wait i%2==0 -> home 100 away 101)
        game = store.game("g0")
        assert game is not None
        store.upsert_snapshots([_snapshot("g0", minute=10, home=40, away=60)])
        data = build_live_dataset(store)
        assert len(data) == 1
        assert data.labels[0] == (1.0 if game.home_won else 0.0)
        # and the trailing scoreline is intact in features
        assert data.features[0][0] == -20.0

    def test_snapshot_without_final_game_excluded(self, tmp_path: Path) -> None:
        store = _store_with_game(tmp_path)
        store.upsert_games(self.make_games())
        store.upsert_snapshots([_snapshot("orphan", minute=1)])
        assert len(build_live_dataset(store)) == 0

    def test_prior_matches_pregame_elo(self, tmp_path: Path) -> None:
        from engine.features.pregame import build_feature_rows

        store = _store_with_game(tmp_path)
        games = self.make_games()
        store.upsert_games(games)
        store.upsert_snapshots([_snapshot("g3", minute=5)])
        data = build_live_dataset(store)
        expected_prior = {r.game_id: r.elo_home_prob for r in build_feature_rows(games)}["g3"]
        assert data.features[0][3] == pytest.approx(expected_prior)


def synthetic_live_dataset(n_games: int, seasons: list[int], seed: int = 3) -> LiveDataset:
    """Games whose outcome is driven by the (noisy) score path, built via the
    real store + builder so shapes stay honest."""
    from engine.features.live import LiveDataset  # noqa: F401 — return type

    rng = random.Random(seed)
    teams = ["Boston Celtics", "Miami Heat", "Denver Nuggets", "Phoenix Suns"]
    games: list[Game] = []
    snapshots: list[LiveGameState] = []
    idx = 0
    for season in seasons:
        for season_game in range(n_games):
            home, away = rng.sample(teams, 2)
            # spread within Nov-Apr of one season; never bleed past August
            start = datetime(season - 1, 11, 1, tzinfo=UTC) + timedelta(
                minutes=(150 * 24 * 60 // max(1, n_games)) * season_game
            )
            drift = rng.gauss(0, 8)
            diff = 0.0
            for minute in range(0, 48, 2):
                diff += rng.gauss(drift / 24, 3)
                period = min(4, minute // 12 + 1)
                secs = 720.0 - (minute % 12) * 60.0
                base = 80 + minute
                snapshots.append(
                    LiveGameState(
                        game_id=f"lg{idx}",
                        as_of=start + timedelta(minutes=minute + 1),
                        period=period,
                        seconds_remaining_in_period=secs,
                        home_score=base + max(0, round(diff)),
                        away_score=base + max(0, round(-diff)),
                        source_play_id=f"lg{idx}-{minute}",
                    )
                )
            final_diff = round(diff) if round(diff) != 0 else (1 if drift > 0 else -1)
            games.append(
                Game(
                    game_id=f"lg{idx}",
                    home_team=home,
                    away_team=away,
                    start_time=start,
                    status=GameStatus.FINAL,
                    home_score=128 + max(0, final_diff),
                    away_score=128 + max(0, -final_diff),
                    season_type=2,
                )
            )
            idx += 1
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(f"sqlite:///{tmp}/syn.db")
        store.init_schema()
        store.upsert_games(games)
        store.upsert_snapshots(snapshots)
        return build_live_dataset(store)


class TestLiveWpModel:
    def test_refuses_tiny_training_set(self) -> None:
        data = synthetic_live_dataset(10, [2025])
        with pytest.raises(ValueError, match="refusing to fit"):
            LiveWpModel().fit(data)

    def test_walk_forward_learns_score_time_relationship(self) -> None:
        data = synthetic_live_dataset(500, [2025, 2026])
        preds = walk_forward_live(data, first_test_season=2026)
        assert [p.season for p in preds] == [2026]
        test = preds[0]
        from engine.backtest.metrics import brier_score

        assert brier_score(test.model, test.data.labels) < 0.22
        # late, large leads must be called more confidently than early ties
        late_big_lead = (test.data.features[:, 0] >= 10) & (test.data.features[:, 1] < 360)
        early_tie = (np.abs(test.data.features[:, 0]) <= 1) & (test.data.features[:, 1] > 2400)
        assert test.model[late_big_lead].mean() > 0.8
        assert 0.3 < test.model[early_tie].mean() < 0.7

    def test_deterministic(self) -> None:
        data = synthetic_live_dataset(500, [2025, 2026])
        a = walk_forward_live(data, first_test_season=2026, seed=5)[0].model
        b = walk_forward_live(data, first_test_season=2026, seed=5)[0].model
        assert np.array_equal(a, b)


class TestInGameMarketJoin:
    def _setup(self, tmp_path: Path) -> tuple[Store, LiveSeasonPredictions]:
        store = _store_with_game(tmp_path)
        game = Game(
            game_id="g1",
            home_team="San Antonio Spurs",
            away_team="New York Knicks",
            start_time=TIP,
            status=GameStatus.FINAL,
            home_score=90,
            away_score=94,
            season_type=3,
        )
        store.upsert_games([game])
        for team, ticker in (
            ("San Antonio Spurs", "E-SAS"),
            ("New York Knicks", "E-NYK"),
        ):
            store.upsert_markets(
                [
                    MarketInfo(
                        ticker=ticker,
                        event_ticker="E",
                        game_id="g1",
                        yes_team=team,
                        result=MarketResult.NO,
                        open_time=TIP - timedelta(days=1),
                        close_time=TIP + timedelta(hours=3),
                        volume=1.0,
                    )
                ]
            )

        def mk_candle(ticker: str, minute: int, bid: int, ask: int) -> Candle:
            return Candle(
                ticker=ticker,
                end_time=TIP + timedelta(minutes=minute),
                period_seconds=60,
                yes_bid_close=bid,
                yes_ask_close=ask,
                trade_close=None,
                volume=1.0,
                open_interest=1.0,
            )

        store.upsert_candles(
            [
                mk_candle("E-SAS", 10, 60, 62),
                mk_candle("E-SAS", 30, 70, 72),
                mk_candle("E-NYK", 10, 38, 40),
                mk_candle("E-NYK", 30, 28, 30),
            ]
        )
        # two model snapshots: minute 12 (should see minute-10 candles) and
        # minute 30 (sees minute-30)
        fake_pred = LiveSeasonPredictions(
            season=2026,
            data=_dataset_for(store, minutes=[12, 30]),
            model=np.array([0.55, 0.6]),
        )
        return store, fake_pred

    def test_join_uses_latest_candle_at_or_before_snapshot(self, tmp_path: Path) -> None:
        store, pred = self._setup(tmp_path)
        joined = join_market_ingame(store, [pred])
        assert len(joined) == 2
        # snapshot @12min -> candles @10: raw .61 vs .39 -> devig .61
        assert joined.market[0] == pytest.approx(0.61 / (0.61 + 0.39))
        # snapshot @30min -> candles @30: raw .71 vs .29
        assert joined.market[1] == pytest.approx(0.71 / (0.71 + 0.29))

    def test_stale_quotes_dropped(self, tmp_path: Path) -> None:
        store, pred = self._setup(tmp_path)
        joined = join_market_ingame(store, [pred], max_quote_age_seconds=60)
        # snapshot @12min is 2min after the freshest candle -> dropped
        assert len(joined) == 1


def _dataset_for(store: Store, *, minutes: list[int]) -> LiveDataset:

    store.upsert_snapshots([_snapshot("g1", minute=m, period=3, home=60, away=55) for m in minutes])
    return build_live_dataset(store)


from engine.features.live import LiveDataset  # noqa: E402


class TestGroupedBootstrap:
    def test_resamples_games_not_snapshots(self) -> None:
        # 2 games x many identical snapshots; per-game resampling must produce
        # CI width driven by n_games=2, i.e. huge — snapshot-level would be tiny
        from engine.models.live_evaluation import InGameMarketSeries

        n = 200
        series = InGameMarketSeries(
            game_ids=["a"] * n + ["b"] * n,
            as_of_epoch=np.arange(2 * n, dtype=np.int64),
            model=np.array([0.9] * n + [0.2] * n),
            market=np.array([0.6] * n + [0.6] * n),
            labels=np.array([1.0] * n + [0.0] * n),
        )
        point, lo, hi = game_grouped_bootstrap_brier_diff(series, n_resamples=500)
        assert point < 0  # model scored better here
        # with only 2 games the interval must span the per-game extremes
        assert hi - lo > 0.1
