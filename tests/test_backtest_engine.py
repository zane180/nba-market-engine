"""Backtest correctness: fee math against Kalshi's published formula, Kelly
edge cases, and hand-verifiable bankroll accounting through the simulator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from engine.backtest.engine import (
    BacktestResult,
    Opportunity,
    StrategyParams,
    decide,
    simulate,
)
from engine.backtest.fees import fee_per_contract_dollars, taker_fee_dollars
from engine.backtest.metrics import max_drawdown
from engine.backtest.sizing import SizingParams, kelly_fraction, size_contracts
from engine.data.models import Side

T0 = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)


class TestFees:
    @pytest.mark.parametrize(
        ("price_cents", "contracts", "expected"),
        [
            (50, 100, 1.75),  # 0.07*100*0.25 = 1.75 exactly
            (50, 1, 0.02),  # 0.00175 -> ceil to next cent
            (99, 100, 0.07),  # 0.07*100*0.0099 = 0.0693 -> 0.07
            (1, 100, 0.07),  # symmetric at the other extreme
            (50, 0.5, 0.01),  # fractional contracts still round up
        ],
    )
    def test_matches_published_formula(
        self, price_cents: int, contracts: float, expected: float
    ) -> None:
        assert taker_fee_dollars(price_cents, contracts) == pytest.approx(expected)

    def test_exact_cent_not_rounded_up(self) -> None:
        # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly — must stay 1.75, not 1.76
        assert taker_fee_dollars(50, 100) == 1.75

    def test_fee_maximal_at_fifty_cents(self) -> None:
        fees = [fee_per_contract_dollars(p) for p in range(1, 100)]
        assert max(fees) == fee_per_contract_dollars(50)

    @pytest.mark.parametrize("price", [0, 100])
    def test_untradable_price_rejected(self, price: int) -> None:
        with pytest.raises(ValueError):
            taker_fee_dollars(price, 1)


class TestKelly:
    def test_hand_computed_without_fees(self) -> None:
        # q=0.6, p=0.5 -> (0.6-0.5)/0.5 = 0.2
        assert kelly_fraction(0.6, 50, fee_multiplier=0.0) == pytest.approx(0.2)

    def test_fees_shrink_the_fraction(self) -> None:
        with_fees = kelly_fraction(0.6, 50)
        without = kelly_fraction(0.6, 50, fee_multiplier=0.0)
        assert 0 < with_fees < without

    def test_no_edge_sizes_to_zero(self) -> None:
        assert kelly_fraction(0.5, 50) == 0.0
        assert kelly_fraction(0.3, 50) == 0.0

    def test_edge_smaller_than_fee_sizes_to_zero(self) -> None:
        # 1c of raw edge at p=0.50; fee/contract = 0.0175 > 0.01
        assert kelly_fraction(0.51, 50) == 0.0
        assert kelly_fraction(0.51, 50, fee_multiplier=0.0) > 0.0

    def test_cap_binds(self) -> None:
        params = SizingParams(kelly_multiplier=1.0, max_fraction=0.05, max_contracts=1e9)
        contracts = size_contracts(q=0.95, price_cents=50, bankroll=1000.0, params=params)
        assert contracts * 0.5 == pytest.approx(50.0)  # 5% of bankroll, not full Kelly

    def test_contract_depth_guard_binds(self) -> None:
        params = SizingParams(kelly_multiplier=1.0, max_fraction=1.0, max_contracts=10)
        assert size_contracts(q=0.99, price_cents=50, bankroll=1e6, params=params) == 10

    def test_zero_bankroll_never_trades(self) -> None:
        assert size_contracts(q=0.9, price_cents=50, bankroll=0.0, params=SizingParams()) == 0.0

    def test_invalid_q_raises(self) -> None:
        with pytest.raises(ValueError):
            kelly_fraction(1.5, 50)


def opportunity(
    ticker: str = "M1",
    *,
    minute: int = 0,
    model_prob: float = 0.7,
    yes_bid: int | None = 48,
    yes_ask: int | None = 50,
    settle_minute: int = 200,
) -> Opportunity:
    return Opportunity(
        ticker=ticker,
        decision_time=T0 + timedelta(minutes=minute),
        model_prob=model_prob,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        settle_time=T0 + timedelta(minutes=settle_minute),
    )


class TestDecide:
    def test_buys_yes_when_model_high(self) -> None:
        result = decide(opportunity(model_prob=0.70), 1000.0, StrategyParams())
        assert result is not None
        side, contracts, price = result
        assert side is Side.YES and price == 50 and contracts > 0

    def test_buys_no_when_model_low(self) -> None:
        result = decide(opportunity(model_prob=0.30), 1000.0, StrategyParams())
        assert result is not None
        side, _, price = result
        assert side is Side.NO
        assert price == 100 - 48  # implied NO ask from YES bid

    def test_no_trade_inside_threshold(self) -> None:
        # model 0.53 vs ask 0.50: EV = 0.03 - fee 0.0175 = 0.0125 < 0.02
        assert decide(opportunity(model_prob=0.53), 1000.0, StrategyParams()) is None

    def test_no_trade_when_book_empty(self) -> None:
        assert (
            decide(
                opportunity(model_prob=0.9, yes_bid=None, yes_ask=None),
                1000.0,
                StrategyParams(),
            )
            is None
        )

    def test_spread_blocks_both_sides(self) -> None:
        # wide book: ask 60 / bid 40; model 0.5 has no edge either way
        assert (
            decide(opportunity(model_prob=0.5, yes_bid=40, yes_ask=60), 1000.0, StrategyParams())
            is None
        )


class TestSimulate:
    def test_hand_verified_winning_trade(self) -> None:
        """q=0.7 vs ask 50c, bankroll 1000, kelly x0.25 cap 5%:
        effective price = 0.5 + 0.0175 = 0.5175
        f* = (0.7 - 0.5175)/(1 - 0.5175) = 0.378..; x0.25 = 0.0946 -> capped 0.05
        stake = $50 -> 100 contracts @ 50c, but depth guard caps at 100 -> equal
        fee = ceil(0.07*100*0.25) = $1.75; cost = $51.75
        YES settles true -> payout $100; final = 1000 - 51.75 + 100 = 1048.25
        """
        result = simulate([opportunity(model_prob=0.7)], {"M1": True})
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.contracts == pytest.approx(100.0)
        assert trade.fee == pytest.approx(1.75)
        assert trade.cost == pytest.approx(51.75)
        assert result.final_bankroll == pytest.approx(1048.25)
        assert result.total_fees == pytest.approx(1.75)
        assert result.win_rate == 1.0

    def test_hand_verified_losing_trade(self) -> None:
        result = simulate([opportunity(model_prob=0.7)], {"M1": False})
        assert result.final_bankroll == pytest.approx(1000.0 - 51.75)
        assert result.win_rate == 0.0

    def test_one_position_per_market(self) -> None:
        opps = [opportunity(minute=m) for m in range(5)]
        result = simulate(opps, {"M1": True})
        assert len(result.trades) == 1

    def test_one_position_per_game_across_mirror_markets(self) -> None:
        """A game's two markets are the same bet in opposite clothing: buying
        YES home and NO away doubles exposure. The group key must prevent it."""
        home_mkt = Opportunity(
            ticker="G1-HOME",
            decision_time=T0,
            model_prob=0.7,  # model loves the home team
            yes_bid=48,
            yes_ask=50,
            settle_time=T0 + timedelta(hours=3),
            group="game-1",
        )
        away_mkt = Opportunity(
            ticker="G1-AWAY",
            decision_time=T0 + timedelta(minutes=1),
            model_prob=0.3,  # same belief, mirrored
            yes_bid=50,
            yes_ask=52,
            settle_time=T0 + timedelta(hours=3),
            group="game-1",
        )
        result = simulate([home_mkt, away_mkt], {"G1-HOME": True, "G1-AWAY": False})
        assert len(result.trades) == 1

    def test_settlement_frees_bankroll_for_later_trades(self) -> None:
        early = opportunity("A", minute=0, settle_minute=10)
        late = opportunity("B", minute=20)
        result = simulate([early, late], {"A": True, "B": True}, initial_bankroll=60.0)
        assert len(result.trades) == 2  # A settled before B's decision

    def test_never_levers_up(self) -> None:
        # bankroll too small to cover cost at max sizing -> second trade skipped
        a = opportunity("A", minute=0, settle_minute=500)
        b = opportunity("B", minute=1, settle_minute=500, model_prob=0.99, yes_ask=99, yes_bid=98)
        result = simulate([a, b], {"A": True, "B": True}, initial_bankroll=52.0)
        # trade A costs ~51.75 of the 52 bankroll; B cannot fund a min stake
        total_cost = sum(t.cost for t in result.trades)
        assert total_cost <= 52.0

    def test_zero_fee_diagnostic_differs(self) -> None:
        fees_on = simulate([opportunity(model_prob=0.7)], {"M1": True})
        fees_off = simulate(
            [opportunity(model_prob=0.7)],
            {"M1": True},
            params=StrategyParams(fee_multiplier=0.0),
        )
        assert fees_off.total_fees == 0.0
        assert fees_off.final_bankroll > fees_on.final_bankroll

    def test_missing_outcome_raises(self) -> None:
        with pytest.raises(KeyError, match="M1"):
            simulate([opportunity(model_prob=0.7)], {})

    def test_no_opportunities_is_flat(self) -> None:
        result = simulate([], {})
        assert result.final_bankroll == result.initial_bankroll
        assert result.roi == 0.0

    def test_equity_curve_marks_open_positions_at_cost(self) -> None:
        result = simulate([opportunity(model_prob=0.7)], {"M1": True})
        # after entry, equity = bankroll_cash + open cost = initial (no MTM jumps)
        assert result.equity[0] == pytest.approx(1000.0)
        assert result.equity[-1] == pytest.approx(1048.25)


class TestMaxDrawdown:
    def test_hand_case(self) -> None:
        curve = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
        # worst: 120 -> 80 = 1/3
        assert max_drawdown(curve) == pytest.approx(1 / 3)

    def test_monotone_curve_has_zero(self) -> None:
        assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_result_property(self) -> None:
        result = BacktestResult(
            initial_bankroll=100.0,
            final_bankroll=90.0,
            trades=[],
            equity_times=[T0, T0],
            equity=np.array([100.0, 90.0]),
        )
        assert result.max_drawdown == pytest.approx(0.1)
