"""Tests for engine.metrics.compute_metrics.

Expected values for the max_drawdown and sharpe_ratio scenarios were
independently hand-computed (not just read back from the implementation)
before being locked into these tests -- see the PR/commit discussion for
the by-hand derivation.
"""
import math
import statistics
from datetime import date

import pytest

from engine.metrics import compute_metrics
from models.schemas import OptionLeg, Trade

_DAYS_PER_YEAR = 365.25


def make_trade(entry_date, expiry_date, exit_date, pnl, exit_reason="expiry", capital_at_risk=100_000.0):
    leg = OptionLeg(strike=24500, option_type="CE", side="sell", entry_price=100.0, exit_price=50.0, lots=1)
    return Trade(
        entry_date=entry_date,
        expiry_date=expiry_date,
        exit_date=exit_date,
        legs=[leg],
        exit_reason=exit_reason,
        pnl=pnl,
        capital_at_risk=capital_at_risk,
    )


class TestEmptyTrades:
    def test_raises_without_initial_capital(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_returns_zeroed_result_with_initial_capital(self):
        result = compute_metrics([], initial_capital=100_000)
        assert result.num_trades == 0
        assert result.total_return_pct == 0.0
        assert result.total_return_abs == 0.0
        assert result.win_rate == 0.0
        assert result.max_drawdown == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.equity_curve == []

    def test_num_skipped_insufficient_margin_passthrough_on_empty(self):
        result = compute_metrics([], initial_capital=100_000, num_skipped_insufficient_margin=7)
        assert result.num_skipped_insufficient_margin == 7


class TestTotalReturn:
    def test_total_return_pct_and_abs(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=5000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-2000),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.total_return_abs == pytest.approx(3000.0)
        assert result.total_return_pct == pytest.approx(3.0)

    def test_capital_basis_falls_back_to_first_trade_capital_at_risk(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000, capital_at_risk=50_000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=1000, capital_at_risk=60_000),
        ]
        result = compute_metrics(trades)  # no initial_capital given
        # total pnl 2000 / capital_basis 50000 (first trade's capital_at_risk) = 4%
        assert result.total_return_pct == pytest.approx(4.0)


class TestWinRate:
    def test_mixed_results(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-1),
            make_trade(date(2024, 1, 12), date(2024, 1, 18), date(2024, 1, 18), pnl=1),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.win_rate == pytest.approx(200 / 3)

    def test_all_winning(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=1),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.win_rate == pytest.approx(100.0)

    def test_all_losing(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=-1),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-1),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.win_rate == pytest.approx(0.0)

    def test_zero_pnl_trade_does_not_count_as_a_win(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=0),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=1),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.win_rate == pytest.approx(50.0)


class TestAvgPnlPerTrade:
    def test_average(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-400),
            make_trade(date(2024, 1, 12), date(2024, 1, 18), date(2024, 1, 18), pnl=600),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.avg_pnl_per_trade == pytest.approx(400.0)
        assert result.num_trades == 3


class TestEquityCurveAndDrawdown:
    def test_equity_curve_points(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=10_000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-30_000),
            make_trade(date(2024, 1, 12), date(2024, 1, 18), date(2024, 1, 18), pnl=5_000),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.equity_curve == [
            (date(2024, 1, 1), 100_000.0),
            (date(2024, 1, 4), 110_000.0),
            (date(2024, 1, 11), 80_000.0),
            (date(2024, 1, 18), 85_000.0),
        ]

    def test_max_drawdown_is_worst_peak_to_trough_decline(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=10_000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=-30_000),
            make_trade(date(2024, 1, 12), date(2024, 1, 18), date(2024, 1, 18), pnl=5_000),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        # peak 110,000 -> trough 80,000 -> drawdown = -30000/110000
        assert result.max_drawdown == pytest.approx(-30_000 / 110_000)

    def test_equity_curve_sorted_by_exit_date_even_if_input_unordered(self):
        early = make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000)
        late = make_trade(date(2024, 1, 12), date(2024, 1, 18), date(2024, 1, 18), pnl=2000)
        # passed in reverse chronological order
        result = compute_metrics([late, early], initial_capital=100_000)
        dates_in_curve = [d for d, _ in result.equity_curve]
        assert dates_in_curve == sorted(dates_in_curve)

    def test_monotonically_increasing_equity_has_zero_drawdown(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=2000),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.max_drawdown == pytest.approx(0.0)


class TestSharpeRatio:
    def test_fewer_than_two_trades_is_zero(self):
        trades = [make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000)]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.sharpe_ratio == 0.0

    def test_zero_variance_returns_is_zero(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000),
            make_trade(date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 11), pnl=1000),
        ]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.sharpe_ratio == 0.0

    def test_matches_independently_hand_computed_value(self):
        # Two trades, returns 0.1 and 0.3 on a 10,000 capital basis, spanning
        # exactly (2025-01-04 - 2024-01-01).days = 369 calendar days.
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000, capital_at_risk=10_000),
            make_trade(date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 4), pnl=3000, capital_at_risk=10_000),
        ]
        result = compute_metrics(trades, initial_capital=10_000)

        returns = [0.1, 0.3]
        mean_r = statistics.mean(returns)
        stdev_r = statistics.stdev(returns)
        span_days = (date(2025, 1, 4) - date(2024, 1, 1)).days
        years_spanned = span_days / _DAYS_PER_YEAR
        periods_per_year = 2 / years_spanned
        expected_sharpe = (mean_r / stdev_r) * math.sqrt(periods_per_year)

        assert result.sharpe_ratio == pytest.approx(expected_sharpe)

    def test_nonzero_risk_free_rate_reduces_sharpe_for_positive_returns(self):
        trades = [
            make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000, capital_at_risk=10_000),
            make_trade(date(2025, 1, 1), date(2025, 1, 4), date(2025, 1, 4), pnl=3000, capital_at_risk=10_000),
        ]
        zero_rf = compute_metrics(trades, initial_capital=10_000, risk_free_rate_annual=0.0)
        with_rf = compute_metrics(trades, initial_capital=10_000, risk_free_rate_annual=0.05)
        assert with_rf.sharpe_ratio < zero_rf.sharpe_ratio


class TestNumSkippedInsufficientMargin:
    def test_passthrough_with_trades_present(self):
        trades = [make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000)]
        result = compute_metrics(trades, initial_capital=100_000, num_skipped_insufficient_margin=3)
        assert result.num_skipped_insufficient_margin == 3

    def test_defaults_to_zero(self):
        trades = [make_trade(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 4), pnl=1000)]
        result = compute_metrics(trades, initial_capital=100_000)
        assert result.num_skipped_insufficient_margin == 0
