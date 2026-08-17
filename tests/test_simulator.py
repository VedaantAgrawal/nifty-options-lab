"""Tests for engine.simulator.run_single_trade.

All tests price off hand-crafted multi-day DataFrames passed via `chain=`
-- no network access, no dependency on data/cache/.
"""
import logging
from datetime import date

import pandas as pd
import pytest

from engine.position_builder import MissingOptionsChainDataError
from engine.simulator import run_single_trade
from models.schemas import StrategyConfig

COLUMNS = ["date", "expiry_date", "strike", "option_type", "close", "oi", "volume"]

# Tuesday in the pre-Nov-2024 (Thursday) weekly-expiry regime.
ENTRY_DATE = date(2022, 3, 1)
DAY_2 = date(2022, 3, 2)
EXPIRY = date(2022, 3, 3)  # next Thursday
ALL_DAYS = [ENTRY_DATE, DAY_2, EXPIRY]
SPOT = 24700.0
DEFAULT_CLOSE = 20.0


def make_chain(prices_by_day, default_close=DEFAULT_CLOSE, strikes=range(24000, 25200, 50)):
    """prices_by_day: {day: {(option_type, strike): close}}. Any (day, type, strike)
    not listed falls back to `default_close`."""
    rows = []
    for strike in strikes:
        for d in ALL_DAYS:
            for option_type in ("CE", "PE"):
                close = prices_by_day.get(d, {}).get((option_type, strike), default_close)
                rows.append(
                    {
                        "date": d,
                        "expiry_date": EXPIRY,
                        "strike": strike,
                        "option_type": option_type,
                        "close": close,
                        "oi": 100,
                        "volume": 100,
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def make_strangle_config(**overrides):
    defaults = dict(
        structure="short_strangle",
        expiry_cycle="weekly",
        entry_day_of_week=1,
        days_to_expiry_at_entry=2,
        otm_points_call=200,
        otm_points_put=200,
        stop_loss_pct=30,
        capital=10_000_000,
        reentry="none",
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def make_iron_condor_config(**overrides):
    defaults = dict(
        structure="iron_condor",
        expiry_cycle="weekly",
        entry_day_of_week=1,
        days_to_expiry_at_entry=2,
        otm_points_call=200,
        otm_points_put=200,
        wing_width_points=150,
        stop_loss_pct=30,
        capital=10_000_000,
        reentry="none",
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


# Short call strike is 24900 (24700 + 200), short put strike is 24500 (24700 - 200).


class TestStopLossPath:
    def test_exits_early_on_stop_loss_day(self):
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 90.0, ("PE", 24500.0): 40.0},  # combined 130 >= 130 (30% of 100)
            EXPIRY: {("CE", 24900.0): 5.0, ("PE", 24500.0): 5.0},
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert trade is not None
        assert trade.exit_reason == "stop_loss"
        assert trade.exit_date == DAY_2

    def test_stop_loss_pnl_is_computed_from_exit_day_prices(self):
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 90.0, ("PE", 24500.0): 40.0},
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain, lot_size=75)
        # CE: sell, (50-90) = -40; PE: sell, (50-40) = +10; total -30 * 75 = -2250
        assert trade.pnl == pytest.approx(-2250.0)

    def test_exactly_at_threshold_triggers_stop_loss(self):
        # entry combined = 100, stop_loss_pct=30 -> threshold is exactly 130
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 80.0, ("PE", 24500.0): 50.0},  # combined exactly 130
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(stop_loss_pct=30), SPOT, chain=chain)
        assert trade.exit_reason == "stop_loss"
        assert trade.exit_date == DAY_2

    def test_below_threshold_does_not_trigger_stop_loss(self):
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 70.0, ("PE", 24500.0): 50.0},  # combined 120 < 130
            EXPIRY: {("CE", 24900.0): 5.0, ("PE", 24500.0): 5.0},
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(stop_loss_pct=30), SPOT, chain=chain)
        assert trade.exit_reason == "expiry"


class TestExpiryPath:
    def test_exits_at_expiry_when_no_stop_loss_hit(self):
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 55.0, ("PE", 24500.0): 45.0},
            EXPIRY: {("CE", 24900.0): 2.0, ("PE", 24500.0): 3.0},
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert trade.exit_reason == "expiry"
        assert trade.exit_date == EXPIRY

    def test_expiry_pnl_uses_final_day_close_as_settlement(self):
        prices = {
            ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0},
            DAY_2: {("CE", 24900.0): 55.0, ("PE", 24500.0): 45.0},
            EXPIRY: {("CE", 24900.0): 0.0, ("PE", 24500.0): 0.0},  # both expired worthless
        }
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain, lot_size=75)
        # both legs sold at 50, expired worthless -> full premium kept: (50+50)*75
        assert trade.pnl == pytest.approx(7500.0)

    def test_all_legs_have_exit_price_set_on_a_closed_trade(self):
        prices = {ENTRY_DATE: {("CE", 24900.0): 50.0, ("PE", 24500.0): 50.0}}
        chain = make_chain(prices)
        trade = run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert all(leg.exit_price is not None for leg in trade.legs)


class TestIronCondorSimulation:
    def test_returns_four_legs_and_exits_at_expiry(self):
        chain = make_chain({})
        trade = run_single_trade(ENTRY_DATE, make_iron_condor_config(), SPOT, chain=chain)
        assert trade is not None
        assert len(trade.legs) == 4
        assert trade.exit_reason == "expiry"

    def test_iron_condor_margin_is_wing_width_based_not_notional_pct(self):
        # wing_width_points=150, lot_size default 75 -> margin ~ 150*75 = 11250,
        # far less than a short_strangle's notional-based margin would be.
        chain = make_chain({})
        cheap_capital_config = make_iron_condor_config(capital=15_000)
        trade = run_single_trade(ENTRY_DATE, cheap_capital_config, SPOT, chain=chain)
        assert trade is not None
        assert trade.capital_at_risk == pytest.approx(150 * 75)


class TestInsufficientMargin:
    def test_returns_none_when_margin_exceeds_capital(self):
        chain = make_chain({})
        broke_config = make_strangle_config(capital=100)
        result = run_single_trade(ENTRY_DATE, broke_config, SPOT, chain=chain)
        assert result is None

    def test_logs_a_warning_with_the_reason_when_skipped(self, caplog):
        chain = make_chain({})
        broke_config = make_strangle_config(capital=100)
        with caplog.at_level(logging.WARNING, logger="engine.simulator"):
            run_single_trade(ENTRY_DATE, broke_config, SPOT, chain=chain)
        assert any("required margin" in record.message for record in caplog.records)
        assert any("skipped" in record.message for record in caplog.records)

    def test_sufficient_capital_does_not_skip(self):
        chain = make_chain({})
        rich_config = make_strangle_config(capital=10_000_000)
        result = run_single_trade(ENTRY_DATE, rich_config, SPOT, chain=chain)
        assert result is not None


class TestMissingChainData:
    def test_raises_when_no_trading_days_in_range(self):
        # Chain only has data for a completely different expiry/date range.
        other_day = date(2022, 3, 10)
        other_expiry = date(2022, 3, 10)
        rows = [
            {
                "date": other_day,
                "expiry_date": other_expiry,
                "strike": strike,
                "option_type": option_type,
                "close": 20.0,
                "oi": 100,
                "volume": 100,
            }
            for strike in range(24000, 25200, 50)
            for option_type in ("CE", "PE")
        ]
        chain = pd.DataFrame(rows, columns=COLUMNS)
        with pytest.raises(MissingOptionsChainDataError):
            run_single_trade(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
