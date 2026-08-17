"""Tests for engine.position_builder.

All tests price off hand-crafted in-memory DataFrames passed via the
`chain=` parameter -- no network access and no dependency on data/cache/.
"""
import logging
from datetime import date

import pandas as pd
import pytest

from engine.position_builder import MissingOptionsChainDataError, build_position
from models.schemas import StrategyConfig

COLUMNS = ["date", "expiry_date", "strike", "option_type", "close", "oi", "volume"]


def make_chain_row(d, expiry, strike, option_type, close):
    return {
        "date": d,
        "expiry_date": expiry,
        "strike": strike,
        "option_type": option_type,
        "close": close,
        "oi": 1000,
        "volume": 500,
    }


def make_strangle_config(**overrides):
    defaults = dict(
        structure="short_strangle",
        expiry_cycle="weekly",
        entry_day_of_week=1,
        days_to_expiry_at_entry=2,
        otm_points_call=200,
        otm_points_put=200,
        stop_loss_pct=30,
        capital=100_000,
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
        capital=100_000,
        reentry="none",
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


# entry_date is a Tuesday in the pre-Nov-2024 (Thursday) weekly-expiry regime.
ENTRY_DATE = date(2022, 3, 1)
WEEKLY_EXPIRY = date(2022, 3, 3)  # next Thursday
MONTHLY_EXPIRY = date(2022, 3, 31)  # last Thursday of March 2022
SPOT = 24700.0


def make_full_chain():
    """A dense, evenly-spaced chain for both the weekly and monthly expiry of March 2022."""
    rows = []
    for strike in range(24000, 25400, 50):
        for expiry, base in ((WEEKLY_EXPIRY, 20.0), (MONTHLY_EXPIRY, 200.0)):
            rows.append(make_chain_row(ENTRY_DATE, expiry, strike, "CE", base + abs(SPOT - strike) * 0.01))
            rows.append(make_chain_row(ENTRY_DATE, expiry, strike, "PE", base + abs(SPOT - strike) * 0.01))
    return pd.DataFrame(rows, columns=COLUMNS)


class TestBuildPositionShortStrangle:
    def test_returns_two_legs(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert len(legs) == 2

    def test_strikes_are_otm_points_from_spot_rounded_to_50(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        call_leg = next(l for l in legs if l.option_type == "CE")
        put_leg = next(l for l in legs if l.option_type == "PE")
        assert call_leg.strike == 24900.0  # 24700 + 200
        assert put_leg.strike == 24500.0  # 24700 - 200

    def test_both_legs_are_sell_side(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert all(leg.side == "sell" for leg in legs)

    def test_entry_prices_populated_exit_prices_unset(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        for leg in legs:
            assert leg.entry_price > 0
            assert leg.exit_price is None

    def test_prices_come_from_weekly_expiry_not_monthly(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(expiry_cycle="weekly"), SPOT, chain=chain)
        # weekly rows use base=20.0, monthly rows use base=200.0 -- if the wrong
        # expiry were used, prices would be off by roughly that base amount.
        assert all(leg.entry_price < 50 for leg in legs)

    def test_prices_come_from_monthly_expiry_when_configured(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_strangle_config(expiry_cycle="monthly"), SPOT, chain=chain)
        assert all(leg.entry_price > 150 for leg in legs)

    def test_rounds_fractional_spot_to_nearest_strike_interval(self):
        chain = make_full_chain()
        # 24537.65 + 200 = 24737.65 -> rounds to 24750; 24537.65 - 200 = 24337.65 -> rounds to 24350
        legs = build_position(ENTRY_DATE, make_strangle_config(), 24537.65, chain=chain)
        call_leg = next(l for l in legs if l.option_type == "CE")
        put_leg = next(l for l in legs if l.option_type == "PE")
        assert call_leg.strike == 24750.0
        assert put_leg.strike == 24350.0


class TestBuildPositionIronCondor:
    def test_returns_four_legs(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_iron_condor_config(), SPOT, chain=chain)
        assert len(legs) == 4

    def test_short_legs_sell_long_legs_buy(self):
        chain = make_full_chain()
        legs = build_position(ENTRY_DATE, make_iron_condor_config(), SPOT, chain=chain)
        by_strike = {leg.strike: leg for leg in legs}
        assert by_strike[24900.0].side == "sell"  # short call
        assert by_strike[24500.0].side == "sell"  # short put
        assert by_strike[25050.0].side == "buy"  # long call (short + wing)
        assert by_strike[24350.0].side == "buy"  # long put (short - wing)

    def test_wing_strikes_offset_by_wing_width(self):
        chain = make_full_chain()
        config = make_iron_condor_config(wing_width_points=100)
        legs = build_position(ENTRY_DATE, config, SPOT, chain=chain)
        strikes_by_type_side = {(leg.option_type, leg.side): leg.strike for leg in legs}
        assert strikes_by_type_side[("CE", "buy")] == strikes_by_type_side[("CE", "sell")] + 100
        assert strikes_by_type_side[("PE", "buy")] == strikes_by_type_side[("PE", "sell")] - 100


class TestBuildPositionMissingStrikes:
    def test_falls_back_to_nearest_available_strike(self):
        # Chain stops at 24800, so the theoretical 24900 short call strike is missing.
        rows = [
            make_chain_row(ENTRY_DATE, WEEKLY_EXPIRY, strike, option_type, 20.0)
            for strike in range(24000, 24850, 50)
            for option_type in ("CE", "PE")
        ]
        chain = pd.DataFrame(rows, columns=COLUMNS)
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        call_leg = next(l for l in legs if l.option_type == "CE")
        assert call_leg.strike == 24800.0  # nearest available, not the theoretical 24900

    def test_missing_strike_logs_a_warning(self, caplog):
        rows = [
            make_chain_row(ENTRY_DATE, WEEKLY_EXPIRY, strike, option_type, 20.0)
            for strike in range(24000, 24850, 50)
            for option_type in ("CE", "PE")
        ]
        chain = pd.DataFrame(rows, columns=COLUMNS)
        with caplog.at_level(logging.WARNING, logger="engine.position_builder"):
            build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert any("not found in chain" in record.message for record in caplog.records)

    def test_does_not_crash_when_strike_missing(self):
        rows = [
            make_chain_row(ENTRY_DATE, WEEKLY_EXPIRY, strike, option_type, 20.0)
            for strike in range(24000, 24850, 50)
            for option_type in ("CE", "PE")
        ]
        chain = pd.DataFrame(rows, columns=COLUMNS)
        legs = build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
        assert len(legs) == 2


class TestBuildPositionMissingChainData:
    def test_raises_when_no_data_for_entry_date(self):
        chain = make_full_chain()
        other_day = date(2022, 3, 2)
        with pytest.raises(MissingOptionsChainDataError):
            build_position(other_day, make_strangle_config(), SPOT, chain=chain)

    def test_raises_when_option_type_entirely_absent_for_expiry(self):
        rows = [
            make_chain_row(ENTRY_DATE, WEEKLY_EXPIRY, strike, "CE", 20.0)
            for strike in range(24000, 25400, 50)
        ]  # no PE rows at all
        chain = pd.DataFrame(rows, columns=COLUMNS)
        with pytest.raises(MissingOptionsChainDataError):
            build_position(ENTRY_DATE, make_strangle_config(), SPOT, chain=chain)
