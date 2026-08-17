"""Validation tests for models/schemas.py.

These are schema-level tests only: valid inputs construct successfully,
invalid inputs raise pydantic.ValidationError. No engine/backtest logic
is exercised here.
"""
from datetime import date

import pytest
from pydantic import ValidationError

from models.schemas import OptionLeg, ParameterGrid, StrategyConfig, SweepResult, Trade


def make_leg(**overrides):
    defaults = dict(
        strike=24500,
        option_type="CE",
        side="sell",
        entry_price=150.0,
        exit_price=50.0,
        lots=1,
    )
    defaults.update(overrides)
    return OptionLeg(**defaults)


class TestOptionLeg:
    def test_valid_leg_constructs(self):
        leg = make_leg()
        assert leg.strike == 24500
        assert leg.option_type == "CE"

    def test_strike_not_multiple_of_50_raises(self):
        with pytest.raises(ValidationError):
            make_leg(strike=24510)

    def test_strike_zero_raises(self):
        with pytest.raises(ValidationError):
            make_leg(strike=0)

    def test_strike_negative_raises(self):
        with pytest.raises(ValidationError):
            make_leg(strike=-100)

    def test_entry_price_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_leg(entry_price=0)

    def test_exit_price_can_be_zero_expired_worthless(self):
        leg = make_leg(exit_price=0)
        assert leg.exit_price == 0

    def test_lots_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_leg(lots=0)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValidationError):
            make_leg(option_type="XX")

    def test_invalid_side_raises(self):
        with pytest.raises(ValidationError):
            make_leg(side="hold")

    def test_exit_price_defaults_to_none_when_omitted(self):
        leg = OptionLeg(strike=24500, option_type="CE", side="sell", entry_price=150.0, lots=1)
        assert leg.exit_price is None

    def test_exit_price_none_is_explicitly_allowed(self):
        leg = make_leg(exit_price=None)
        assert leg.exit_price is None

    def test_exit_price_negative_raises(self):
        with pytest.raises(ValidationError):
            make_leg(exit_price=-10)


def make_trade(**overrides):
    defaults = dict(
        entry_date=date(2024, 1, 2),
        expiry_date=date(2024, 1, 4),
        exit_date=date(2024, 1, 4),
        legs=[make_leg()],
        exit_reason="expiry",
        pnl=1500.0,
        capital_at_risk=50000.0,
    )
    defaults.update(overrides)
    return Trade(**defaults)


class TestTrade:
    def test_valid_trade_constructs(self):
        trade = make_trade()
        assert len(trade.legs) == 1

    def test_requires_at_least_one_leg(self):
        with pytest.raises(ValidationError):
            make_trade(legs=[])

    def test_exit_before_entry_raises(self):
        with pytest.raises(ValidationError):
            make_trade(entry_date=date(2024, 1, 4), exit_date=date(2024, 1, 2))

    def test_exit_after_expiry_raises(self):
        with pytest.raises(ValidationError):
            make_trade(
                entry_date=date(2024, 1, 2),
                expiry_date=date(2024, 1, 4),
                exit_date=date(2024, 1, 5),
            )

    def test_entry_after_expiry_raises(self):
        with pytest.raises(ValidationError):
            make_trade(entry_date=date(2024, 1, 10), expiry_date=date(2024, 1, 4), exit_date=date(2024, 1, 4))

    def test_exit_reason_expiry_requires_exit_date_equals_expiry_date(self):
        with pytest.raises(ValidationError):
            make_trade(
                entry_date=date(2024, 1, 2),
                expiry_date=date(2024, 1, 4),
                exit_date=date(2024, 1, 3),
                exit_reason="expiry",
            )

    def test_exit_reason_stop_loss_allows_exit_before_expiry(self):
        trade = make_trade(
            entry_date=date(2024, 1, 2),
            expiry_date=date(2024, 1, 4),
            exit_date=date(2024, 1, 3),
            exit_reason="stop_loss",
        )
        assert trade.exit_reason == "stop_loss"

    def test_capital_at_risk_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_trade(capital_at_risk=0)

    def test_leg_with_unset_exit_price_raises(self):
        open_leg = make_leg(exit_price=None)
        with pytest.raises(ValidationError):
            make_trade(legs=[open_leg])


def make_config(**overrides):
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


class TestStrategyConfig:
    def test_valid_short_strangle_constructs_with_no_wing(self):
        config = make_config()
        assert config.wing_width_points is None

    def test_valid_iron_condor_requires_wing_width(self):
        config = make_config(structure="iron_condor", wing_width_points=100)
        assert config.wing_width_points == 100

    def test_iron_condor_without_wing_width_raises(self):
        with pytest.raises(ValidationError):
            make_config(structure="iron_condor")

    def test_short_strangle_with_wing_width_raises(self):
        with pytest.raises(ValidationError):
            make_config(structure="short_strangle", wing_width_points=100)

    def test_wing_width_not_multiple_of_50_raises(self):
        with pytest.raises(ValidationError):
            make_config(structure="iron_condor", wing_width_points=75)

    def test_otm_points_not_multiple_of_50_raises(self):
        with pytest.raises(ValidationError):
            make_config(otm_points_call=175)

    def test_otm_points_negative_raises(self):
        with pytest.raises(ValidationError):
            make_config(otm_points_put=-100)

    def test_stop_loss_pct_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_config(stop_loss_pct=0)

    def test_stop_loss_pct_negative_raises(self):
        with pytest.raises(ValidationError):
            make_config(stop_loss_pct=-10)

    def test_capital_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_config(capital=0)

    def test_entry_day_of_week_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            make_config(entry_day_of_week=7)

    def test_days_to_expiry_negative_raises(self):
        with pytest.raises(ValidationError):
            make_config(days_to_expiry_at_entry=-1)


def make_grid(**overrides):
    defaults = dict(
        structure=["short_strangle", "iron_condor"],
        expiry_cycle=["weekly"],
        entry_day_of_week=[1, 3],
        days_to_expiry_at_entry=[2, 5],
        otm_points_call=[150, 200],
        otm_points_put=[150, 200],
        wing_width_points=[None, 100],
        stop_loss_pct=[25, 30],
        capital=[100_000],
        reentry=["none", "immediate"],
    )
    defaults.update(overrides)
    return ParameterGrid(**defaults)


class TestParameterGrid:
    def test_valid_grid_constructs(self):
        grid = make_grid()
        assert grid.structure == ["short_strangle", "iron_condor"]

    def test_wing_width_defaults_to_none_only_when_omitted(self):
        # default_factory should kick in when the field is omitted entirely
        grid = ParameterGrid(
            structure=["short_strangle"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[1],
            days_to_expiry_at_entry=[2],
            otm_points_call=[150],
            otm_points_put=[150],
            stop_loss_pct=[25],
            capital=[100_000],
            reentry=["none"],
        )
        assert grid.wing_width_points == [None]

    def test_otm_points_list_with_bad_multiple_raises(self):
        with pytest.raises(ValidationError):
            make_grid(otm_points_call=[150, 175])

    def test_wing_width_list_with_bad_multiple_raises(self):
        with pytest.raises(ValidationError):
            make_grid(wing_width_points=[None, 90])

    def test_stop_loss_list_with_non_positive_raises(self):
        with pytest.raises(ValidationError):
            make_grid(stop_loss_pct=[25, 0])

    def test_capital_list_with_non_positive_raises(self):
        with pytest.raises(ValidationError):
            make_grid(capital=[100_000, -5])

    def test_entry_day_of_week_list_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            make_grid(entry_day_of_week=[1, 9])

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            make_grid(structure=[])


def make_sweep_result(**overrides):
    defaults = dict(
        config=make_config(),
        total_return=0.15,
        win_rate=0.6,
        max_drawdown=-0.08,
        sharpe=1.2,
        avg_pnl_per_trade=1500.0,
        num_trades=40,
    )
    defaults.update(overrides)
    return SweepResult(**defaults)


class TestSweepResult:
    def test_valid_sweep_result_constructs(self):
        result = make_sweep_result()
        assert result.config.structure == "short_strangle"
        assert result.num_trades == 40

    def test_win_rate_above_one_raises(self):
        with pytest.raises(ValidationError):
            make_sweep_result(win_rate=1.5)

    def test_win_rate_below_zero_raises(self):
        with pytest.raises(ValidationError):
            make_sweep_result(win_rate=-0.1)

    def test_win_rate_boundary_values_are_valid(self):
        assert make_sweep_result(win_rate=0.0).win_rate == 0.0
        assert make_sweep_result(win_rate=1.0).win_rate == 1.0

    def test_positive_max_drawdown_raises(self):
        with pytest.raises(ValidationError):
            make_sweep_result(max_drawdown=0.05)

    def test_zero_max_drawdown_is_valid(self):
        assert make_sweep_result(max_drawdown=0.0).max_drawdown == 0.0

    def test_num_trades_negative_raises(self):
        with pytest.raises(ValidationError):
            make_sweep_result(num_trades=-1)

    def test_config_must_be_a_valid_strategy_config(self):
        with pytest.raises(ValidationError):
            make_sweep_result(config={"structure": "iron_condor"})  # missing required fields
