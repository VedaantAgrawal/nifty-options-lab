"""Validation tests for models/schemas.py.

These are schema-level tests only: valid inputs construct successfully,
invalid inputs raise pydantic.ValidationError. No engine/backtest logic
is exercised here.
"""
from datetime import date

import pytest
from pydantic import ValidationError

from models.schemas import OptionLeg, Trade


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
