"""Tests for data.expiry_calendar.

Two kinds of tests live here:

1. Logic/self-consistency tests (this file, filled in) that check the
   regime table, weekday resolution, holiday shifting, and the
   regime-transition edge case behave correctly relative to each other --
   these don't require an external source of truth.
2. Known-good NSE expiry dates (see `KNOWN_EXPIRIES` below), one per
   regime, verified against real NSE data. These are placeholders until
   real dates are supplied -- see the TODO on that list.
"""
from datetime import date

import pytest

from data.expiry_calendar import (
    THURSDAY,
    TUESDAY,
    HolidayCalendar,
    UnsupportedDateError,
    next_monthly_expiry,
    next_weekly_expiry,
    weekday_for_date,
)


class TestWeekdayForDate:
    def test_pre_consolidation_regime_is_thursday(self):
        assert weekday_for_date(date(2021, 6, 15)) == THURSDAY

    def test_post_consolidation_regime_is_still_thursday(self):
        assert weekday_for_date(date(2024, 12, 5)) == THURSDAY

    def test_tuesday_regime_after_switch(self):
        assert weekday_for_date(date(2025, 9, 2)) == TUESDAY
        assert weekday_for_date(date(2025, 12, 25)) == TUESDAY

    def test_day_before_switch_is_still_thursday_regime(self):
        assert weekday_for_date(date(2025, 9, 1)) == THURSDAY

    def test_date_before_earliest_regime_raises(self):
        with pytest.raises(UnsupportedDateError):
            weekday_for_date(date(2018, 1, 1))


class TestNextWeeklyExpiry:
    def test_from_tuesday_finds_same_week_thursday(self):
        assert next_weekly_expiry(date(2022, 3, 1)) == date(2022, 3, 3)

    def test_inclusive_on_expiry_day_returns_itself(self):
        assert next_weekly_expiry(date(2022, 3, 3), inclusive=True) == date(2022, 3, 3)

    def test_exclusive_on_expiry_day_skips_to_next_week(self):
        assert next_weekly_expiry(date(2022, 3, 3), inclusive=False) == date(2022, 3, 10)

    def test_from_wednesday_after_switch_finds_next_tuesday(self):
        assert next_weekly_expiry(date(2025, 9, 3)) == date(2025, 9, 9)

    def test_regime_transition_edge_case(self):
        # Aug 29 2025 is the last day of the Thursday regime. The naive next
        # Thursday (Sep 4) has already rolled into the Tuesday regime, so the
        # real next weekly expiry is Sep 2 (Tuesday), not Sep 4.
        assert next_weekly_expiry(date(2025, 8, 29)) == date(2025, 9, 2)

    def test_holiday_on_expiry_day_shifts_to_previous_trading_day(self):
        holidays = HolidayCalendar([date(2022, 3, 3)])
        assert next_weekly_expiry(date(2022, 2, 28), holidays=holidays) == date(2022, 3, 2)

    def test_holiday_shift_that_would_precede_from_date_rolls_to_next_cycle(self):
        # If from_date IS the (holiday) expiry day itself, shifting backward
        # would land before from_date -- we should roll forward to the next
        # cycle's expiry instead of returning a date earlier than from_date.
        holidays = HolidayCalendar([date(2022, 3, 3)])
        result = next_weekly_expiry(date(2022, 3, 3), holidays=holidays, inclusive=True)
        assert result >= date(2022, 3, 3)
        assert result == date(2022, 3, 10)


class TestNextMonthlyExpiry:
    def test_last_thursday_of_month_pre_switch(self):
        assert next_monthly_expiry(date(2022, 3, 1)) == date(2022, 3, 31)

    def test_last_thursday_of_month_that_stays_within_thursday_regime(self):
        assert next_monthly_expiry(date(2025, 8, 1)) == date(2025, 8, 28)

    def test_last_tuesday_of_month_after_switch(self):
        assert next_monthly_expiry(date(2025, 9, 1)) == date(2025, 9, 30)

    def test_holiday_on_monthly_expiry_shifts_back(self):
        holidays = HolidayCalendar([date(2022, 3, 31)])
        assert next_monthly_expiry(date(2022, 3, 1), holidays=holidays) == date(2022, 3, 30)


class TestHolidayCalendar:
    def test_weekend_is_always_non_trading(self):
        hc = HolidayCalendar()
        saturday = date(2024, 1, 6)
        assert hc.is_trading_day(saturday) is False

    def test_previous_trading_day_skips_consecutive_holidays_and_weekend(self):
        # Thursday + Friday both holidays, so previous trading day from
        # Thursday should skip back over the weekend to the prior Wednesday.
        thursday = date(2024, 1, 4)
        friday = date(2024, 1, 5)
        hc = HolidayCalendar([thursday, friday])
        assert hc.previous_trading_day(thursday) == date(2024, 1, 3)


# TODO: replace/extend with real NSE-verified expiry dates, at least one per
# regime, as supplied by the user. Each tuple is
# (query_date, expected_weekly_expiry, expected_monthly_expiry, regime_label).
KNOWN_EXPIRIES = [
    # pytest.param(date(2021, 1, 4), date(2021, 1, 7), date(2021, 1, 28), id="pre-consolidation"),
    # pytest.param(date(2024, 12, 2), date(2024, 12, 5), date(2024, 12, 26), id="post-consolidation-thursday"),
    # pytest.param(date(2025, 9, 8), date(2025, 9, 9), date(2025, 9, 30), id="tuesday-regime"),
]


@pytest.mark.skipif(not KNOWN_EXPIRIES, reason="Waiting on real NSE-verified dates to fill in KNOWN_EXPIRIES")
@pytest.mark.parametrize("query_date, expected_weekly, expected_monthly, regime_label", KNOWN_EXPIRIES)
def test_known_expiry_matches_real_nse_data(query_date, expected_weekly, expected_monthly, regime_label):
    holidays = HolidayCalendar.from_csv()
    assert next_weekly_expiry(query_date, holidays=holidays) == expected_weekly
    assert next_monthly_expiry(query_date, holidays=holidays) == expected_monthly
