"""Tests for data.lot_size_calendar.

Boundary dates here are the NSE-circular-verified transition dates
documented in the module docstring (see commit history / PR discussion for
sources), not assumptions.
"""
from datetime import date

import pytest

from data.lot_size_calendar import UnsupportedLotSizeDateError, lot_size_for_date


class TestLotSizeForDate:
    def test_pre_aug_2021_is_75(self):
        assert lot_size_for_date(date(2021, 1, 4)) == 75
        assert lot_size_for_date(date(2021, 7, 31)) == 75

    def test_aug_2021_cutover_is_50(self):
        assert lot_size_for_date(date(2021, 8, 1)) == 50

    def test_mid_regime_2022_is_50(self):
        assert lot_size_for_date(date(2022, 3, 1)) == 50

    def test_day_before_april_2024_cutover_is_still_50(self):
        assert lot_size_for_date(date(2024, 4, 25)) == 50

    def test_april_2024_cutover_is_25(self):
        assert lot_size_for_date(date(2024, 4, 26)) == 25

    def test_day_before_nov_2024_cutover_is_still_25(self):
        assert lot_size_for_date(date(2024, 11, 19)) == 25

    def test_nov_2024_cutover_is_75(self):
        assert lot_size_for_date(date(2024, 11, 20)) == 75

    def test_day_before_oct_2025_cutover_is_still_75(self):
        assert lot_size_for_date(date(2025, 10, 27)) == 75

    def test_oct_2025_cutover_is_65(self):
        assert lot_size_for_date(date(2025, 10, 28)) == 65

    def test_after_oct_2025_cutover_is_65(self):
        assert lot_size_for_date(date(2026, 1, 1)) == 65

    def test_date_before_earliest_regime_raises(self):
        with pytest.raises(UnsupportedLotSizeDateError):
            lot_size_for_date(date(2018, 1, 1))
