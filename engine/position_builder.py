"""Build the initial leg structure (strikes + entry premiums) for a strategy.

`build_position` answers one question: given a `StrategyConfig` and the
NIFTY spot close on `entry_date`, which strikes trade, and what did they
cost? It does not compute P&L, decide exits, or size positions -- the
returned `OptionLeg`s have `entry_price` populated and `exit_price` left
unset (None), ready for something downstream to close out and wrap into a
`Trade`.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import List, Optional

import pandas as pd

from data.expiry_calendar import HolidayCalendar, next_monthly_expiry, next_weekly_expiry
from data.providers import NSEBhavcopyProvider, OptionsChainProvider
from models.schemas import STRIKE_INTERVAL, OptionLeg, StrategyConfig

logger = logging.getLogger(__name__)


class MissingOptionsChainDataError(RuntimeError):
    """Raised when there's no options chain data at all for the resolved (date, expiry, type)."""


def _round_to_strike_interval(value: float, interval: int = STRIKE_INTERVAL) -> float:
    """Round to the nearest strike interval, half-up (ties round away from zero)."""
    return math.floor(value / interval + 0.5) * interval


def _resolve_expiry(entry_date: date, config: StrategyConfig, holidays: Optional[HolidayCalendar] = None) -> date:
    """Resolve the weekly/monthly expiry (per `config.expiry_cycle`) applicable to `entry_date`."""
    holidays = holidays or HolidayCalendar.from_csv()
    if config.expiry_cycle == "weekly":
        return next_weekly_expiry(entry_date, holidays=holidays)
    return next_monthly_expiry(entry_date, holidays=holidays)


def _target_strikes(config: StrategyConfig, spot_price: float) -> dict:
    """Compute theoretical target strikes for `config`'s legs, before chain availability is checked."""
    short_call = _round_to_strike_interval(spot_price + config.otm_points_call)
    short_put = _round_to_strike_interval(spot_price - config.otm_points_put)
    strikes = {"short_call": short_call, "short_put": short_put}
    if config.structure == "iron_condor":
        assert config.wing_width_points is not None  # enforced by StrategyConfig validation
        strikes["long_call"] = short_call + config.wing_width_points
        strikes["long_put"] = short_put - config.wing_width_points
    return strikes
