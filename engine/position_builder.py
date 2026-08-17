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


def _nearest_available_strike(target: float, available: List[float]) -> float:
    """Nearest strike to `target`; ties break toward the lower strike (deterministic)."""
    return min(available, key=lambda strike: (abs(strike - target), strike))


def _lookup_premium(chain: pd.DataFrame, expiry_date: date, target_strike: float, option_type: str) -> tuple[float, float]:
    """Return (actual_strike_used, close_price) for the leg, substituting the nearest
    available strike (with a warning) if `target_strike` isn't in that day's chain."""
    subset = chain[(chain["expiry_date"] == expiry_date) & (chain["option_type"] == option_type)]
    if subset.empty:
        raise MissingOptionsChainDataError(
            f"No {option_type} options chain data for expiry {expiry_date}"
        )

    exact = subset[subset["strike"] == target_strike]
    if not exact.empty:
        return target_strike, float(exact.iloc[0]["close"])

    available_strikes = subset["strike"].unique().tolist()
    nearest = _nearest_available_strike(target_strike, available_strikes)
    logger.warning(
        "Strike %s %s not found in chain for expiry %s; using nearest available strike %s instead",
        target_strike, option_type, expiry_date, nearest,
    )
    price = float(subset[subset["strike"] == nearest].iloc[0]["close"])
    return nearest, price


#: (strikes-dict key, option_type, side) for each structure's legs.
_STRANGLE_LEGS = [
    ("short_call", "CE", "sell"),
    ("short_put", "PE", "sell"),
]
_IRON_CONDOR_EXTRA_LEGS = [
    ("long_call", "CE", "buy"),
    ("long_put", "PE", "buy"),
]


def build_position(
    entry_date: date,
    config: StrategyConfig,
    spot_price: float,
    provider: Optional[OptionsChainProvider] = None,
    chain: Optional[pd.DataFrame] = None,
) -> List[OptionLeg]:
    """Build the initial leg structure for `config`, priced off `spot_price` on `entry_date`.

    Returns 2 legs (short call + short put) for `short_strangle`, or 4 legs
    (adding long call + long put wings) for `iron_condor`. Each leg's
    `entry_price` is the actual chain close for its strike; `exit_price` is
    left unset (None) since no exit has happened yet.

    Pass `chain` directly to price off an already-fetched DataFrame (e.g. in
    tests) instead of fetching via `provider`. `chain` should already carry
    the normalized columns from `data/bhavcopy_loader.py`
    (date, expiry_date, strike, option_type, close, oi, volume).

    Every leg is sized at `lots=1` -- position sizing against `config.capital`
    is not part of this module and is left to whatever calls it.
    """
    expiry = _resolve_expiry(entry_date, config)

    if chain is None:
        provider = provider or NSEBhavcopyProvider()
        chain = provider.get_options_chain(start=entry_date, end=entry_date)
    day_chain = chain[chain["date"] == entry_date]
    if day_chain.empty:
        raise MissingOptionsChainDataError(f"No options chain data available for {entry_date}")

    target_strikes = _target_strikes(config, spot_price)
    leg_specs = list(_STRANGLE_LEGS)
    if config.structure == "iron_condor":
        leg_specs += _IRON_CONDOR_EXTRA_LEGS

    legs = []
    for strike_key, option_type, side in leg_specs:
        actual_strike, price = _lookup_premium(
            day_chain, expiry, target_strikes[strike_key], option_type
        )
        legs.append(
            OptionLeg(
                strike=actual_strike,
                option_type=option_type,
                side=side,
                entry_price=price,
                exit_price=None,
                lots=1,
            )
        )
    return legs
