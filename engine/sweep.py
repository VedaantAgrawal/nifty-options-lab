"""Parameter sweep: Cartesian product of a ParameterGrid, run in parallel
across a train/validation date split.

Reuses the exact reentry-loop logic scripts/validate_sample.py introduced
(`run_trade_cycle_loop` below is that same logic, extracted here as the
canonical implementation -- validate_sample.py now imports it rather than
keeping its own copy, so there's exactly one reentry scheduler, not two
that could quietly drift apart).
"""
from __future__ import annotations

import itertools
import logging
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from data.expiry_calendar import HolidayCalendar
from data.providers import NSEBhavcopyProvider, OptionsChainProvider
from engine.position_builder import _resolve_expiry
from engine.simulator import run_single_trade
from models.schemas import ParameterGrid, StrategyConfig, Trade

logger = logging.getLogger(__name__)

#: ParameterGrid field names, in the order StrategyConfig's constructor expects them.
_GRID_FIELD_NAMES = [
    "structure",
    "expiry_cycle",
    "entry_day_of_week",
    "days_to_expiry_at_entry",
    "otm_points_call",
    "otm_points_put",
    "wing_width_points",
    "stop_loss_pct",
    "capital",
    "reentry",
]


def generate_configs(grid: ParameterGrid) -> Tuple[List[StrategyConfig], int]:
    """Cartesian product of every field in `grid`, as StrategyConfig instances.

    Combos that fail StrategyConfig's own cross-field validation (e.g.
    wing_width_points set for a non-iron_condor structure) are skipped, not
    raised -- ParameterGrid's docstring already establishes this as the
    expected split of responsibility: the grid only validates each
    dimension's own values, and cross-field checks happen per generated
    combo, here.

    Returns (valid_configs, num_invalid_skipped).
    """
    value_lists = [getattr(grid, name) for name in _GRID_FIELD_NAMES]
    configs: List[StrategyConfig] = []
    num_invalid = 0
    for combo in itertools.product(*value_lists):
        kwargs = dict(zip(_GRID_FIELD_NAMES, combo))
        try:
            configs.append(StrategyConfig(**kwargs))
        except ValidationError:
            num_invalid += 1
    return configs, num_invalid


def _next_occurrence_of_weekday(after: date, weekday: int) -> date:
    d = after
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def _next_trading_day(after: date, holidays: HolidayCalendar) -> date:
    d = after
    while not holidays.is_trading_day(d):
        d += timedelta(days=1)
    return d


def _next_entry_after(after: date, config: StrategyConfig, holidays: HolidayCalendar) -> date:
    """Next scheduled cycle's entry date strictly after `after` (next occurrence
    of config.entry_day_of_week, shifted onto a real trading day)."""
    candidate = _next_occurrence_of_weekday(after + timedelta(days=1), config.entry_day_of_week)
    return _next_trading_day(candidate, holidays)


def _next_entry_after_exit(exit_date: date, config: StrategyConfig, holidays: HolidayCalendar) -> Optional[date]:
    """Where the next trade cycle should start, per config.reentry. None means stop."""
    if config.reentry == "none":
        return None
    if config.reentry == "immediate":
        return _next_trading_day(exit_date + timedelta(days=1), holidays)
    # "next_cycle"
    return _next_entry_after(exit_date, config, holidays)


def estimate_spot_price(day_chain: pd.DataFrame, expiry: date) -> Optional[float]:
    """Synthetic spot estimate via put-call parity (S ~= K + call_close - put_close),
    averaged over the 5 strikes closest to at-the-money.

    Our options chain data (data/bhavcopy_loader.py) carries no direct
    underlying/spot price column, so this is an estimate, not a recorded
    value. Good enough for OTM-points strike selection (which rounds to the
    nearest 50 anyway) -- don't rely on it for anything needing true spot
    precision.
    """
    subset = day_chain[day_chain["expiry_date"] == expiry]
    calls = subset[subset["option_type"] == "CE"].set_index("strike")["close"]
    puts = subset[subset["option_type"] == "PE"].set_index("strike")["close"]
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        return None
    diffs = (calls.loc[common_strikes] - puts.loc[common_strikes]).abs()
    nearest = diffs.nsmallest(min(5, len(diffs))).index
    implied_spots = [strike + calls[strike] - puts[strike] for strike in nearest]
    return float(sum(implied_spots) / len(implied_spots))


def run_trade_cycle_loop(
    config: StrategyConfig,
    start: date,
    end: date,
    chain: pd.DataFrame,
    holidays: Optional[HolidayCalendar] = None,
) -> Tuple[List[Trade], List[date]]:
    """Run the full entry -> exit -> reentry loop for ONE config over [start, end].

    Walks scheduled entry dates (per config.entry_day_of_week), opening a
    trade via run_single_trade at each one, and decides the next entry date
    from the previous trade's exit per config.reentry ("immediate" tries
    again the next trading day even mid-cycle; "next_cycle" waits for the
    next scheduled entry; "none" stops after the first trade). Stops
    without attempting a cycle whose expiry would fall after `end` (can't
    validate a still-open position with data that doesn't exist yet).

    Returns (closed_trades, skipped_entry_dates) -- skipped_entry_dates are
    the entry attempts where run_single_trade returned None (insufficient
    margin); len(skipped_entry_dates) is what compute_metrics wants as
    num_skipped_insufficient_margin.
    """
    holidays = holidays or HolidayCalendar.from_csv()
    trades: List[Trade] = []
    skipped_entry_dates: List[date] = []

    entry_date: Optional[date] = _next_entry_after(start - timedelta(days=1), config, holidays)

    while entry_date is not None and entry_date <= end:
        day_chain = chain[chain["date"] == entry_date]
        if day_chain.empty:
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        expiry = _resolve_expiry(entry_date, config, holidays=holidays)
        # `expiry <= end` alone isn't enough: `end` is a nominal date, but the
        # chain might not actually have a published row for it yet (e.g. a
        # same-day run before today's EOD bhavcopy is out). Without this,
        # run_single_trade's walk falls short of the real expiry but still
        # labels the trade exit_reason="expiry", misrepresenting an
        # incomplete cycle as a naturally closed one.
        if expiry > end or chain[chain["date"] == expiry].empty:
            break

        spot_price = estimate_spot_price(day_chain, expiry)
        if spot_price is None:
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        trade = run_single_trade(entry_date, config, spot_price, chain=chain)

        if trade is None:
            skipped_entry_dates.append(entry_date)
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        trades.append(trade)
        entry_date = _next_entry_after_exit(trade.exit_date, config, holidays)

    return trades, skipped_entry_dates
