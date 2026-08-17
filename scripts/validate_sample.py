"""Validation run: a real 3-month backtest for one fixed StrategyConfig,
dumped as a plain, eyeball-able CSV trade log.

Not the final product -- no aggregation, no metrics, no sweep. Loads real
(cached-or-downloaded) NIFTY options data, runs the reentry loop, writes
scripts/output/validate_sample_trades.csv, and prints trade count plus any
margin-skip / missing-strike warnings so data-quality issues are visible
immediately rather than silently averaged away.

Run from the repo root: python scripts/validate_sample.py
"""
from __future__ import annotations

import csv
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.expiry_calendar import HolidayCalendar  # noqa: E402
from data.providers import NSEBhavcopyProvider  # noqa: E402
from engine.position_builder import _resolve_expiry  # noqa: E402
from engine.simulator import run_single_trade  # noqa: E402
from models.schemas import StrategyConfig, Trade  # noqa: E402

OUTPUT_CSV = Path(__file__).resolve().parent / "output" / "validate_sample_trades.csv"

# --- Fixed config for this validation run ----------------------------------
# entry_day_of_week=2 (Wednesday) assumes entering right after the prior
# week's expiry, for the current Tuesday weekly-expiry regime;
# days_to_expiry_at_entry=4 is the matching descriptive value (4 trading
# days before a Tuesday expiry: Wed, Thu, Fri, Mon, Tue). reentry="next_cycle"
# is the "normal" systematic weekly-seller cadence: don't re-enter mid-cycle
# after a stop-loss, wait for the next scheduled cycle. None of this was
# specified by the task beyond structure/expiry_cycle/OTM/stop_loss/capital
# -- adjust here if you want a different cadence for this validation run.
#
# capital=1,000,000: the user's actual available capital. Note that
# capital=100,000 (the number originally given in the task prompt) skips
# every single cycle in this window -- at current NIFTY levels (~24,700)
# with lot_size=65, the placeholder 15%-of-notional margin formula in
# engine/simulator.py computes ~Rs 4.6-4.8 lakh required per short strangle
# lot, so 100,000 capital can never open one.
CONFIG = StrategyConfig(
    structure="short_strangle",
    expiry_cycle="weekly",
    entry_day_of_week=2,
    days_to_expiry_at_entry=4,
    otm_points_call=500,
    otm_points_put=500,
    stop_loss_pct=100,
    capital=1_000_000,
    reentry="next_cycle",
)

CSV_FIELDS = [
    "entry_date",
    "expiry_date",
    "exit_date",
    "exit_reason",
    "call_strike",
    "call_entry_premium",
    "call_exit_premium",
    "put_strike",
    "put_entry_premium",
    "put_exit_premium",
    "pnl",
    "capital_at_risk",
]


class _CollectingHandler(logging.Handler):
    """Captures WARNING+ log records from specific loggers for the end-of-run summary."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


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


def _estimate_spot_price(day_chain: pd.DataFrame, expiry: date) -> Optional[float]:
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


def _trade_to_row(trade: Trade) -> Dict:
    call = next(leg for leg in trade.legs if leg.option_type == "CE")
    put = next(leg for leg in trade.legs if leg.option_type == "PE")
    return {
        "entry_date": trade.entry_date,
        "expiry_date": trade.expiry_date,
        "exit_date": trade.exit_date,
        "exit_reason": trade.exit_reason,
        "call_strike": call.strike,
        "call_entry_premium": call.entry_price,
        "call_exit_premium": call.exit_price,
        "put_strike": put.strike,
        "put_entry_premium": put.entry_price,
        "put_exit_premium": put.exit_price,
        "pnl": round(trade.pnl, 2),
        "capital_at_risk": round(trade.capital_at_risk, 2),
    }


def main() -> None:
    end = date.today()
    start = end - timedelta(days=90)

    warning_handler = _CollectingHandler()
    warning_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger("engine.position_builder").addHandler(warning_handler)
    logging.getLogger("engine.simulator").addHandler(warning_handler)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading NIFTY options chain data for {start} to {end} (cached-or-downloaded)...")
    provider = NSEBhavcopyProvider()
    chain = provider.get_options_chain(start=start, end=end, symbol="NIFTY")
    if chain.empty:
        print("No chain data available for this window -- nothing to validate.")
        return
    print(f"Loaded {len(chain)} rows across {chain['date'].nunique()} trading days.")

    holidays = HolidayCalendar.from_csv()

    trades: List[Trade] = []
    margin_skips: List[Dict] = []

    entry_date: Optional[date] = _next_entry_after(start - timedelta(days=1), CONFIG, holidays)

    while entry_date is not None and entry_date <= end:
        day_chain = chain[chain["date"] == entry_date]
        if day_chain.empty:
            print(f"  {entry_date}: no chain data (holiday or gap) -- skipping to next cycle")
            entry_date = _next_entry_after(entry_date, CONFIG, holidays)
            continue

        expiry = _resolve_expiry(entry_date, CONFIG, holidays=holidays)
        if expiry > end:
            print(f"  {entry_date}: cycle expiry {expiry} is beyond today ({end}) -- stopping (incomplete cycle)")
            break

        spot_price = _estimate_spot_price(day_chain, expiry)
        if spot_price is None:
            print(f"  {entry_date}: could not estimate spot price (no CE/PE overlap) -- skipping to next cycle")
            entry_date = _next_entry_after(entry_date, CONFIG, holidays)
            continue

        trade = run_single_trade(entry_date, CONFIG, spot_price, chain=chain)

        if trade is None:
            print(f"  {entry_date}: SKIPPED (insufficient margin) -- see warning above")
            margin_skips.append({"entry_date": entry_date})
            entry_date = _next_entry_after(entry_date, CONFIG, holidays)
            continue

        trades.append(trade)
        print(
            f"  {entry_date} -> {trade.exit_date} ({trade.exit_reason}): "
            f"pnl={trade.pnl:.2f}"
        )
        entry_date = _next_entry_after_exit(trade.exit_date, CONFIG, holidays)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for trade in trades:
            writer.writerow(_trade_to_row(trade))

    missing_strike_warnings = [r for r in warning_handler.records if "not found in chain" in r]
    margin_skip_log_lines = [r for r in warning_handler.records if "skipped: required margin" in r]
    other_warnings = [
        r for r in warning_handler.records
        if r not in missing_strike_warnings and r not in margin_skip_log_lines
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Window: {start} to {end}")
    print(f"Config: {CONFIG.structure}, {CONFIG.expiry_cycle}, "
          f"OTM call/put={CONFIG.otm_points_call}/{CONFIG.otm_points_put}, "
          f"stop_loss_pct={CONFIG.stop_loss_pct}, capital={CONFIG.capital}, "
          f"reentry={CONFIG.reentry}")
    print(f"Trades closed: {len(trades)}")
    print(f"Margin-skip events: {len(margin_skips)}")
    for skip in margin_skips:
        print(f"  - {skip['entry_date']}")
    print(f"Missing-strike warnings (nearest-strike substitution): {len(missing_strike_warnings)}")
    for w in missing_strike_warnings:
        print(f"  - {w}")
    if other_warnings:
        print(f"Other warnings: {len(other_warnings)}")
        for w in other_warnings:
            print(f"  - {w}")
    print(f"\nCSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
