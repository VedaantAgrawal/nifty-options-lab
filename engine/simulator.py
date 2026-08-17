"""Core single-trade backtest loop: open a position, walk it forward day by
day to a stop-loss or expiry, and return the closed Trade.

No sweep/portfolio logic here -- this module is exactly one entry -> one
exit. `engine.position_builder.build_position` handles strike selection and
entry pricing; this module adds the day-by-day mark-to-market walk, the
stop-loss decision, the margin/capital gate, and final P&L.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from data.providers import NSEBhavcopyProvider, OptionsChainProvider
from engine.position_builder import MissingOptionsChainDataError, _resolve_expiry, build_position
from models.schemas import OptionLeg, StrategyConfig, Trade

logger = logging.getLogger(__name__)

#: Fraction of short-leg notional charged as margin for undefined-risk structures
#: (short_strangle). Purely a placeholder -- not a real SPAN/exposure margin model.
DEFAULT_MARGIN_PCT = 0.15

#: Placeholder NIFTY lot size (contracts per lot). NSE has revised this multiple
#: times historically -- verify the correct value for the specific backtest
#: period before trusting margin/PnL figures derived from it.
DEFAULT_LOT_SIZE = 75


def _short_legs(legs: List[OptionLeg]) -> List[OptionLeg]:
    return [leg for leg in legs if leg.side == "sell"]


def estimate_margin_required(
    config: StrategyConfig,
    legs: List[OptionLeg],
    lot_size: int = DEFAULT_LOT_SIZE,
    margin_pct: float = DEFAULT_MARGIN_PCT,
) -> float:
    """Rough placeholder margin estimate, evaluated at entry (before any price move).

    short_strangle (undefined risk): margin_pct of the short legs' strike notional.

    iron_condor (defined risk): approximated directly as
    max(wing_width_points, points_moved) x lot_size. Evaluated at entry,
    points_moved is 0, so this reduces to wing_width_points x lot_size --
    i.e. the wing width itself, not a percentage of notional, since a
    defined-risk spread's max loss is capped by its width.
    """
    if config.structure == "iron_condor":
        assert config.wing_width_points is not None  # enforced by StrategyConfig validation
        points_moved_at_entry = 0
        return max(config.wing_width_points, points_moved_at_entry) * lot_size

    short_notional = sum(leg.strike * lot_size * leg.lots for leg in _short_legs(legs))
    return short_notional * margin_pct


def _trading_days_in_range(chain: pd.DataFrame, expiry: date, start: date, end: date) -> List[date]:
    subset = chain[(chain["expiry_date"] == expiry) & (chain["date"] >= start) & (chain["date"] <= end)]
    return sorted(subset["date"].unique())


def _leg_price_on_day(day_chain: pd.DataFrame, expiry: date, leg: OptionLeg) -> Optional[float]:
    rows = day_chain[
        (day_chain["expiry_date"] == expiry)
        & (day_chain["option_type"] == leg.option_type)
        & (day_chain["strike"] == leg.strike)
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["close"])


def _mark_to_market(
    chain: pd.DataFrame,
    expiry: date,
    legs: List[OptionLeg],
    day: date,
    last_known_prices: Dict[int, float],
) -> Dict[int, float]:
    """Return {id(leg): price} for `day`, carrying forward the last known price
    (with a warning) for any leg whose exact strike didn't trade that day."""
    day_chain = chain[chain["date"] == day]
    prices = {}
    for leg in legs:
        price = _leg_price_on_day(day_chain, expiry, leg)
        if price is None:
            price = last_known_prices[id(leg)]
            logger.warning(
                "No chain data for strike %s %s on %s (expiry %s); carrying forward last known price %.2f",
                leg.strike, leg.option_type, day, expiry, price,
            )
        prices[id(leg)] = price
    return prices


def _walk_to_exit(
    chain: pd.DataFrame,
    expiry: date,
    legs: List[OptionLeg],
    entry_date: date,
    stop_loss_pct: float,
) -> tuple[date, str, Dict[int, float]]:
    """Walk trading days from entry_date to expiry, returning (exit_date, exit_reason, exit_prices).

    exit_prices maps id(leg) -> close price on the exit day. Stops early with
    exit_reason='stop_loss' the first day the short legs' combined premium is
    >= stop_loss_pct percent above what was collected at entry; otherwise
    exits at expiry using that day's close prices (the day's official
    close/settlement row -- no separate early-assignment logic is needed
    since NIFTY index options are European-style and cash-settled).
    """
    trading_days = _trading_days_in_range(chain, expiry, entry_date, expiry)
    if not trading_days:
        raise MissingOptionsChainDataError(
            f"No options chain data between {entry_date} and {expiry} for expiry {expiry}"
        )

    entry_short_premium = sum(leg.entry_price for leg in _short_legs(legs))
    stop_loss_threshold = entry_short_premium * (1 + stop_loss_pct / 100)

    last_known_prices = {id(leg): leg.entry_price for leg in legs}
    day_prices = last_known_prices
    for day in trading_days:
        day_prices = _mark_to_market(chain, expiry, legs, day, last_known_prices)
        last_known_prices = day_prices

        short_premium_today = sum(day_prices[id(leg)] for leg in _short_legs(legs))
        if short_premium_today >= stop_loss_threshold:
            return day, "stop_loss", day_prices

    # Loop ran to completion without triggering a stop-loss: exit at expiry
    # using the final trading day's prices (the last one visited above).
    return trading_days[-1], "expiry", day_prices


def _leg_pnl_per_unit(leg: OptionLeg, exit_price: float) -> float:
    if leg.side == "sell":
        return leg.entry_price - exit_price
    return exit_price - leg.entry_price


def run_single_trade(
    entry_date: date,
    config: StrategyConfig,
    spot_price: float,
    provider: Optional[OptionsChainProvider] = None,
    chain: Optional[pd.DataFrame] = None,
    lot_size: int = DEFAULT_LOT_SIZE,
    margin_pct: float = DEFAULT_MARGIN_PCT,
) -> Optional[Trade]:
    """Run one full trade cycle: open at `entry_date`, walk to stop-loss or expiry.

    Returns None (after logging the reason) if the estimated required margin
    exceeds `config.capital` -- no position is opened in that case, and no
    Trade is constructed. Otherwise returns a fully closed Trade.

    Pass `chain` directly to price off an already-fetched DataFrame (e.g. in
    tests) instead of fetching via `provider`.
    """
    expiry = _resolve_expiry(entry_date, config)

    if chain is None:
        provider = provider or NSEBhavcopyProvider()
        chain = provider.get_options_chain(start=entry_date, end=expiry)

    legs = build_position(entry_date, config, spot_price, chain=chain)

    margin_required = estimate_margin_required(config, legs, lot_size=lot_size, margin_pct=margin_pct)
    if margin_required > config.capital:
        logger.warning(
            "skipped: required margin ₹%.0f exceeds capital ₹%.0f (entry_date=%s, structure=%s)",
            margin_required, config.capital, entry_date, config.structure,
        )
        return None

    exit_date, exit_reason, exit_prices = _walk_to_exit(
        chain, expiry, legs, entry_date, config.stop_loss_pct
    )

    closed_legs = []
    total_pnl = 0.0
    for leg in legs:
        exit_price = exit_prices[id(leg)]
        total_pnl += _leg_pnl_per_unit(leg, exit_price) * lot_size * leg.lots
        closed_legs.append(leg.model_copy(update={"exit_price": exit_price}))

    return Trade(
        entry_date=entry_date,
        expiry_date=expiry,
        exit_date=exit_date,
        legs=closed_legs,
        exit_reason=exit_reason,
        pnl=total_pnl,
        capital_at_risk=margin_required,
    )
