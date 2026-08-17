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
