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
