"""Source-agnostic interface for historical options chain data.

Downstream code (backtests, strategy logic, etc.) should depend on
`OptionsChainProvider` rather than a specific data source, so the source
can be swapped later (a paid vendor, a different exchange feed) without
touching anything past this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from .bhavcopy_loader import DEFAULT_START_DATE, BhavcopyLoader


class OptionsChainProvider(ABC):
    """Abstract interface for a historical options chain data source."""

    @abstractmethod
    def get_options_chain(
        self,
        start: date = DEFAULT_START_DATE,
        end: Optional[date] = None,
        symbol: str = "NIFTY",
    ) -> pd.DataFrame:
        """Return normalized options chain rows for [start, end] inclusive.

        Columns: date, expiry_date, strike, option_type, close, oi, volume.
        """
        raise NotImplementedError


class NSEBhavcopyProvider(OptionsChainProvider):
    """OptionsChainProvider backed by NSE daily F&O bhavcopy files."""

    def __init__(self, loader: Optional[BhavcopyLoader] = None):
        self._loader = loader or BhavcopyLoader()

    def get_options_chain(
        self,
        start: date = DEFAULT_START_DATE,
        end: Optional[date] = None,
        symbol: str = "NIFTY",
    ) -> pd.DataFrame:
        self._loader.symbol = symbol
        return self._loader.load(start=start, end=end)
