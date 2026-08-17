"""Pydantic schemas for the NIFTY options backtesting engine.

These mirror the normalized schema produced by `data/bhavcopy_loader.py`
(`option_type` is "CE"/"PE", strikes are NIFTY strike-interval multiples)
so rows from that layer can be turned directly into `OptionLeg` instances.
"""
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

#: NIFTY strikes are quoted in multiples of this many points.
STRIKE_INTERVAL = 50


def _is_multiple_of(value: float, step: int) -> bool:
    """True if `value` is a (positive-direction) multiple of `step`, tolerant of float error."""
    ratio = value / step
    return abs(ratio - round(ratio)) < 1e-6


class OptionLeg(BaseModel):
    """A single option leg (one strike/type/side) within a multi-leg trade."""

    strike: float = Field(description="Strike price, must be a positive multiple of 50")
    option_type: Literal["CE", "PE"]
    side: Literal["buy", "sell"]
    entry_price: float = Field(gt=0, description="Premium paid/received at entry")
    exit_price: float = Field(ge=0, description="Premium paid/received at exit (0 if expired worthless)")
    lots: int = Field(gt=0, description="Number of lots traded for this leg")

    @field_validator("strike")
    @classmethod
    def _strike_is_positive_multiple_of_interval(cls, v: float) -> float:
        if v <= 0 or not _is_multiple_of(v, STRIKE_INTERVAL):
            raise ValueError(f"strike must be a positive multiple of {STRIKE_INTERVAL}, got {v}")
        return v
