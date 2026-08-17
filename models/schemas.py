"""Pydantic schemas for the NIFTY options backtesting engine.

These mirror the normalized schema produced by `data/bhavcopy_loader.py`
(`option_type` is "CE"/"PE", strikes are NIFTY strike-interval multiples)
so rows from that layer can be turned directly into `OptionLeg` instances.
"""
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

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


class Trade(BaseModel):
    """A complete multi-leg trade, from entry to exit."""

    entry_date: date
    expiry_date: date
    exit_date: date
    legs: List[OptionLeg] = Field(min_length=1, description="At least one leg")
    exit_reason: Literal["stop_loss", "expiry"]
    pnl: float = Field(description="Realized P&L for the trade, in currency units")
    capital_at_risk: float = Field(gt=0, description="Capital allocated/blocked for this trade")

    @model_validator(mode="after")
    def _dates_are_consistent(self) -> "Trade":
        if self.entry_date > self.expiry_date:
            raise ValueError("entry_date must be on or before expiry_date")
        if self.exit_date < self.entry_date:
            raise ValueError("exit_date must be on or after entry_date")
        if self.exit_date > self.expiry_date:
            raise ValueError("exit_date must be on or before expiry_date")
        if self.exit_reason == "expiry" and self.exit_date != self.expiry_date:
            raise ValueError("exit_reason='expiry' requires exit_date == expiry_date")
        return self


class StrategyConfig(BaseModel):
    """Parameters for one options-selling strategy configuration.

    `entry_day_of_week` follows `datetime.date.weekday()` convention
    (Monday=0 ... Sunday=6), matching `data/expiry_calendar.py`.
    """

    structure: Literal["short_strangle", "iron_condor"]
    expiry_cycle: Literal["weekly", "monthly"]
    entry_day_of_week: int = Field(ge=0, le=6, description="Monday=0 ... Sunday=6")
    days_to_expiry_at_entry: int = Field(ge=0, description="Days before expiry to enter the trade")
    otm_points_call: float = Field(description="Call leg strike distance from spot, positive multiple of 50")
    otm_points_put: float = Field(description="Put leg strike distance from spot, positive multiple of 50")
    wing_width_points: Optional[float] = Field(
        default=None, description="Iron condor wing width; must be None for short_strangle"
    )
    stop_loss_pct: float = Field(gt=0, description="Stop-loss as a positive percentage of premium/capital")
    capital: float = Field(gt=0)
    reentry: Literal["immediate", "next_cycle", "none"]

    @field_validator("otm_points_call", "otm_points_put")
    @classmethod
    def _otm_points_are_positive_multiples_of_interval(cls, v: float, info: ValidationInfo) -> float:
        if v <= 0 or not _is_multiple_of(v, STRIKE_INTERVAL):
            raise ValueError(f"{info.field_name} must be a positive multiple of {STRIKE_INTERVAL}, got {v}")
        return v

    @field_validator("wing_width_points")
    @classmethod
    def _wing_width_is_positive_multiple_of_interval_if_set(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0 or not _is_multiple_of(v, STRIKE_INTERVAL)):
            raise ValueError(f"wing_width_points must be a positive multiple of {STRIKE_INTERVAL}, got {v}")
        return v

    @model_validator(mode="after")
    def _wing_width_matches_structure(self) -> "StrategyConfig":
        if self.structure == "iron_condor" and self.wing_width_points is None:
            raise ValueError("wing_width_points must be set when structure='iron_condor'")
        if self.structure != "iron_condor" and self.wing_width_points is not None:
            raise ValueError("wing_width_points must be None unless structure='iron_condor'")
        return self
