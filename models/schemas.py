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
    exit_price: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Premium paid/received at exit (0 if expired worthless); None if the leg "
            "hasn't been closed yet, e.g. as returned by engine/position_builder.py "
            "before a trade is closed and wrapped into a Trade"
        ),
    )
    lots: int = Field(gt=0, description="Number of lots traded for this leg")

    @field_validator("strike")
    @classmethod
    def _strike_is_positive_multiple_of_interval(cls, v: float) -> float:
        if v <= 0 or not _is_multiple_of(v, STRIKE_INTERVAL):
            raise ValueError(f"strike must be a positive multiple of {STRIKE_INTERVAL}, got {v}")
        return v


class Trade(BaseModel):
    """A trade that was actually opened and has since fully closed.

    Every leg must have both `entry_price` and `exit_price` populated
    (`entry_price` is already guaranteed non-None by `OptionLeg` itself;
    `exit_price` is checked below since `OptionLeg` allows it to be unset
    for legs that haven't closed yet).

    A trade attempt that never opened -- e.g. skipped because required
    margin exceeded available capital -- has no entry, no exit, and no
    `exit_reason` to report, so it must never be represented as a `Trade`
    instance (not even with sentinel/zero values). Whatever code decides
    whether to open a position should return `None` for that case instead.
    """

    entry_date: date
    expiry_date: date
    exit_date: date
    legs: List[OptionLeg] = Field(min_length=1, description="At least one leg")
    exit_reason: Literal["stop_loss", "expiry"] = Field(
        description="How this opened trade closed. Never used to represent a trade that never opened."
    )
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

    @model_validator(mode="after")
    def _all_legs_are_closed(self) -> "Trade":
        if any(leg.exit_price is None for leg in self.legs):
            raise ValueError("every leg must have exit_price set for a closed Trade")
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


class ParameterGrid(BaseModel):
    """Same fields as `StrategyConfig`, each a list of candidate values.

    The engine (not built yet) is expected to take the Cartesian product of
    all fields to produce individual `StrategyConfig` instances -- so
    cross-field checks like the iron_condor/wing_width_points relationship
    are validated per-generated-combo by `StrategyConfig` itself, not here.
    This schema only validates that each dimension's own values are
    individually well-formed.
    """

    structure: List[Literal["short_strangle", "iron_condor"]] = Field(min_length=1)
    expiry_cycle: List[Literal["weekly", "monthly"]] = Field(min_length=1)
    entry_day_of_week: List[int] = Field(min_length=1)
    days_to_expiry_at_entry: List[int] = Field(min_length=1)
    otm_points_call: List[float] = Field(min_length=1)
    otm_points_put: List[float] = Field(min_length=1)
    wing_width_points: List[Optional[float]] = Field(
        default_factory=lambda: [None],
        description="Candidate wing widths; include None for combos where structure != iron_condor",
    )
    stop_loss_pct: List[float] = Field(min_length=1)
    capital: List[float] = Field(min_length=1)
    reentry: List[Literal["immediate", "next_cycle", "none"]] = Field(min_length=1)

    @field_validator("entry_day_of_week")
    @classmethod
    def _entry_day_of_week_values_in_range(cls, v: List[int]) -> List[int]:
        for day in v:
            if not (0 <= day <= 6):
                raise ValueError(f"entry_day_of_week values must be 0-6 (Monday=0), got {day}")
        return v

    @field_validator("days_to_expiry_at_entry")
    @classmethod
    def _days_to_expiry_values_non_negative(cls, v: List[int]) -> List[int]:
        for days in v:
            if days < 0:
                raise ValueError(f"days_to_expiry_at_entry values must be >= 0, got {days}")
        return v

    @field_validator("otm_points_call", "otm_points_put")
    @classmethod
    def _otm_points_values_are_positive_multiples_of_interval(cls, v: List[float], info: ValidationInfo) -> List[float]:
        for points in v:
            if points <= 0 or not _is_multiple_of(points, STRIKE_INTERVAL):
                raise ValueError(
                    f"{info.field_name} values must be positive multiples of {STRIKE_INTERVAL}, got {points}"
                )
        return v

    @field_validator("wing_width_points")
    @classmethod
    def _wing_width_values_are_positive_multiples_of_interval_if_set(
        cls, v: List[Optional[float]]
    ) -> List[Optional[float]]:
        for width in v:
            if width is not None and (width <= 0 or not _is_multiple_of(width, STRIKE_INTERVAL)):
                raise ValueError(f"wing_width_points values must be positive multiples of {STRIKE_INTERVAL}, got {width}")
        return v

    @field_validator("stop_loss_pct")
    @classmethod
    def _stop_loss_pct_values_are_positive(cls, v: List[float]) -> List[float]:
        for pct in v:
            if pct <= 0:
                raise ValueError(f"stop_loss_pct values must be positive, got {pct}")
        return v

    @field_validator("capital")
    @classmethod
    def _capital_values_are_positive(cls, v: List[float]) -> List[float]:
        for amount in v:
            if amount <= 0:
                raise ValueError(f"capital values must be positive, got {amount}")
        return v


class SweepResult(BaseModel):
    """Aggregate backtest performance for a single `StrategyConfig` from a sweep run."""

    config: StrategyConfig
    total_return: float = Field(description="Total return over the backtest period, as a fraction of capital")
    win_rate: float = Field(ge=0, le=1, description="Fraction of trades that were profitable")
    max_drawdown: float = Field(le=0, description="Peak-to-trough decline as a non-positive fraction of capital")
    sharpe: float
    avg_pnl_per_trade: float
    num_trades: int = Field(ge=0)

    # Anti-overfitting diagnostics (see engine/robustness.py). Optional since
    # they're only populated once a sweep has been run through
    # compute_sweep_dsr/analyze_sweep_robustness -- a plain single-backtest
    # SweepResult has no meaningful "sweep" to deflate against.
    sharpe_lo_adjusted: Optional[float] = Field(
        default=None,
        description="Lo (2002) autocorrelation-adjusted annualized Sharpe, correcting for "
        "inflation from overlapping/immediately-reentered positions",
    )
    skew: Optional[float] = Field(
        default=None, description="Bias-corrected sample skew of per-cycle trade returns"
    )
    kurtosis: Optional[float] = Field(
        default=None,
        description="Bias-corrected sample kurtosis (NOT excess) of per-cycle trade returns",
    )
    dsr: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): probability the "
        "true Sharpe exceeds the benchmark expected from n_eff_trials independent searches",
    )
    n_eff_trials: Optional[int] = Field(
        default=None,
        ge=2,
        description="Effective number of independent trials across the whole sweep "
        "(engine.robustness.effective_n_trials, floored at 2) -- the same value on every "
        "row of a sweep",
    )
    robustness_flag: Optional[Literal["green", "amber", "red"]] = Field(
        default=None,
        description="green: dsr>=0.95, amber: 0.90<=dsr<0.95, red: dsr<0.90",
    )
