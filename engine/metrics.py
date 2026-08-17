"""Aggregate performance metrics computed from a list of closed Trade objects.

No sweep/portfolio logic here -- this module only summarizes an
already-run list[Trade] (e.g. produced by calling
engine.simulator.run_single_trade repeatedly in a reentry loop, like
scripts/validate_sample.py does).
"""
from __future__ import annotations

import math
import statistics
from datetime import date
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from models.schemas import Trade

#: Annualized risk-free rate used by the Sharpe ratio unless overridden.
#: Flagged explicitly per the task: 0.0 unless told otherwise.
DEFAULT_RISK_FREE_RATE_ANNUAL = 0.0

_DAYS_PER_YEAR = 365.25


class MetricsResult(BaseModel):
    """Aggregate performance summary for a list of closed trades."""

    total_return_pct: float = Field(description="Total P&L as a percentage of initial_capital")
    total_return_abs: float = Field(description="Total P&L in currency units")
    win_rate: float = Field(
        ge=0, le=100,
        description="Percentage (0-100) of trades with pnl > 0 -- note this differs from "
        "models.schemas.SweepResult.win_rate, which is a 0-1 fraction",
    )
    max_drawdown: float = Field(
        le=0,
        description="Peak-to-trough decline on the equity curve, as a non-positive fraction of initial_capital",
    )
    sharpe_ratio: float = Field(
        description=(
            "Annualized Sharpe from per-trade returns (pnl / initial_capital). Periods/year "
            "is inferred from the trades' own date span (num_trades / years spanned) rather "
            "than assumed, since trades don't land on a fixed calendar grid (weekends, "
            "holidays, skipped cycles). 0.0 if fewer than 2 trades or the returns have zero "
            "variance."
        )
    )
    avg_pnl_per_trade: float
    num_trades: int = Field(ge=0)
    num_skipped_insufficient_margin: int = Field(
        ge=0,
        description=(
            "Count of entry attempts skipped for insufficient margin. Not derivable from "
            "`trades` -- run_single_trade returns None (not a Trade) for those, so the caller "
            "must track the count separately and pass it in."
        ),
    )
    equity_curve: List[Tuple[date, float]] = Field(
        description="(date, cumulative_capital) points, starting at (first trade's entry_date, initial_capital)"
    )


def compute_metrics(
    trades: List[Trade],
    initial_capital: Optional[float] = None,
    num_skipped_insufficient_margin: int = 0,
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
) -> MetricsResult:
    """Compute aggregate performance metrics for a list of closed trades.

    `initial_capital` is the capital basis total_return / max_drawdown /
    Sharpe are computed against. Trade only carries per-trade
    `capital_at_risk` (margin blocked for that one position), not the
    strategy's overall capital allocation, so this can't be reliably
    derived from `trades` alone in general -- if omitted, the first
    trade's `capital_at_risk` is used as a fallback approximation. Pass it
    explicitly (e.g. the `StrategyConfig.capital` that produced these
    trades) for an accurate result.

    `num_skipped_insufficient_margin` similarly can't be derived from
    `trades` and must be tracked by the caller (see the margin_skips list
    in scripts/validate_sample.py for an example) and passed in here.
    """
    if not trades:
        if initial_capital is None:
            raise ValueError("initial_capital must be provided when trades is empty")
        return MetricsResult(
            total_return_pct=0.0,
            total_return_abs=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            avg_pnl_per_trade=0.0,
            num_trades=0,
            num_skipped_insufficient_margin=num_skipped_insufficient_margin,
            equity_curve=[],
        )

    capital_basis = initial_capital if initial_capital is not None else trades[0].capital_at_risk
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)

    total_pnl = sum(t.pnl for t in trades)
    total_return_pct = (total_pnl / capital_basis) * 100 if capital_basis else 0.0

    num_winning = sum(1 for t in trades if t.pnl > 0)
    win_rate = (num_winning / len(trades)) * 100

    avg_pnl_per_trade = total_pnl / len(trades)

    # Equity curve: starts at the first trade's entry_date (before any pnl
    # has realized), then steps at each subsequent trade's exit_date (when
    # its pnl actually realizes).
    equity_curve: List[Tuple[date, float]] = [(sorted_trades[0].entry_date, capital_basis)]
    running_capital = capital_basis
    for t in sorted_trades:
        running_capital += t.pnl
        equity_curve.append((t.exit_date, running_capital))

    peak = capital_basis
    max_drawdown = 0.0
    for _, capital in equity_curve:
        peak = max(peak, capital)
        if peak > 0:
            max_drawdown = min(max_drawdown, (capital - peak) / peak)

    sharpe_ratio = 0.0
    if capital_basis:
        trade_returns = [t.pnl / capital_basis for t in trades]
        if len(trade_returns) >= 2:
            stdev_return = statistics.stdev(trade_returns)
            span_days = (sorted_trades[-1].exit_date - sorted_trades[0].entry_date).days
            years_spanned = span_days / _DAYS_PER_YEAR
            if stdev_return > 0 and years_spanned > 0:
                periods_per_year = len(trades) / years_spanned
                mean_return = statistics.mean(trade_returns)
                risk_free_rate_per_period = risk_free_rate_annual / periods_per_year
                sharpe_ratio = (
                    (mean_return - risk_free_rate_per_period) / stdev_return
                    * math.sqrt(periods_per_year)
                )

    return MetricsResult(
        total_return_pct=total_return_pct,
        total_return_abs=total_pnl,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        avg_pnl_per_trade=avg_pnl_per_trade,
        num_trades=len(trades),
        num_skipped_insufficient_margin=num_skipped_insufficient_margin,
        equity_curve=equity_curve,
    )
