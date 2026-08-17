"""Anti-overfitting safeguards for parameter sweeps.

Two-tier design, cheapest first:

1. Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014), "The
   Deflated Sharpe Ratio" -- computed per config, O(1) each. Cheap enough
   to run on every config in a sweep.
2. Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
   Cross-Validation (CSCV) -- Bailey, Borwein, Lopez de Prado & Zhu, "The
   Probability of Backtest Overfitting" -- run only on the small shortlist
   of top candidates by DSR, since CSCV is combinatorially expensive.

DSR (Part A-C below) is implemented from scratch: the `purgedcv` package
(https://pypi.org/project/purgedcv/) implements the same underlying math,
but its `effective_n_trials` and `deflated_sharpe_ratio` use different
input shapes/algorithms than what this module needs (autocorrelation of a
1-D trial-Sharpe sequence, vs. the correlation-clustering approach and
scalar-moments signature specified for this module) -- their formula for
the deflated benchmark SR* was cross-checked numerically against
`purgedcv.deflated_sharpe_ratio` during development and matches to
floating-point precision, so this reimplementation isn't guessing.

PBO (Part D) instead calls `purgedcv.probability_of_backtest_overfitting`
directly -- its CSCV implementation, default Sharpe metric (mean/std,
ddof=1, matching "compute each config's Sharpe on IS and on OOS"), and
`PBOResult.is_oos_performance` output all line up with what's needed here,
so this module adapts its input/output shape rather than reimplementing
CSCV's combinatorics from scratch.
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _scipy_skew

from models.schemas import Trade

logger = logging.getLogger(__name__)

#: Bailey & Lopez de Prado's DSR formula constant.
EULER_MASCHERONI = 0.5772156649

#: Assumed trade cycles per year for lo_adjusted_sharpe's annualization when
#: the caller doesn't override it. Matches this project's dominant cadence
#: (NIFTY weekly options, ~52 cycles/year). Pass an explicit
#: periods_per_year for monthly configs or windows with real gaps -- e.g.
#: computed the same way engine/metrics.py derives it, from the actual
#: trade date span.
DEFAULT_PERIODS_PER_YEAR = 52.0


def trade_level_returns(trades: List[Trade]) -> np.ndarray:
    """One return observation per trade cycle: pnl / capital_at_risk.

    Deliberately NOT a daily-marked series -- options positions here are
    held ~1 week and settled, so per-cycle returns are the honest
    granularity, not smoothed daily marks. Sorted by exit_date (the same
    chronological convention engine/metrics.py uses), since downstream
    consumers here (lo_adjusted_sharpe) depend on genuine time order for
    their autocorrelation estimates.
    """
    sorted_trades = sorted(trades, key=lambda t: t.exit_date)
    return np.array([t.pnl / t.capital_at_risk for t in sorted_trades], dtype=float)


def _sample_autocorrelations(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """rho_1..rho_max_lag of `returns`, using the standard biased ACF estimator
    (single full-sample denominator, as in Lo 2002 / numpy/statsmodels' acf)."""
    deviations = returns - returns.mean()
    denom = float(np.sum(deviations ** 2))
    if denom == 0.0:
        return np.zeros(max_lag)
    rhos = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        rhos[k - 1] = float(np.sum(deviations[k:] * deviations[:-k])) / denom
    return rhos


def lo_adjusted_sharpe(
    returns: np.ndarray,
    max_lag: int = 5,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Lo (2002) autocorrelation-adjusted annualized Sharpe ratio.

    Naive annualization (SR * sqrt(q)) assumes IID returns. When returns
    are positively autocorrelated -- which immediately-reentered or
    overlapping positions (config.reentry="immediate") tend to induce --
    naive annualization overstates the Sharpe. Lo's correction:

        SR(q) = SR(1) * eta(q),   eta(q) = q / sqrt(q + 2*sum_k (q-k)*rho_k)

    where SR(1) is the per-period Sharpe and rho_k is the k-th sample
    autocorrelation. rho_k=0 for all k recovers eta(q)=sqrt(q), the naive
    case. Summed over k=1..max_lag (not the full q-1 terms of Lo's
    original derivation) since higher-lag autocorrelation estimates get
    unreliable fast with the small trade counts this project deals with.

    `periods_per_year` isn't part of Lo's own notation but is required to
    actually annualize; see DEFAULT_PERIODS_PER_YEAR for the assumption
    used when the caller doesn't override it.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    sr1 = returns.mean() / std

    q = periods_per_year
    usable_max_lag = max(0, min(max_lag, n - 1))
    rhos = _sample_autocorrelations(returns, usable_max_lag) if usable_max_lag > 0 else np.array([])

    correction = q
    for k in range(1, usable_max_lag + 1):
        correction += 2 * (q - k) * rhos[k - 1]
    # A short, noisy sample can push the correction to <=0 (not
    # theoretically possible for a true variance, but the finite-lag
    # estimator isn't guaranteed positive-definite) -- guard rather than
    # let sqrt raise or eta blow up.
    correction = max(correction, 1e-12)

    eta = q / math.sqrt(correction)
    return float(sr1 * eta)
