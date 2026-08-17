"""NIFTY weekly/monthly expiry calendar, accounting for historical regime changes.

Regime history modeled here (NIFTY only — see caveat below):

* Weekly NIFTY index options have existed since 2019, expiring on Thursday.
* 20 Nov 2024: SEBI mandated that each exchange offer only one weekly
  expiry. NSE consolidated onto Thursday (NIFTY did not change weekday at
  this boundary; Bank Nifty/Fin Nifty/Midcap Nifty did). The boundary is
  still modeled explicitly here for traceability even though the weekday
  is unchanged for NIFTY.
* 2 Sept 2025: NSE moved its weekly expiry day from Thursday to Tuesday.

Caveat: before 20 Nov 2024, other NSE indices (Bank Nifty, Fin Nifty,
Midcap Nifty) had their own weekly expiry weekdays, different from
NIFTY's. Do not reuse NIFTY_WEEKLY_EXPIRY_REGIMES for those indices.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from typing import List

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = range(5)


@dataclass(frozen=True)
class ExpiryRegime:
    """A weekly-expiry weekday that applies from `start` until the next regime's start."""

    start: date
    weekday: int
    label: str = ""


# Sorted ascending by start date. weekday_for_date() picks the regime with
# the latest start <= the queried date, so there are no gaps between
# regimes regardless of the exact boundary dates below.
NIFTY_WEEKLY_EXPIRY_REGIMES: List[ExpiryRegime] = [
    ExpiryRegime(date(2019, 1, 1), THURSDAY, "pre-consolidation (NIFTY already Thursday)"),
    ExpiryRegime(date(2024, 11, 20), THURSDAY, "post-SEBI consolidation, still Thursday"),
    ExpiryRegime(date(2025, 9, 2), TUESDAY, "NSE moved weekly expiry to Tuesday"),
]

_REGIME_STARTS = [regime.start for regime in NIFTY_WEEKLY_EXPIRY_REGIMES]


class UnsupportedDateError(ValueError):
    """Raised when a date falls outside any known expiry regime."""


def weekday_for_date(d: date) -> int:
    """Return the weekday (Monday=0) that governs NIFTY weekly/monthly expiries on `d`."""
    idx = bisect_right(_REGIME_STARTS, d) - 1
    if idx < 0:
        raise UnsupportedDateError(
            f"No known NIFTY weekly-expiry regime covers {d} "
            f"(earliest modeled regime starts {_REGIME_STARTS[0]})"
        )
    return NIFTY_WEEKLY_EXPIRY_REGIMES[idx].weekday
