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

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = range(5)

DEFAULT_HOLIDAY_CSV = Path(__file__).resolve().parent / "holidays" / "nse_trading_holidays.csv"


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


class HolidayCalendar:
    """Loadable set of NSE trading holidays, used to shift computed expiries.

    Weekends are always treated as non-trading days in addition to whatever
    dates are loaded here.
    """

    def __init__(self, holidays: Optional[Iterable[date]] = None):
        self._holidays = set(holidays or [])

    @classmethod
    def from_csv(cls, path: Path = DEFAULT_HOLIDAY_CSV) -> "HolidayCalendar":
        """Load a holiday list from a CSV with a `date` column (ISO format, YYYY-MM-DD).

        Missing file or empty CSV both just produce a calendar with no
        holidays (weekends still excluded) rather than raising, since the
        shipped CSV is a stub the user fills in over time.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        holidays = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                raw = (row.get("date") or "").strip()
                if not raw:
                    continue
                holidays.append(date.fromisoformat(raw))
        return cls(holidays)

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self.is_holiday(d)

    def previous_trading_day(self, d: date) -> date:
        """Return `d` itself if it's a trading day, else the nearest earlier trading day."""
        cur = d
        while not self.is_trading_day(cur):
            cur -= timedelta(days=1)
        return cur
