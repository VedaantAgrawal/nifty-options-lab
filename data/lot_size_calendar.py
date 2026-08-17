"""NIFTY lot size (contracts per lot) calendar, accounting for historical
NSE lot-size revisions.

Dates below were verified against NSE circulars and market-bulletin
write-ups (Zerodha, Business Standard, Angel One), not assumed:

* Before 1 Aug 2021: 75. NSE cut this to 50 via a circular dated 1 Apr
  2021, but the rollout was phased, not a single date: the first 50-lot
  monthly contract (July 2021 expiry) started trading 30 Apr 2021, while
  the Apr/May/Jun 2021 monthlies kept lot 75 until their own expiry, and
  *weekly* contracts (the dominant share of NIFTY options volume) didn't
  move to 50 until August 2021. This module uses 1 Aug 2021 as a single
  approximating cutover, chosen because it's correct for essentially all
  weekly contracts on both sides -- see the caveat below.
* 1 Aug 2021 - 25 Apr 2024: 50.
* 26 Apr 2024 - 19 Nov 2024: 25. NSE halved the lot size effective for
  contracts trading from 26 Apr 2024 onward (that month's monthly expiry,
  25 Apr 2024, kept lot 50).
* 20 Nov 2024 - 27 Oct 2025: 75. SEBI-driven increase, effective for new
  contracts from 20 Nov 2024 (matches the boundary already used in
  data/expiry_calendar.py for the weekly-expiry-day consolidation).
* From 28 Oct 2025: 65. NSE circular NSE/FAOP/70616 (3 Oct 2025), new
  contracts from EOD 28 Oct 2025; existing 75-lot contracts continued
  trading until their 30 Dec 2025 expiry.

Caveat: lot size is fixed per contract at listing, not per calendar date
across all currently-live contracts. Every boundary above therefore has a
real-world grace window (days to a few months, depending on whether
weekly, monthly, or quarterly contracts are involved) where old- and
new-lot contracts trade side by side. This module approximates each
transition as a single date because the options chain data this project
works with (data/bhavcopy_loader.py) doesn't carry a contract listing
date -- only expiry date and daily prices -- so an exact per-contract
lookup isn't possible from that data alone. The 1 Aug 2021 cutover has by
far the widest error window (see above); the 2024/2025 cutovers are each
documented by NSE as applying to trading from that date, so the
approximation is much tighter there.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from typing import List


@dataclass(frozen=True)
class LotSizeRegime:
    """A NIFTY lot size that applies from `start` until the next regime's start."""

    start: date
    lot_size: int
    label: str = ""


# Sorted ascending by start date. lot_size_for_date() picks the regime with
# the latest start <= the queried date, so there are no gaps between
# regimes regardless of the exact boundary dates above.
NIFTY_LOT_SIZE_REGIMES: List[LotSizeRegime] = [
    LotSizeRegime(date(2019, 1, 1), 75, "pre-2021 baseline"),
    LotSizeRegime(date(2021, 8, 1), 50, "post Apr-2021 circular; weeklies moved Aug 2021 (approximated cutover)"),
    LotSizeRegime(date(2024, 4, 26), 25, "Apr-2024 halving"),
    LotSizeRegime(date(2024, 11, 20), 75, "Nov-2024 SEBI-driven increase"),
    LotSizeRegime(date(2025, 10, 28), 65, "Oct-2025 revision (NSE/FAOP/70616)"),
]

_REGIME_STARTS = [regime.start for regime in NIFTY_LOT_SIZE_REGIMES]


class UnsupportedLotSizeDateError(ValueError):
    """Raised when a date falls outside any known lot-size regime."""


def lot_size_for_date(d: date) -> int:
    """Return the NIFTY lot size (contracts per lot) applicable on `d`.

    See the module docstring for verified source dates and the caveat about
    approximated transition windows, especially around 1 Aug 2021.
    """
    idx = bisect_right(_REGIME_STARTS, d) - 1
    if idx < 0:
        raise UnsupportedLotSizeDateError(
            f"No known NIFTY lot-size regime covers {d} "
            f"(earliest modeled regime starts {_REGIME_STARTS[0]})"
        )
    return NIFTY_LOT_SIZE_REGIMES[idx].lot_size
