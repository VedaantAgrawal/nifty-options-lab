"""Download NSE F&O bhavcopy files and cache them as normalized parquet.

NSE has shipped at least two F&O bhavcopy file formats over the date range
this project cares about (Jan 2021-present): a legacy `fo<ddmmyyyy>bhav.csv`
layout, and a newer "UDiFF" common bhavcopy layout. Rather than hardcode the
exact cutover date (which NSE itself does not document precisely and may
still change), `download_bhavcopy_csv` tries both known URL patterns for
each date, and `normalize_bhavcopy` detects which layout it got by
inspecting the actual column headers.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NSE_HOMEPAGE_URL = "https://www.nseindia.com/"

LEGACY_BHAVCOPY_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{month_abbr}/fo{ddmmyyyy}bhav.csv.zip"
)
UDIFF_BHAVCOPY_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


class BhavcopyDownloadError(RuntimeError):
    """Raised for network/HTTP failures that aren't a plain 'no file for this date'."""


class BhavcopyParseError(RuntimeError):
    """Raised when a downloaded bhavcopy file can't be parsed into the normalized schema."""


class NSESession:
    """requests.Session wrapper that warms up cookies like a browser visiting nseindia.com first.

    NSE's edge routinely blocks requests that don't carry cookies from a
    prior homepage visit, even for the static archive subdomains. We do
    that warm-up once per session and reuse it for every download.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._warmed_up = False

    def _warm_up(self) -> None:
        if self._warmed_up:
            return
        try:
            self._session.get(NSE_HOMEPAGE_URL, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("NSE homepage warm-up request failed (continuing anyway): %s", exc)
        self._warmed_up = True

    def get_bytes(self, url: str) -> Optional[bytes]:
        """Return response body bytes, or None if the URL 404s (no file published for that date)."""
        self._warm_up()
        response = self._session.get(url, timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content


def _month_abbr(d: date) -> str:
    return d.strftime("%b").upper()


def _candidate_urls(d: date) -> List[str]:
    return [
        UDIFF_BHAVCOPY_URL_TEMPLATE.format(yyyymmdd=d.strftime("%Y%m%d")),
        LEGACY_BHAVCOPY_URL_TEMPLATE.format(
            year=d.year, month_abbr=_month_abbr(d), ddmmyyyy=d.strftime("%d%m%Y")
        ),
    ]


def download_bhavcopy_csv(d: date, session: NSESession) -> Optional[pd.DataFrame]:
    """Download and unzip the raw F&O bhavcopy for one date.

    Returns None if no file is published for that date (holiday/weekend/
    not yet available) after trying every known URL pattern. Raises
    BhavcopyDownloadError for real HTTP failures (e.g. 5xx, timeouts) and
    BhavcopyParseError if a response can't be unzipped.
    """
    last_error: Optional[Exception] = None
    for url in _candidate_urls(d):
        try:
            raw = session.get_bytes(url)
        except requests.RequestException as exc:
            logger.debug("Bhavcopy fetch failed for %s at %s: %s", d, url, exc)
            last_error = exc
            continue
        if raw is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)
        except zipfile.BadZipFile as exc:
            raise BhavcopyParseError(f"Downloaded file for {d} at {url} is not a valid zip") from exc
    if last_error is not None:
        raise BhavcopyDownloadError(f"All bhavcopy URLs failed for {d}: {last_error}") from last_error
    return None
