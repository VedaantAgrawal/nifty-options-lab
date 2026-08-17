"""Tests for data.bhavcopy_loader.

Normalization tests use static fixture CSVs (no network). Caching tests
monkeypatch download_bhavcopy_csv so BhavcopyLoader.load() can be exercised
without hitting NSE.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.bhavcopy_loader import (
    NORMALIZED_COLUMNS,
    BhavcopyLoader,
    BhavcopyParseError,
    normalize_bhavcopy,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestNormalizeBhavcopyLegacy:
    def _load_raw(self) -> pd.DataFrame:
        return pd.read_csv(FIXTURES_DIR / "legacy_bhavcopy_sample.csv")

    def test_filters_to_nifty_index_options_only(self):
        raw = self._load_raw()
        result = normalize_bhavcopy(raw, date(2022, 1, 4), symbol="NIFTY")
        # 3 NIFTY OPTIDX rows in the fixture; BANKNIFTY and the FUTIDX row are dropped.
        assert len(result) == 3
        assert list(result.columns) == NORMALIZED_COLUMNS

    def test_parses_expiry_strike_and_option_type(self):
        raw = self._load_raw()
        result = normalize_bhavcopy(raw, date(2022, 1, 4), symbol="NIFTY")
        row = result[(result["strike"] == 17500.0) & (result["option_type"] == "CE")].iloc[0]
        assert row["expiry_date"] == date(2022, 1, 6)
        assert row["close"] == pytest.approx(228.4)
        assert row["oi"] == 45000
        assert row["volume"] == 1200
        assert row["date"] == date(2022, 1, 4)


class TestNormalizeBhavcopyUdiff:
    def _load_raw(self) -> pd.DataFrame:
        return pd.read_csv(FIXTURES_DIR / "udiff_bhavcopy_sample.csv")

    def test_filters_to_nifty_options_only(self):
        raw = self._load_raw()
        result = normalize_bhavcopy(raw, date(2025, 9, 9), symbol="NIFTY")
        # 3 NIFTY CE/PE rows; BANKNIFTY row and the futures (OptnTp=XX) row are dropped.
        assert len(result) == 3
        assert list(result.columns) == NORMALIZED_COLUMNS

    def test_parses_expiry_strike_and_option_type(self):
        raw = self._load_raw()
        result = normalize_bhavcopy(raw, date(2025, 9, 9), symbol="NIFTY")
        row = result[(result["strike"] == 24500.0) & (result["option_type"] == "PE")].iloc[0]
        assert row["expiry_date"] == date(2025, 9, 9)
        assert row["close"] == pytest.approx(118.0)
        assert row["oi"] == 41000
        assert row["volume"] == 12000


class TestNormalizeBhavcopyUnknownFormat:
    def test_raises_on_unrecognized_columns(self):
        raw = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(BhavcopyParseError):
            normalize_bhavcopy(raw, date(2022, 1, 4))


class TestBhavcopyLoaderCaching:
    def test_load_writes_and_reuses_parquet_cache(self, tmp_path, monkeypatch):
        raw = pd.read_csv(FIXTURES_DIR / "legacy_bhavcopy_sample.csv")
        call_count = {"n": 0}

        def fake_download(d, session):
            call_count["n"] += 1
            return raw

        monkeypatch.setattr("data.bhavcopy_loader.download_bhavcopy_csv", fake_download)

        loader = BhavcopyLoader(symbol="NIFTY", cache_dir=tmp_path)
        start, end = date(2022, 1, 3), date(2022, 1, 4)  # one weekday (Mon 1/3 + Tue 1/4)

        first = loader.load(start=start, end=end)
        assert not first.empty
        first_call_count = call_count["n"]
        assert first_call_count == 2  # one download attempt per weekday in range

        cache_file = tmp_path / "nifty_options_2022.parquet"
        assert cache_file.exists()

        second = loader.load(start=start, end=end)
        assert call_count["n"] == first_call_count  # no new downloads, served from cache
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), second.reset_index(drop=True)
        )

    def test_load_skips_dates_with_no_published_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.bhavcopy_loader.download_bhavcopy_csv", lambda d, session: None)
        loader = BhavcopyLoader(symbol="NIFTY", cache_dir=tmp_path)
        result = loader.load(start=date(2022, 1, 3), end=date(2022, 1, 4))
        assert result.empty
        assert list(result.columns) == NORMALIZED_COLUMNS
