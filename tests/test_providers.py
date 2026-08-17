"""Tests for data.providers.

NSEBhavcopyProvider is tested with a fake BhavcopyLoader injected in, so
these don't hit the network -- they just verify the provider correctly
delegates to whatever loader it's given, which is the whole point of the
OptionsChainProvider abstraction (downstream code can swap the loader/
provider without changing call sites).
"""
from datetime import date

import pandas as pd
import pytest

from data.bhavcopy_loader import NORMALIZED_COLUMNS
from data.providers import NSEBhavcopyProvider, OptionsChainProvider


class FakeLoader:
    def __init__(self):
        self.symbol = None
        self.calls = []

    def load(self, start, end=None, force_refresh=False):
        self.calls.append((start, end, force_refresh))
        return pd.DataFrame(
            [
                {
                    "date": start,
                    "expiry_date": start,
                    "strike": 24500.0,
                    "option_type": "CE",
                    "close": 150.0,
                    "oi": 1000,
                    "volume": 500,
                }
            ],
            columns=NORMALIZED_COLUMNS,
        )


class TestNSEBhavcopyProvider:
    def test_is_an_options_chain_provider(self):
        assert isinstance(NSEBhavcopyProvider(loader=FakeLoader()), OptionsChainProvider)

    def test_delegates_to_injected_loader(self):
        fake_loader = FakeLoader()
        provider = NSEBhavcopyProvider(loader=fake_loader)

        result = provider.get_options_chain(
            start=date(2022, 1, 1), end=date(2022, 1, 31), symbol="NIFTY"
        )

        assert fake_loader.symbol == "NIFTY"
        assert fake_loader.calls == [(date(2022, 1, 1), date(2022, 1, 31), False)]
        assert list(result.columns) == NORMALIZED_COLUMNS
        assert len(result) == 1

    def test_symbol_is_passed_through_to_loader(self):
        fake_loader = FakeLoader()
        provider = NSEBhavcopyProvider(loader=fake_loader)
        provider.get_options_chain(start=date(2022, 1, 1), symbol="BANKNIFTY")
        assert fake_loader.symbol == "BANKNIFTY"


def test_options_chain_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        OptionsChainProvider()
