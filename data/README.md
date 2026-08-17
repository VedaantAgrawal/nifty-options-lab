# data/

Historical options chain ingestion for NIFTY50 index options.

## Modules

- `bhavcopy_loader.py` — downloads NSE F&O bhavcopy files, normalizes them
  into a common schema, and caches the result as yearly parquet files under
  `data/cache/`.
- `expiry_calendar.py` — given any date, computes the correct next weekly
  and monthly NIFTY expiry, accounting for NSE's historical expiry-day
  regime changes and trading holidays.
- `providers.py` — `OptionsChainProvider` abstract interface plus the
  concrete `NSEBhavcopyProvider`, so downstream code depends on the
  interface rather than a specific data source.
- `holidays/nse_trading_holidays.csv` — trading holiday list consumed by
  `expiry_calendar.HolidayCalendar`. Ships as a stub (header only) — fill
  in the official NSE holiday list before relying on holiday-shifted
  expiry dates.

## Normalized schema

Every provider returns a DataFrame with these columns:

| column        | type   | notes                              |
|---------------|--------|-------------------------------------|
| date          | date   | trading date the row was recorded  |
| expiry_date   | date   | contract expiry date                |
| strike        | float  | strike price                        |
| option_type   | str    | `CE` or `PE`                        |
| close         | float  | closing price                       |
| oi            | int    | open interest                       |
| volume        | int    | traded volume (contracts)           |

## Usage

```python
from datetime import date
from data.providers import NSEBhavcopyProvider

provider = NSEBhavcopyProvider()
df = provider.get_options_chain(start=date(2021, 1, 1), end=date(2021, 1, 31))
```

Or via CLI:

```
python -m data.bhavcopy_loader --start 2021-01-01 --end 2021-12-31
```

## TODO

- `tests/test_expiry_calendar.py::KNOWN_EXPIRIES` is currently empty. Fill
  in at least one real, NSE-verified (query_date, expected_weekly_expiry,
  expected_monthly_expiry) tuple per regime to turn on
  `test_known_expiry_matches_real_nse_data` (it's skipped until then).
- `data/holidays/nse_trading_holidays.csv` ships with a header only.
  Populate it with the official NSE trading holiday list before relying on
  holiday-shifted expiry dates in `expiry_calendar`.
