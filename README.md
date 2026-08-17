# nifty-options-lab

Data and research tooling for NIFTY50 index options.

## Status

Early stage. In place so far:

- `data/` — NSE bhavcopy download + caching, an expiry-date calendar
  covering NIFTY's historical expiry-day regime changes, and a swappable
  `OptionsChainProvider` interface.
- `models/` — Pydantic schemas for the backtest engine (legs, trades,
  strategy config, parameter sweeps, results).
- `engine/position_builder.py` — computes strikes and looks up entry
  premiums for a strategy config on a given date, with nearest-strike
  fallback for illiquid/missing strikes.

No trade execution, exit logic, P&L calculation, or UI yet.

## Layout

- `data/` — options chain data ingestion, caching, and expiry-date logic.
  See [data/README.md](data/README.md) for details.
- `models/` — Pydantic schemas shared across the backtest engine.
- `engine/` — backtest engine logic (currently just position construction).
- `tests/` — unit tests (pytest).

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Tests

```
python -m pytest
```
