# nifty-options-lab

Data and research tooling for NIFTY50 index options.

## Status

Early stage — currently building the historical data ingestion layer
(`data/` package). No UI or strategy logic yet.

## Layout

- `data/` — options chain data ingestion, caching, and expiry-date logic.
  See [data/README.md](data/README.md) for details.
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
