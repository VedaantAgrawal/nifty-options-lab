# nifty-options-lab

Data and research tooling for NIFTY50 index options.

## Status

Early stage. The historical data ingestion layer (`data/` package) is in
place: NSE bhavcopy download + caching, an expiry-date calendar covering
NIFTY's historical expiry-day regime changes, and a swappable
`OptionsChainProvider` interface. No UI or strategy logic yet.

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
