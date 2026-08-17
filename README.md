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
- `engine/simulator.py` — runs one full trade cycle: opens a position
  (via `position_builder`), walks it forward day by day marking it to
  market, exits on a stop-loss or at expiry, and returns a closed `Trade`
  (or `None`, logged, if required margin exceeds capital). Lot size is
  resolved per `entry_date` via `data/lot_size_calendar.py`.
- `engine/metrics.py` — `compute_metrics(trades)` aggregates a list of
  closed `Trade`s into a `MetricsResult`: total return, win rate, max
  drawdown, annualized Sharpe (risk-free rate defaults to 0), avg P&L per
  trade, trade counts, and the equity curve. `num_skipped_insufficient_margin`
  must be tracked and passed in by the caller, since skipped attempts are
  `None`, never `Trade` instances.
- `engine/robustness.py` — two-tier anti-overfitting safeguard for
  parameter sweeps: cheap Deflated Sharpe Ratio (DSR, Bailey & Lopez de
  Prado 2014) computed per config via `compute_sweep_dsr`, and expensive
  Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
  Cross-Validation computed only on the DSR shortlist via `compute_pbo`
  (delegates CSCV to the `purgedcv` package). `analyze_sweep_robustness`
  wires both together. Not yet called from anywhere — `engine/sweep.py`
  doesn't exist yet.
- `scripts/validate_sample.py` — a real, non-optimized validation run: loads
  the last 3 months of NIFTY data, runs the entry/exit/reentry loop for one
  fixed `short_strangle` config, and writes a plain per-trade CSV plus a
  printed summary of margin-skip and missing-strike events. Run via
  `python scripts/validate_sample.py`.

No parameter sweeps, portfolio-level aggregation, or metrics/UI yet.

## Layout

- `data/` — options chain data ingestion, caching, and expiry-date logic.
  See [data/README.md](data/README.md) for details.
- `models/` — Pydantic schemas shared across the backtest engine.
- `engine/` — backtest engine logic (currently just position construction
  and single-trade simulation).
- `scripts/` — standalone runnable scripts (not imported by anything else).
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
