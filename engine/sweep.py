"""Parameter sweep: Cartesian product of a ParameterGrid, run in parallel
across a train/validation date split.

Reuses the exact reentry-loop logic scripts/validate_sample.py introduced
(`run_trade_cycle_loop` below is that same logic, extracted here as the
canonical implementation -- validate_sample.py now imports it rather than
keeping its own copy, so there's exactly one reentry scheduler, not two
that could quietly drift apart).
"""
from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from data.expiry_calendar import HolidayCalendar
from data.providers import NSEBhavcopyProvider, OptionsChainProvider
from engine.metrics import MetricsResult, compute_metrics
from engine.position_builder import _resolve_expiry
from engine.robustness import compute_pbo, compute_sweep_dsr
from engine.simulator import run_single_trade
from models.schemas import ParameterGrid, PBOResult, StrategyConfig, Trade

logger = logging.getLogger(__name__)

#: ParameterGrid field names, in the order StrategyConfig's constructor expects them.
_GRID_FIELD_NAMES = [
    "structure",
    "expiry_cycle",
    "entry_day_of_week",
    "days_to_expiry_at_entry",
    "otm_points_call",
    "otm_points_put",
    "wing_width_points",
    "stop_loss_pct",
    "capital",
    "reentry",
]


def generate_configs(grid: ParameterGrid) -> Tuple[List[StrategyConfig], int]:
    """Cartesian product of every field in `grid`, as StrategyConfig instances.

    Combos that fail StrategyConfig's own cross-field validation (e.g.
    wing_width_points set for a non-iron_condor structure) are skipped, not
    raised -- ParameterGrid's docstring already establishes this as the
    expected split of responsibility: the grid only validates each
    dimension's own values, and cross-field checks happen per generated
    combo, here.

    Returns (valid_configs, num_invalid_skipped).
    """
    value_lists = [getattr(grid, name) for name in _GRID_FIELD_NAMES]
    configs: List[StrategyConfig] = []
    num_invalid = 0
    for combo in itertools.product(*value_lists):
        kwargs = dict(zip(_GRID_FIELD_NAMES, combo))
        try:
            configs.append(StrategyConfig(**kwargs))
        except ValidationError:
            num_invalid += 1
    return configs, num_invalid


def _next_occurrence_of_weekday(after: date, weekday: int) -> date:
    d = after
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def _next_trading_day(after: date, holidays: HolidayCalendar) -> date:
    d = after
    while not holidays.is_trading_day(d):
        d += timedelta(days=1)
    return d


def _next_entry_after(after: date, config: StrategyConfig, holidays: HolidayCalendar) -> date:
    """Next scheduled cycle's entry date strictly after `after` (next occurrence
    of config.entry_day_of_week, shifted onto a real trading day)."""
    candidate = _next_occurrence_of_weekday(after + timedelta(days=1), config.entry_day_of_week)
    return _next_trading_day(candidate, holidays)


def _next_entry_after_exit(exit_date: date, config: StrategyConfig, holidays: HolidayCalendar) -> Optional[date]:
    """Where the next trade cycle should start, per config.reentry. None means stop."""
    if config.reentry == "none":
        return None
    if config.reentry == "immediate":
        return _next_trading_day(exit_date + timedelta(days=1), holidays)
    # "next_cycle"
    return _next_entry_after(exit_date, config, holidays)


def estimate_spot_price(day_chain: pd.DataFrame, expiry: date) -> Optional[float]:
    """Synthetic spot estimate via put-call parity (S ~= K + call_close - put_close),
    averaged over the 5 strikes closest to at-the-money.

    Our options chain data (data/bhavcopy_loader.py) carries no direct
    underlying/spot price column, so this is an estimate, not a recorded
    value. Good enough for OTM-points strike selection (which rounds to the
    nearest 50 anyway) -- don't rely on it for anything needing true spot
    precision.
    """
    subset = day_chain[day_chain["expiry_date"] == expiry]
    calls = subset[subset["option_type"] == "CE"].set_index("strike")["close"]
    puts = subset[subset["option_type"] == "PE"].set_index("strike")["close"]
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        return None
    diffs = (calls.loc[common_strikes] - puts.loc[common_strikes]).abs()
    nearest = diffs.nsmallest(min(5, len(diffs))).index
    implied_spots = [strike + calls[strike] - puts[strike] for strike in nearest]
    return float(sum(implied_spots) / len(implied_spots))


def run_trade_cycle_loop(
    config: StrategyConfig,
    start: date,
    end: date,
    chain: pd.DataFrame,
    holidays: Optional[HolidayCalendar] = None,
) -> Tuple[List[Trade], List[date]]:
    """Run the full entry -> exit -> reentry loop for ONE config over [start, end].

    Walks scheduled entry dates (per config.entry_day_of_week), opening a
    trade via run_single_trade at each one, and decides the next entry date
    from the previous trade's exit per config.reentry ("immediate" tries
    again the next trading day even mid-cycle; "next_cycle" waits for the
    next scheduled entry; "none" stops after the first trade). Stops
    without attempting a cycle whose expiry would fall after `end` (can't
    validate a still-open position with data that doesn't exist yet).

    Returns (closed_trades, skipped_entry_dates) -- skipped_entry_dates are
    the entry attempts where run_single_trade returned None (insufficient
    margin); len(skipped_entry_dates) is what compute_metrics wants as
    num_skipped_insufficient_margin.
    """
    holidays = holidays or HolidayCalendar.from_csv()
    trades: List[Trade] = []
    skipped_entry_dates: List[date] = []

    entry_date: Optional[date] = _next_entry_after(start - timedelta(days=1), config, holidays)

    while entry_date is not None and entry_date <= end:
        day_chain = chain[chain["date"] == entry_date]
        if day_chain.empty:
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        expiry = _resolve_expiry(entry_date, config, holidays=holidays)
        # `expiry <= end` alone isn't enough: `end` is a nominal date, but the
        # chain might not actually have a published row for it yet (e.g. a
        # same-day run before today's EOD bhavcopy is out). Without this,
        # run_single_trade's walk falls short of the real expiry but still
        # labels the trade exit_reason="expiry", misrepresenting an
        # incomplete cycle as a naturally closed one.
        if expiry > end or chain[chain["date"] == expiry].empty:
            break

        spot_price = estimate_spot_price(day_chain, expiry)
        if spot_price is None:
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        trade = run_single_trade(entry_date, config, spot_price, chain=chain)

        if trade is None:
            skipped_entry_dates.append(entry_date)
            entry_date = _next_entry_after(entry_date, config, holidays)
            continue

        trades.append(trade)
        entry_date = _next_entry_after_exit(trade.exit_date, config, holidays)

    return trades, skipped_entry_dates


# --- Parallel batch execution ------------------------------------------------
# Module-level globals set once per worker process via the ProcessPoolExecutor
# initializer, not passed per-task: with sweeps running into the thousands of
# configs, re-pickling `chain` (potentially large) for every single task
# instead of once per worker would dominate runtime.
_worker_chain: Optional[pd.DataFrame] = None
_worker_holidays: Optional[HolidayCalendar] = None


def _init_worker(chain: pd.DataFrame, holidays: HolidayCalendar) -> None:
    global _worker_chain, _worker_holidays
    _worker_chain = chain
    _worker_holidays = holidays


def _run_one_config(args: Tuple[StrategyConfig, date, date]) -> Tuple[MetricsResult, List[Trade]]:
    config, start, end = args
    trades, skipped_entry_dates = run_trade_cycle_loop(config, start, end, _worker_chain, _worker_holidays)
    metrics = compute_metrics(
        trades,
        initial_capital=config.capital,
        num_skipped_insufficient_margin=len(skipped_entry_dates),
    )
    return metrics, trades


def _run_batch(
    configs: List[StrategyConfig],
    start: date,
    end: date,
    chain: pd.DataFrame,
    holidays: HolidayCalendar,
    max_workers: Optional[int],
) -> List[Tuple[MetricsResult, List[Trade]]]:
    """Run every config in `configs` over [start, end] in parallel worker
    processes. Returns (MetricsResult, trades) in the SAME ORDER as
    `configs` (ProcessPoolExecutor.map preserves input order), so callers
    can pair results back up by index."""
    args = [(config, start, end) for config in configs]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(chain, holidays)) as executor:
        return list(executor.map(_run_one_config, args))


def _flatten_config(config: StrategyConfig) -> dict:
    return {name: getattr(config, name) for name in _GRID_FIELD_NAMES}


#: MetricsResult fields usable as rank_metric / flattened into DataFrame columns.
#: equity_curve is excluded -- it's a list of (date, capital) points, not a
#: scalar, and doesn't fit a per-config comparison row (or CSV export).
_METRIC_FIELD_NAMES = [name for name in MetricsResult.model_fields if name != "equity_curve"]

#: SweepResult fields populated from engine.robustness.compute_sweep_dsr,
#: copied onto each output row.
_ROBUSTNESS_FIELD_NAMES = ["sharpe_lo_adjusted", "skew", "kurtosis", "dsr", "n_eff_trials", "robustness_flag"]


def _build_pnl_matrix(trades_per_config: List[List[Trade]]) -> pd.DataFrame:
    """(T x N) aligned per-cycle return matrix for engine.robustness, built
    from each config's own realized trades.

    Indexed by the UNION of exit_dates seen across every config (T rows),
    columned by config index 0..N-1 (matching the row order of the
    `configs`/`train_metrics` lists this is built alongside). A config
    without a trade on some other config's exit_date gets NaN there -- this
    is expected, not a bug: configs don't necessarily share an identical
    reentry schedule (e.g. one hit a margin skip another didn't, or their
    stop-loss triggers diverged), and this alignment can't assume otherwise.
    effective_n_trials' correlation matrix tolerates NaN (pandas .corr()
    uses pairwise-complete observations); compute_pbo's CSCV cannot, so its
    caller is responsible for dropping incomplete rows from whatever subset
    of columns it actually uses.
    """
    series_list = []
    for idx, trades in enumerate(trades_per_config):
        if not trades:
            series_list.append(pd.Series(name=idx, dtype=float))
            continue
        returns = {t.exit_date: t.pnl / t.capital_at_risk for t in trades}
        series_list.append(pd.Series(returns, name=idx))
    return pd.concat(series_list, axis=1).sort_index()


def run_sweep(
    grid: ParameterGrid,
    train_range: Tuple[date, date],
    validation_range: Tuple[date, date],
    provider: Optional[OptionsChainProvider] = None,
    chain: Optional[pd.DataFrame] = None,
    holidays: Optional[HolidayCalendar] = None,
    top_n: int = 20,
    rank_metric: str = "sharpe_ratio",
    max_workers: Optional[int] = None,
    top_n_for_pbo: int = 30,
    s_splits: int = 10,
) -> Tuple[pd.DataFrame, PBOResult]:
    """Run every valid StrategyConfig in `grid`'s Cartesian product over
    `train_range`, rank by `rank_metric` (any MetricsResult field except
    equity_curve; higher is always better, including for max_drawdown where
    less-negative is the better raw value), and re-run only the top `top_n`
    over `validation_range` -- so in-sample and out-of-sample performance
    are visible side by side for every shortlisted config, not just
    whichever one "won" on train.

    Also runs the engine.robustness anti-overfitting pipeline on the train
    results: Deflated Sharpe Ratio (Parts A-C) for every config, and
    Probability of Backtest Overfitting via CSCV (Part D) on the top
    `top_n_for_pbo` configs by dsr. dsr < 0.90 ("red") excludes a config
    from `recommended` -- deprioritized, not dropped: it's still a row in
    the returned DataFrame, fully inspectable.

    Returns (df, pbo_result):
      df -- one row per config, sorted by train_{rank_metric} descending:
        the config's own fields, train_* metrics (every config),
        validation_* metrics (only the top_n configs; None elsewhere, since
        re-running the whole grid twice would defeat the point of the
        split), the robustness columns (sharpe_lo_adjusted, skew, kurtosis,
        dsr, n_eff_trials, robustness_flag), and `recommended` (bool:
        robustness_flag != "red").
      pbo_result -- a PBOResult: a property of the sweep/selection process
        as a whole, not of any individual config, so it isn't a per-row
        column.

    NOTE: the task that requested this wiring described the per-config
    return type as `list[SweepResult]`. This still returns the flat
    DataFrame the previous step built instead -- SweepResult has no
    train-vs-validation shape (it's a single flat metric set), and
    reshaping it to fit would mean either duplicating every metric field
    with train_/validation_ prefixes inside the model too, or dropping the
    flattened comparison columns that already work. Only the *addition* of
    PBOResult as a second, separate return value is new.

    Pass `chain` directly (e.g. in tests) to skip fetching via `provider`.
    Parallelized with multiprocessing (ProcessPoolExecutor): each config's
    backtest is independent, and sweeps can run into the thousands of
    combos. `max_workers=None` uses ProcessPoolExecutor's default
    (os.process_cpu_count()); pass a small value to keep test runs from
    paying full process-pool startup overhead.
    """
    if rank_metric not in _METRIC_FIELD_NAMES:
        raise ValueError(
            f"rank_metric must be a numeric MetricsResult field, got {rank_metric!r} "
            f"(options: {sorted(_METRIC_FIELD_NAMES)})"
        )

    configs, num_invalid = generate_configs(grid)
    if num_invalid:
        logger.info(
            "Skipped %d invalid parameter combinations (failed StrategyConfig validation)", num_invalid
        )
    if not configs:
        raise ValueError("No valid StrategyConfig combinations were generated from this grid")

    holidays = holidays or HolidayCalendar.from_csv()
    train_start, train_end = train_range
    val_start, val_end = validation_range

    if chain is None:
        provider = provider or NSEBhavcopyProvider()
        fetch_start = min(train_start, val_start)
        fetch_end = max(train_end, val_end)
        chain = provider.get_options_chain(start=fetch_start, end=fetch_end)

    train_results = _run_batch(configs, train_start, train_end, chain, holidays, max_workers)
    train_metrics = [r[0] for r in train_results]
    train_trades = [r[1] for r in train_results]

    ranked_idx = sorted(range(len(configs)), key=lambda i: getattr(train_metrics[i], rank_metric), reverse=True)
    top_idx = ranked_idx[:top_n]
    top_configs = [configs[i] for i in top_idx]

    val_results_for_top = _run_batch(top_configs, val_start, val_end, chain, holidays, max_workers)
    val_metrics_by_idx = dict(zip(top_idx, (r[0] for r in val_results_for_top)))

    # --- Anti-overfitting pipeline (engine.robustness), on train performance ---
    sweep_df = pd.DataFrame({"trades": train_trades})
    pnl_matrix = _build_pnl_matrix(train_trades)
    annotated = compute_sweep_dsr(sweep_df, pnl_matrix)

    dsr_ranked_idx = annotated.sort_values("dsr", ascending=False).index[: max(top_n_for_pbo, 0)]
    pbo_pnl_matrix = pnl_matrix[dsr_ranked_idx].dropna()
    if len(pbo_pnl_matrix) < s_splits:
        raise ValueError(
            f"Only {len(pbo_pnl_matrix)} cycles have data for every one of the top "
            f"{len(dsr_ranked_idx)} configs by dsr (need >= s_splits={s_splits} for CSCV) -- "
            "these configs' reentry schedules diverge too much to compare on a shared time axis."
        )
    pbo_dict = compute_pbo(pbo_pnl_matrix, s_splits=s_splits)
    pbo_result = PBOResult(
        pbo=pbo_dict["pbo"],
        sweep_risk_flag=pbo_dict["sweep_risk_flag"],
        is_oos_pairs=pbo_dict["is_oos_pairs"],
    )

    rows = []
    for i in ranked_idx:
        config = configs[i]
        row = _flatten_config(config)
        tm = train_metrics[i]
        for name in _METRIC_FIELD_NAMES:
            row[f"train_{name}"] = getattr(tm, name)
        vm = val_metrics_by_idx.get(i)
        for name in _METRIC_FIELD_NAMES:
            row[f"validation_{name}"] = getattr(vm, name) if vm is not None else None
        for name in _ROBUSTNESS_FIELD_NAMES:
            row[name] = annotated.loc[i, name]
        row["recommended"] = annotated.loc[i, "robustness_flag"] != "red"
        rows.append(row)

    return pd.DataFrame(rows), pbo_result
