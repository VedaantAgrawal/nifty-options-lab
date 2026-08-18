"""Tests for engine.sweep.

run_sweep tests use a tiny synthetic chain (no network/cache dependency)
and a small max_workers so the ProcessPoolExecutor pool stays cheap to
spin up, while still exercising genuine multiprocessing rather than
mocking it away.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from data.expiry_calendar import HolidayCalendar
from engine.sweep import generate_configs, run_sweep, run_trade_cycle_loop
from models.schemas import ParameterGrid, StrategyConfig

COLUMNS = ["date", "expiry_date", "strike", "option_type", "close", "oi", "volume"]


def make_chain_row(d, expiry, strike, option_type, close):
    return {
        "date": d,
        "expiry_date": expiry,
        "strike": strike,
        "option_type": option_type,
        "close": close,
        "oi": 1000,
        "volume": 500,
    }


def build_synthetic_chain(cycles, spot=24700.0, strikes=range(23500, 26000, 50)):
    """cycles: list of (cycle_start, cycle_expiry) date pairs. Builds one row
    per weekday in each cycle for every strike/option_type, using a flat
    (non-random) premium curve so results are deterministic."""
    rows = []
    for cycle_start, expiry in cycles:
        d = cycle_start
        while d <= expiry:
            if d.weekday() < 5:
                for strike in strikes:
                    base = 20.0 + abs(spot - strike) * 0.02
                    rows.append(make_chain_row(d, expiry, strike, "CE", base))
                    rows.append(make_chain_row(d, expiry, strike, "PE", base))
            d += timedelta(days=1)
    return pd.DataFrame(rows, columns=COLUMNS)


# Tuesday weekly-expiry regime (post-2-Sept-2025). Three consecutive weekly
# cycles: entry Wednesday, expiry the following Tuesday.
CYCLES = [
    (date(2025, 9, 3), date(2025, 9, 9)),
    (date(2025, 9, 10), date(2025, 9, 16)),
    (date(2025, 9, 17), date(2025, 9, 23)),
]
TRAIN_RANGE = (date(2025, 9, 3), date(2025, 9, 16))
VALIDATION_RANGE = (date(2025, 9, 17), date(2025, 9, 23))


class TestGenerateConfigs:
    def test_cartesian_product_size_before_filtering(self):
        grid = ParameterGrid(
            structure=["short_strangle", "iron_condor"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[2],
            days_to_expiry_at_entry=[4],
            otm_points_call=[200, 400],
            otm_points_put=[200],
            wing_width_points=[None, 100],
            stop_loss_pct=[30],
            capital=[1_000_000],
            reentry=["next_cycle"],
        )
        configs, num_invalid = generate_configs(grid)
        total_combos = 2 * 1 * 1 * 1 * 2 * 1 * 2 * 1 * 1 * 1
        assert len(configs) + num_invalid == total_combos

    def test_invalid_structure_wing_width_combos_are_filtered_not_raised(self):
        grid = ParameterGrid(
            structure=["short_strangle", "iron_condor"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[2],
            days_to_expiry_at_entry=[4],
            otm_points_call=[200],
            otm_points_put=[200],
            wing_width_points=[None, 100],
            stop_loss_pct=[30],
            capital=[1_000_000],
            reentry=["next_cycle"],
        )
        configs, num_invalid = generate_configs(grid)
        # valid: (short_strangle, None) and (iron_condor, 100) -- the other
        # two combinations fail StrategyConfig's cross-field validation.
        assert len(configs) == 2
        assert num_invalid == 2
        assert all(
            (c.structure == "short_strangle") == (c.wing_width_points is None)
            for c in configs
        )

    def test_2x2_grid_produces_four_valid_configs(self):
        grid = ParameterGrid(
            structure=["short_strangle"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[2],
            days_to_expiry_at_entry=[4],
            otm_points_call=[200, 400],
            otm_points_put=[200],
            stop_loss_pct=[30, 50],
            capital=[1_000_000],
            reentry=["next_cycle"],
        )
        configs, num_invalid = generate_configs(grid)
        assert len(configs) == 4
        assert num_invalid == 0
        combos = {(c.otm_points_call, c.stop_loss_pct) for c in configs}
        assert combos == {(200, 30), (200, 50), (400, 30), (400, 50)}


class TestRunTradeCycleLoop:
    def _config(self, **overrides):
        defaults = dict(
            structure="short_strangle",
            expiry_cycle="weekly",
            entry_day_of_week=2,
            days_to_expiry_at_entry=4,
            otm_points_call=200,
            otm_points_put=200,
            stop_loss_pct=30,
            capital=1_000_000,
            reentry="next_cycle",
        )
        defaults.update(overrides)
        return StrategyConfig(**defaults)

    def test_runs_one_trade_per_cycle_across_the_window(self):
        chain = build_synthetic_chain(CYCLES)
        config = self._config()
        trades, skipped = run_trade_cycle_loop(
            config, CYCLES[0][0], CYCLES[-1][1], chain, HolidayCalendar()
        )
        assert len(trades) == 3
        assert skipped == []

    def test_stops_before_a_cycle_whose_expiry_data_is_incomplete(self):
        chain = build_synthetic_chain(CYCLES)
        # Chop off the chain's last cycle entirely, simulating "not published yet".
        incomplete_chain = chain[chain["date"] < CYCLES[-1][0]]
        config = self._config()
        trades, skipped = run_trade_cycle_loop(
            config, CYCLES[0][0], CYCLES[-1][1], incomplete_chain, HolidayCalendar()
        )
        assert len(trades) == 2  # only the two fully-covered cycles

    def test_reentry_none_stops_after_first_trade(self):
        chain = build_synthetic_chain(CYCLES)
        config = self._config(reentry="none")
        trades, skipped = run_trade_cycle_loop(
            config, CYCLES[0][0], CYCLES[-1][1], chain, HolidayCalendar()
        )
        assert len(trades) == 1


class TestRunSweep:
    def _grid(self):
        return ParameterGrid(
            structure=["short_strangle"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[2],
            days_to_expiry_at_entry=[4],
            otm_points_call=[200, 400],
            otm_points_put=[200],
            stop_loss_pct=[30, 50],
            capital=[1_000_000],
            reentry=["next_cycle"],
        )

    def test_returns_one_row_per_valid_config(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=2, max_workers=2)
        assert len(df) == 4  # tiny 2x2 grid, all combos valid (short_strangle only)

    def test_config_columns_and_metric_columns_present(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=2, max_workers=2)
        for col in ["structure", "otm_points_call", "stop_loss_pct"]:
            assert col in df.columns
        for prefix in ("train_", "validation_"):
            assert f"{prefix}sharpe_ratio" in df.columns
            assert f"{prefix}num_trades" in df.columns
        assert "equity_curve" not in df.columns
        assert "train_equity_curve" not in df.columns

    def test_sorted_by_train_rank_metric_descending(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=2, max_workers=2)
        values = df["train_sharpe_ratio"].tolist()
        assert values == sorted(values, reverse=True)

    def test_only_top_n_configs_get_validation_metrics(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=2, max_workers=2)
        assert df["validation_num_trades"].notna().sum() == 2
        assert df["validation_num_trades"].isna().sum() == 2
        # the validated rows should be exactly the top 2 by train rank
        assert df.head(2)["validation_num_trades"].notna().all()
        assert df.tail(2)["validation_num_trades"].isna().all()

    def test_validation_trade_count_matches_the_single_validation_cycle(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=4, max_workers=2)
        # VALIDATION_RANGE covers exactly one weekly cycle.
        assert (df["validation_num_trades"] == 1).all()
        # TRAIN_RANGE covers exactly two weekly cycles.
        assert (df["train_num_trades"] == 2).all()

    def test_invalid_rank_metric_raises(self):
        chain = build_synthetic_chain(CYCLES)
        with pytest.raises(ValueError):
            run_sweep(self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, rank_metric="not_a_real_field")

    def test_custom_rank_metric_is_respected(self):
        chain = build_synthetic_chain(CYCLES)
        df = run_sweep(
            self._grid(), TRAIN_RANGE, VALIDATION_RANGE, chain=chain, top_n=2, max_workers=2,
            rank_metric="total_return_abs",
        )
        values = df["train_total_return_abs"].tolist()
        assert values == sorted(values, reverse=True)

    def test_empty_grid_after_filtering_raises(self):
        grid = ParameterGrid(
            structure=["iron_condor"],
            expiry_cycle=["weekly"],
            entry_day_of_week=[2],
            days_to_expiry_at_entry=[4],
            otm_points_call=[200],
            otm_points_put=[200],
            wing_width_points=[None],  # every combo invalid: iron_condor needs a width
            stop_loss_pct=[30],
            capital=[1_000_000],
            reentry=["next_cycle"],
        )
        chain = build_synthetic_chain(CYCLES)
        with pytest.raises(ValueError):
            run_sweep(grid, TRAIN_RANGE, VALIDATION_RANGE, chain=chain)
