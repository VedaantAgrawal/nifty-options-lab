"""NiftyCorridor -- a NIFTY50 options backtester.

Streamlit UI wrapping the existing, already-tested backtest engine (data
layer, position builder, simulator, metrics, robustness/DSR/PBO, sweep
engine). No engine or data logic lives here -- this only wires existing
functions to a UI. See README.md for what each engine module does.

Run locally: streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.expiry_calendar import HolidayCalendar  # noqa: E402
from data.providers import NSEBhavcopyProvider  # noqa: E402
from engine.metrics import MetricsResult, compute_metrics  # noqa: E402
from engine.sweep import run_sweep, run_trade_cycle_loop  # noqa: E402
from models.schemas import ParameterGrid, StrategyConfig, Trade  # noqa: E402

APP_NAME = "NiftyCorridor"

STRUCTURE_OPTIONS = ["short_strangle", "iron_condor"]
EXPIRY_CYCLE_OPTIONS = ["weekly", "monthly"]
REENTRY_OPTIONS = ["immediate", "next_cycle", "none"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# StrategyConfig's own field names, in constructor order -- used both to
# build a config from form values and to reconstruct one from a sweep
# leaderboard row (kept local to the UI layer rather than importing
# engine.sweep's private _GRID_FIELD_NAMES across a module boundary).
CONFIG_FIELDS = [
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

st.set_page_config(page_title=APP_NAME, page_icon="\U0001F4C8", layout="wide")


# --- Cached resources ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_chain(start: date, end: date) -> pd.DataFrame:
    """Cached so switching strategy parameters and re-running doesn't
    re-read the underlying parquet cache from disk every time -- only a
    changed date range invalidates this."""
    provider = NSEBhavcopyProvider()
    return provider.get_options_chain(start=start, end=end, symbol="NIFTY")


@st.cache_resource(show_spinner=False)
def load_holidays() -> HolidayCalendar:
    return HolidayCalendar.from_csv()


# --- Sidebar: single-run form --------------------------------------------
def _single_run_form() -> tuple[StrategyConfig, date, date]:
    st.sidebar.subheader("Strategy")
    structure = st.sidebar.selectbox("Structure", STRUCTURE_OPTIONS)
    expiry_cycle = st.sidebar.selectbox("Expiry cycle", EXPIRY_CYCLE_OPTIONS)
    entry_day_of_week = WEEKDAY_NAMES.index(
        st.sidebar.selectbox("Entry day of week", WEEKDAY_NAMES, index=2)
    )
    days_to_expiry_at_entry = st.sidebar.number_input(
        "Days to expiry at entry", min_value=0, value=4, step=1
    )
    otm_points_call = st.sidebar.number_input(
        "OTM points (call)", min_value=50, value=200, step=50
    )
    otm_points_put = st.sidebar.number_input(
        "OTM points (put)", min_value=50, value=200, step=50
    )
    wing_width_points: Optional[float] = None
    if structure == "iron_condor":
        wing_width_points = st.sidebar.number_input(
            "Wing width points", min_value=50, value=100, step=50
        )
    stop_loss_pct = st.sidebar.number_input(
        "Stop-loss %", min_value=1.0, value=30.0, step=5.0
    )
    capital = st.sidebar.number_input(
        "Capital (Rs)", min_value=1000.0, value=1_000_000.0, step=50_000.0
    )
    reentry = st.sidebar.selectbox("Reentry", REENTRY_OPTIONS, index=1)

    st.sidebar.subheader("Date range")
    default_end = date.today()
    default_start = default_end - timedelta(days=180)
    start = st.sidebar.date_input("Start", value=default_start)
    end = st.sidebar.date_input("End", value=default_end)

    config = StrategyConfig(
        structure=structure,
        expiry_cycle=expiry_cycle,
        entry_day_of_week=entry_day_of_week,
        days_to_expiry_at_entry=int(days_to_expiry_at_entry),
        otm_points_call=float(otm_points_call),
        otm_points_put=float(otm_points_put),
        wing_width_points=float(wing_width_points) if wing_width_points is not None else None,
        stop_loss_pct=float(stop_loss_pct),
        capital=float(capital),
        reentry=reentry,
    )
    return config, start, end


def _range_values(label: str, default_min: float, default_max: float, default_step: float, key: str) -> List[float]:
    """Min/max/step trio -> the list of values it spans, for sweep fields
    that live on a numeric grid (OTM points, stop-loss %, capital, ...).

    All three number_inputs are forced to float -- Streamlit's number_input
    requires value/min_value/step to share exactly one numeric type, and
    mixing an int default_step with a float min_value raises at runtime.
    """
    default_min, default_max, default_step = float(default_min), float(default_max), float(default_step)
    c1, c2, c3 = st.sidebar.columns(3)
    lo = c1.number_input("min", value=default_min, step=default_step, key=f"{key}_min")
    hi = c2.number_input("max", value=default_max, step=default_step, key=f"{key}_max")
    step = c3.number_input("step", value=default_step, min_value=1e-6, step=default_step, key=f"{key}_step")
    st.sidebar.caption(label)
    values = []
    v = lo
    while v <= hi + 1e-9:
        values.append(round(v, 4))
        v += step
    return values or [lo]


def _sweep_form() -> tuple[ParameterGrid, date, date, date, date]:
    st.sidebar.subheader("Strategy grid")
    structures = st.sidebar.multiselect("Structure", STRUCTURE_OPTIONS, default=STRUCTURE_OPTIONS)
    if not structures:
        structures = [STRUCTURE_OPTIONS[0]]
    expiry_cycles = st.sidebar.multiselect("Expiry cycle", EXPIRY_CYCLE_OPTIONS, default=["weekly"])
    if not expiry_cycles:
        expiry_cycles = ["weekly"]
    entry_days = st.sidebar.multiselect(
        "Entry day of week", WEEKDAY_NAMES, default=[WEEKDAY_NAMES[2]]
    )
    entry_days_of_week = [WEEKDAY_NAMES.index(d) for d in entry_days] or [2]
    days_to_expiry_values = _range_values("Days to expiry at entry", 4, 4, 1, "dte")
    otm_call_values = _range_values("OTM points (call)", 150, 300, 50, "otm_call")
    otm_put_values = _range_values("OTM points (put)", 150, 300, 50, "otm_put")

    wing_width_values: List[Optional[float]] = [None]
    if "iron_condor" in structures:
        widths = _range_values("Wing width points (iron_condor only)", 100, 150, 50, "wing")
        wing_width_values = widths if "short_strangle" not in structures else [None, *widths]

    stop_loss_values = _range_values("Stop-loss %", 30, 50, 10, "sl")
    capital_values = [st.sidebar.number_input("Capital (Rs)", min_value=1000.0, value=1_000_000.0, step=50_000.0)]
    reentry_values = st.sidebar.multiselect("Reentry", REENTRY_OPTIONS, default=["next_cycle"])
    if not reentry_values:
        reentry_values = ["next_cycle"]

    st.sidebar.subheader("Train / validation split")
    default_end = date.today()
    default_val_start = default_end - timedelta(days=60)
    default_train_start = default_end - timedelta(days=240)
    train_start = st.sidebar.date_input("Train start", value=default_train_start)
    train_end = st.sidebar.date_input("Train end", value=default_val_start - timedelta(days=1))
    val_start = st.sidebar.date_input("Validation start", value=default_val_start)
    val_end = st.sidebar.date_input("Validation end", value=default_end)

    grid = ParameterGrid(
        structure=structures,
        expiry_cycle=expiry_cycles,
        entry_day_of_week=entry_days_of_week,
        days_to_expiry_at_entry=[int(v) for v in days_to_expiry_values],
        otm_points_call=otm_call_values,
        otm_points_put=otm_put_values,
        wing_width_points=wing_width_values,
        stop_loss_pct=stop_loss_values,
        capital=capital_values,
        reentry=reentry_values,
    )
    return grid, train_start, train_end, val_start, val_end


# --- Result rendering ------------------------------------------------------
def _trade_to_row(trade: Trade) -> dict:
    legs_summary = " | ".join(
        f"{leg.side} {leg.option_type} {leg.strike:.0f} ({leg.entry_price:.2f}->{leg.exit_price:.2f})"
        for leg in trade.legs
    )
    return {
        "entry_date": trade.entry_date,
        "expiry_date": trade.expiry_date,
        "exit_date": trade.exit_date,
        "exit_reason": trade.exit_reason,
        "legs": legs_summary,
        "pnl": round(trade.pnl, 2),
        "capital_at_risk": round(trade.capital_at_risk, 2),
    }


def _render_summary_tab(metrics: MetricsResult) -> None:
    cols = st.columns(5)
    cols[0].metric("Total Return", f"{metrics.total_return_pct:.2f}%", f"Rs {metrics.total_return_abs:,.0f}")
    cols[1].metric("Win Rate", f"{metrics.win_rate:.1f}%")
    cols[2].metric("Max Drawdown", f"{metrics.max_drawdown * 100:.2f}%")
    cols[3].metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    cols[4].metric("Avg P&L / Trade", f"Rs {metrics.avg_pnl_per_trade:,.0f}")
    st.caption(
        f"{metrics.num_trades} trades closed | "
        f"{metrics.num_skipped_insufficient_margin} skipped for insufficient margin"
    )


def _render_equity_curve_tab(metrics: MetricsResult) -> None:
    if not metrics.equity_curve:
        st.info("No trades closed in this window -- nothing to plot.")
        return
    equity_df = pd.DataFrame(metrics.equity_curve, columns=["date", "capital"]).set_index("date")
    st.line_chart(equity_df)


def _render_trade_log_tab(trades: List[Trade]) -> None:
    if not trades:
        st.info("No trades closed in this window.")
        return
    st.dataframe(pd.DataFrame([_trade_to_row(t) for t in trades]), width="stretch")


def _render_single_run_results(config: StrategyConfig, trades: List[Trade], metrics: MetricsResult) -> None:
    tabs = st.tabs(["Summary", "Equity Curve", "Trade Log"])
    with tabs[0]:
        _render_summary_tab(metrics)
    with tabs[1]:
        _render_equity_curve_tab(metrics)
    with tabs[2]:
        _render_trade_log_tab(trades)


_FLAG_BADGE_STYLE = {
    "green": "background-color: #d4edda; color: #155724; font-weight: 600;",
    "amber": "background-color: #fff3cd; color: #856404; font-weight: 600;",
    "red": "background-color: #f8d7da; color: #721c24; font-weight: 600;",
}
_RED_ROW_STYLE = "color: #999999; background-color: #fafafa;"


def _style_leaderboard(df: pd.DataFrame):
    def _style_row(row: pd.Series) -> List[str]:
        # Deprioritize (grey out), don't hide: a red-flagged config stays in
        # its rank position and fully visible, just visually muted so it
        # doesn't read as a good pick.
        base = _RED_ROW_STYLE if row.get("robustness_flag") == "red" else ""
        return [base] * len(row)

    def _badge(val: object) -> str:
        return _FLAG_BADGE_STYLE.get(val, "")

    return df.style.apply(_style_row, axis=1).map(_badge, subset=["robustness_flag"])


def _explain_dsr_and_pbo() -> None:
    with st.expander("What does this mean?"):
        st.markdown(
            "**DSR (Deflated Sharpe Ratio)** asks: if you tried this many different "
            "strategy variations, how likely is it that this particular one's good "
            "performance is real skill rather than luck? The more variations you "
            "tried, the higher the bar for \"real skill\" gets. A DSR near 1.0 means "
            "the result is unlikely to be a fluke; a DSR below 0.90 (flagged red) "
            "means it's quite plausible this config just got lucky."
        )
        st.markdown(
            "**PBO (Probability of Backtest Overfitting)** checks whether picking "
            "the best-looking strategy from part of the data actually holds up on "
            "the rest of the data. If PBO is high, whatever looked best in one "
            "period tends to look mediocre or worse in another -- a sign the sweep "
            "is fitting noise, not finding a real edge. PBO above 0.50 (red) means "
            "picking a \"winner\" here is worse than a coin flip at predicting real "
            "future performance."
        )


def _render_sweep_leaderboard(df: pd.DataFrame, pbo_pbo: float, pbo_flag: str) -> None:
    banner = f"This sweep's selection reliability: PBO = {pbo_pbo:.2f} — {pbo_flag.upper()}"
    if pbo_flag == "green":
        st.success(banner)
    elif pbo_flag == "amber":
        st.warning(banner)
    else:
        st.error(banner)

    _explain_dsr_and_pbo()

    display_cols = [
        c for c in CONFIG_FIELDS if c != "wing_width_points" or (df["wing_width_points"].notna().any())
    ] + [
        "train_total_return_pct", "train_win_rate", "train_max_drawdown", "train_sharpe_ratio", "train_num_trades",
        "validation_total_return_pct", "validation_win_rate", "validation_max_drawdown",
        "validation_sharpe_ratio", "validation_num_trades",
        "dsr", "n_eff_trials", "robustness_flag", "recommended",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(_style_leaderboard(df[display_cols]), width="stretch", height=500)


def _config_from_leaderboard_row(row: pd.Series) -> StrategyConfig:
    values = {name: row[name] for name in CONFIG_FIELDS}
    if pd.isna(values.get("wing_width_points")):
        values["wing_width_points"] = None
    values["entry_day_of_week"] = int(values["entry_day_of_week"])
    values["days_to_expiry_at_entry"] = int(values["days_to_expiry_at_entry"])
    return StrategyConfig(**values)


def _row_label(i: int, row: pd.Series) -> str:
    flag = row.get("robustness_flag", "?")
    return (
        f"#{i + 1} [{flag}] {row['structure']} "
        f"OTM {row['otm_points_call']:.0f}/{row['otm_points_put']:.0f} "
        f"SL {row['stop_loss_pct']:.0f}%"
    )


# --- Main -------------------------------------------------------------------
def main() -> None:
    st.title(f"\U0001F4C8 {APP_NAME}")
    st.caption("NIFTY50 options backtester")

    sweep_mode = st.sidebar.toggle("Sweep mode (ParameterGrid)", value=False)

    if sweep_mode:
        grid, train_start, train_end, val_start, val_end = _sweep_form()
    else:
        config, start, end = _single_run_form()

    run_clicked = st.sidebar.button("Run Backtest", type="primary", width="stretch")

    if run_clicked:
        try:
            if sweep_mode:
                with st.spinner("Loading data and running sweep..."):
                    chain = load_chain(min(train_start, val_start), max(train_end, val_end))
                    holidays = load_holidays()
                    df, pbo_result = run_sweep(
                        grid,
                        (train_start, train_end),
                        (val_start, val_end),
                        chain=chain,
                        holidays=holidays,
                    )
                st.session_state["mode"] = "sweep"
                st.session_state["sweep_df"] = df
                st.session_state["pbo_result"] = pbo_result
                st.session_state["sweep_train_range"] = (train_start, train_end)
                st.session_state["sweep_chain"] = chain
                st.session_state["sweep_holidays"] = holidays
            else:
                with st.spinner("Loading data and running backtest..."):
                    chain = load_chain(start, end)
                    holidays = load_holidays()
                    trades, skipped = run_trade_cycle_loop(config, start, end, chain, holidays)
                    metrics = compute_metrics(
                        trades, initial_capital=config.capital, num_skipped_insufficient_margin=len(skipped)
                    )
                st.session_state["mode"] = "single"
                st.session_state["single_config"] = config
                st.session_state["single_trades"] = trades
                st.session_state["single_metrics"] = metrics
        except Exception as exc:
            # Broad on purpose: this is a production-facing UI boundary, and
            # failures here are as likely to be an NSE network hiccup
            # (requests/BhavcopyDownloadError) as a bad parameter combo
            # (ValueError) -- either way the user should see a clean message,
            # not a raw traceback.
            st.error(f"Could not complete the run: {exc}")
            return

    mode = st.session_state.get("mode")
    if mode is None:
        st.info("Configure a strategy (or a sweep) in the sidebar and click **Run Backtest**.")
        return

    if mode == "single":
        _render_single_run_results(
            st.session_state["single_config"],
            st.session_state["single_trades"],
            st.session_state["single_metrics"],
        )
        return

    # mode == "sweep"
    df: pd.DataFrame = st.session_state["sweep_df"]
    pbo_result = st.session_state["pbo_result"]
    train_start, train_end = st.session_state["sweep_train_range"]
    chain = st.session_state["sweep_chain"]
    holidays = st.session_state["sweep_holidays"]

    tabs = st.tabs(["Summary", "Equity Curve", "Trade Log", "Leaderboard"])

    default_idx = 0
    if df["recommended"].any():
        default_idx = int(df.reset_index(drop=True)["recommended"].idxmax())
    row_labels = [_row_label(i, row) for i, row in df.reset_index(drop=True).iterrows()]

    with tabs[3]:
        _render_sweep_leaderboard(df, pbo_result.pbo, pbo_result.sweep_risk_flag)

    selected_label = st.selectbox(
        "Inspect a config from the leaderboard (Summary / Equity Curve / Trade Log below "
        "show this config's own train-range run)",
        row_labels,
        index=default_idx,
    )
    selected_row = df.reset_index(drop=True).iloc[row_labels.index(selected_label)]
    detail_config = _config_from_leaderboard_row(selected_row)
    detail_trades, detail_skipped = run_trade_cycle_loop(detail_config, train_start, train_end, chain, holidays)
    detail_metrics = compute_metrics(
        detail_trades, initial_capital=detail_config.capital, num_skipped_insufficient_margin=len(detail_skipped)
    )

    with tabs[0]:
        _render_summary_tab(detail_metrics)
    with tabs[1]:
        _render_equity_curve_tab(detail_metrics)
    with tabs[2]:
        _render_trade_log_tab(detail_trades)


if __name__ == "__main__":
    main()
