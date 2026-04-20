"""
Derived portfolio-level metrics from the v1 trade log.

Reads `portfolio_snapshot` records only. The equity curve is whatever the
bot reported — reconstruction fidelity depends on snapshot cadence and is
the bot's responsibility (one snapshot per evaluation tick is the contract).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bt_trading_tools.tracking.reader import iter_trade_log


def equity_series(
    log_path: str | Path,
    *,
    bot_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame of portfolio_snapshot records, sorted by timestamp.

    Columns: timestamp (UTC datetime), bot_id, capital_tao,
    positions_value_tao, total_equity_tao, realized_pnl_to_date_tao,
    open_positions_count.

    If `bot_id` is provided, filters to that bot.
    """
    rows = []
    for rec in iter_trade_log(log_path):
        if rec.get("record_type") != "portfolio_snapshot":
            continue
        if bot_id is not None and rec.get("bot_id") != bot_id:
            continue
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def drawdown_series(
    log_path: str | Path,
    *,
    bot_id: Optional[str] = None,
) -> pd.DataFrame:
    """Running-peak drawdown of total equity.

    Returns a DataFrame with columns: timestamp, total_equity_tao,
    running_peak_tao, drawdown_tao, drawdown_pct (negative or zero).
    """
    df = equity_series(log_path, bot_id=bot_id)
    if df.empty:
        return df
    df = df[["timestamp", "total_equity_tao"]].copy()
    df["running_peak_tao"] = df["total_equity_tao"].cummax()
    df["drawdown_tao"] = df["total_equity_tao"] - df["running_peak_tao"]
    df["drawdown_pct"] = np.where(
        df["running_peak_tao"] > 0,
        df["drawdown_tao"] / df["running_peak_tao"] * 100.0,
        0.0,
    )
    return df


def max_drawdown(
    log_path: str | Path,
    *,
    bot_id: Optional[str] = None,
) -> dict[str, float | pd.Timestamp | None]:
    """Worst peak-to-trough drawdown in the equity series.

    Returns a dict with: max_drawdown_tao (<=0), max_drawdown_pct (<=0),
    peak_timestamp, trough_timestamp. Empty log returns all-None values.
    """
    dd = drawdown_series(log_path, bot_id=bot_id)
    if dd.empty:
        return {
            "max_drawdown_tao": None,
            "max_drawdown_pct": None,
            "peak_timestamp": None,
            "trough_timestamp": None,
        }
    idx = dd["drawdown_tao"].idxmin()
    trough_ts = dd.loc[idx, "timestamp"]
    # Peak is the running peak at the trough point; find the last time equity equaled that peak
    peak_val = dd.loc[idx, "running_peak_tao"]
    peak_rows = dd[(dd["total_equity_tao"] >= peak_val - 1e-12) & (dd["timestamp"] <= trough_ts)]
    peak_ts = peak_rows["timestamp"].iloc[-1] if not peak_rows.empty else None
    return {
        "max_drawdown_tao": float(dd.loc[idx, "drawdown_tao"]),
        "max_drawdown_pct": float(dd.loc[idx, "drawdown_pct"]),
        "peak_timestamp": peak_ts,
        "trough_timestamp": trough_ts,
    }


def sharpe(
    log_path: str | Path,
    *,
    bot_id: Optional[str] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """Annualized Sharpe ratio computed from tick-to-tick equity returns.

    Returns None when there are fewer than 2 snapshots or when return
    variance is zero.

    `periods_per_year` should match the snapshot cadence:
        - 252  ≈ daily returns
        - 252 * 6.5 ≈ hourly during equity market hours (use only if you
          have data aligned to that)
        - For a 5-min paper-trading tick running 24/7:
              periods_per_year = 365 * 24 * 12 = 105_120

    `risk_free_rate` is an annualized rate subtracted from annualized return
    before dividing by annualized volatility.
    """
    df = equity_series(log_path, bot_id=bot_id)
    if len(df) < 2:
        return None
    eq = df["total_equity_tao"].astype(float).to_numpy()
    # Guard against zero or negative equity entries
    if np.any(eq <= 0):
        return None
    returns = np.diff(eq) / eq[:-1]
    if returns.size == 0 or np.std(returns, ddof=1) == 0.0:
        return None
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    ann_ret = mean_ret * periods_per_year
    ann_vol = std_ret * math.sqrt(periods_per_year)
    return (ann_ret - risk_free_rate) / ann_vol
