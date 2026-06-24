"""TickData loaders for BacktestEngine.

Generic Bittensor-trading data loaders that convert source CSV/parquet
into `list[TickData]` for `BacktestEngine`. Public-safe: just data
plumbing, no strategy or universe logic.

Three sources supported:
  * SDK pool state CSV (15-minute cadence; written by the SDK backfill
    cron at `bot-vps:/root/sdk_backfill/sdk_snapshots/sdk_pool_state.csv`
    and synced locally to `/tmp/autobot_paper_data/`).
  * Delegation OHLCV parquet (hourly; derived from delegation events).
  * Pool history parquet (daily/4-hourly; per-subnet AMM state snapshots).

The SDK source is the trust anchor: paper-bot ground-truth alignment is
calibrated against it. The parquet sources cover a longer history
(back to 2025-02) and are the only option for OOS windows pre-2026-02.
See `docs/realistic_backtesting_guide.md` for the source-choice
guidance.

Hexagonal precision note: pandas 3.0 defaults to microsecond resolution
on datetime64; `astype("int64")` returns microseconds, not nanoseconds.
`_to_unix_seconds` handles both; do NOT use the `// 10**9` shortcut
which is 1000x off on us-precision data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


# ── Canonical paths (override-able) ───────────────────────────────────

_PARQUET_ROOT = "/var/home/aoz/data/taostats_parquet"
DEFAULT_OHLCV_HOURLY_PARQUET = f"{_PARQUET_ROOT}/delegation_ohlcv_hourly.parquet"
DEFAULT_POOL_HISTORY_PARQUET = f"{_PARQUET_ROOT}/pool_history.parquet"
DEFAULT_SDK_POOL_STATE_CSV = "/tmp/autobot_paper_data/sdk_pool_state.csv"


# ── Helpers ───────────────────────────────────────────────────────────


def to_unix_seconds(s: pd.Series) -> pd.Series:
    """Convert a datetime Series to unix seconds, handling Pandas 3.0's
    default microsecond precision (astype int64 returns microseconds, not
    nanoseconds on `datetime64[us]`).

    Memory: this is a recurring footgun. Three different research scripts
    used the buggy `// 10**6 // 1000` shortcut (effectively // 10**9) and
    were silently off by 1000x on us-precision data, breaking days-based
    annualization. Centralize here.
    """
    s = pd.to_datetime(s, utc=True)
    precision_div = 10**9 if "[ns" in str(s.dtype) else 10**6
    return (s.astype("int64") // precision_div).astype("int64")


def coerce_to_utc_timestamp(t) -> pd.Timestamp:
    """Accept str, datetime (naive or tz-aware), or pd.Timestamp. Return
    UTC tz-aware pd.Timestamp.

    Avoids the recurring `Cannot pass a datetime ... with tzinfo with the
    tz parameter` error that comes from `pd.Timestamp(dt, tz='UTC')` when
    `dt` already has tzinfo.
    """
    ts = pd.Timestamp(t)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# ── SDK CSV → TickData ────────────────────────────────────────────────


def load_sdk_ticks(
    start, end,
    csv_path: Union[str, Path] = DEFAULT_SDK_POOL_STATE_CSV,
) -> list:
    """Build TickData list from SDK pool-state CSV (15-minute cadence).

    Each unique timestamp becomes one TickData containing every subnet's
    SubnetTick at that timestamp. SDK data lives at
    `/root/sdk_backfill/sdk_snapshots/sdk_pool_state.csv` on bot-vps; pull
    locally via rsync or use the existing `/tmp/autobot_paper_data/` cache.

    Columns expected: timestamp, netuid, price, tao_in, alpha_in. Other
    SDK columns (block, emission, k, ...) are ignored.

    Returns
    -------
    list[TickData]
        Empty if no rows in window. Order is timestamp-ascending.
    """
    from bt_trading_tools.backtest.types import SubnetTick, TickData

    start_ts = coerce_to_utc_timestamp(start)
    end_ts = coerce_to_utc_timestamp(end)
    df = pd.read_csv(
        csv_path,
        usecols=["timestamp", "netuid", "price", "tao_in", "alpha_in"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    df = df[(df["price"] > 0) & (df["tao_in"] > 0) & (df["alpha_in"] > 0)]
    df["unix_ts"] = to_unix_seconds(df["timestamp"])

    ticks = []
    for ts, group in df.sort_values("unix_ts").groupby("unix_ts", sort=True):
        subnets = {}
        for _, row in group.iterrows():
            subnets[int(row["netuid"])] = SubnetTick(
                netuid=int(row["netuid"]),
                price=float(row["price"]),
                tao_pool=float(row["tao_in"]),
                alpha_pool=float(row["alpha_in"]),
                signals={},
            )
        ticks.append(TickData(timestamp=int(ts), subnets=subnets))
    return ticks


# ── Parquet (hourly OHLCV + daily pool) → TickData ────────────────────


def load_parquet_ticks(
    start, end,
    ohlcv_parquet: Union[str, Path] = DEFAULT_OHLCV_HOURLY_PARQUET,
    pool_parquet: Union[str, Path] = DEFAULT_POOL_HISTORY_PARQUET,
    pool_tolerance_days: int = 2,
    drop_startup_mode: bool = True,
) -> list:
    """Build TickData list from delegation_ohlcv_hourly + pool_history parquet.

    Use for OOS windows where SDK pool_state.csv doesn't reach (SDK starts
    2026-02-11). Hourly cadence vs SDK's 15-min; pool data is daily,
    forward-filled to each hourly tick via `pd.merge_asof`.

    `pool_history.parquet` stores `total_tao` and `alpha_in_pool` in RAO;
    this loader converts to TAO / alpha-tokens before building ticks.

    Returns
    -------
    list[TickData]
        Empty if no rows in window. Order is timestamp-ascending. Subnets
        with no pool data within `pool_tolerance_days` of a tick are
        skipped for that tick.

    Notes
    -----
    Per the 2026-06-12 backtest-vs-paper-bot alignment finding, this
    loader's output produces slightly different decisions than SDK-fed
    backtests (data-source artifact, not engine bug). Prefer SDK for
    windows that overlap paper-bot operation; parquet only for
    long-history OOS work.
    """
    from bt_trading_tools.backtest.types import SubnetTick, TickData

    start_ts = coerce_to_utc_timestamp(start)
    end_ts = coerce_to_utc_timestamp(end)

    ohlcv = pd.read_parquet(ohlcv_parquet)
    ohlcv["unix_ts"] = to_unix_seconds(ohlcv["time"])
    ohlcv = ohlcv[
        (ohlcv["unix_ts"] >= start_ts.timestamp())
        & (ohlcv["unix_ts"] <= end_ts.timestamp())
    ]

    pool = pd.read_parquet(pool_parquet)
    pool["unix_ts"] = to_unix_seconds(pool["timestamp"])
    pool = pool[
        (pool["unix_ts"] >= start_ts.timestamp() - 86400)
        & (pool["unix_ts"] <= end_ts.timestamp())
    ]
    if drop_startup_mode:
        pool = pool[pool["startup_mode"] == False]
    pool = pool.assign(
        total_tao=pool["total_tao"] / 1e9,
        alpha_in_pool=pool["alpha_in_pool"] / 1e9,
    )
    pool = pool[["netuid", "unix_ts", "total_tao", "alpha_in_pool"]].sort_values(
        ["netuid", "unix_ts"]
    )

    merged = pd.merge_asof(
        ohlcv.sort_values("unix_ts"),
        pool.sort_values("unix_ts"),
        on="unix_ts",
        by="netuid",
        direction="backward",
        tolerance=pool_tolerance_days * 86400,
    )
    merged = merged.dropna(subset=["total_tao", "alpha_in_pool"])
    merged = merged[(merged["close"] > 0) & (merged["total_tao"] > 0) & (merged["alpha_in_pool"] > 0)]

    ticks = []
    for ts, group in merged.groupby("unix_ts", sort=True):
        subnets = {}
        for _, row in group.iterrows():
            subnets[int(row["netuid"])] = SubnetTick(
                netuid=int(row["netuid"]),
                price=float(row["close"]),
                tao_pool=float(row["total_tao"]),
                alpha_pool=float(row["alpha_in_pool"]),
                signals={},
            )
        ticks.append(TickData(timestamp=int(ts), subnets=subnets))
    return ticks


__all__ = [
    "DEFAULT_OHLCV_HOURLY_PARQUET",
    "DEFAULT_POOL_HISTORY_PARQUET",
    "DEFAULT_SDK_POOL_STATE_CSV",
    "to_unix_seconds",
    "coerce_to_utc_timestamp",
    "load_sdk_ticks",
    "load_parquet_ticks",
]
