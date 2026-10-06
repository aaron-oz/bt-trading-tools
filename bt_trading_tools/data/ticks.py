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

KNOWN TRAP (reserves vs price): BacktestEngine fills from the pool RESERVES
(``tao_pool``, ``alpha_pool``), not from ``SubnetTick.price``. The parquet
source pairs an HOURLY close with a DAILY reserve snapshot; carried forward,
the reserves lag the price by up to a day. Around surges the pool-implied
price averaged 0.92 x the tick price at entry (measured 2026-10 on 160 surge
episodes), i.e. entries priced about 8% below the tick price, which lifted the
mean 24-hour return on those same episodes from +7.8% (contemporaneous SDK
reserves) to +11.2%. ``load_parquet_ticks``
now rescales the reserves to the hourly close by default (``reserves="rescale"``);
this removes the price mismatch but not any change in pool DEPTH during the day.
Where depth matters (surges, large clips) use SDK ticks. SDK snapshots exist
from 2026-02-11 locally and, on bot-vps, as a 5-minute file back to
2025-08-18 (``/root/sdk_backfill/sdk_snapshots/sdk_5min.csv``, ends 2026-02-14).
Check any tick source with ``reserve_price_gap``. See docs/known_traps.md.

Hexagonal precision note: pandas 3.0 defaults to microsecond resolution
on datetime64; `astype("int64")` returns microseconds, not nanoseconds.
`_to_unix_seconds` handles both; do NOT use the `// 10**9` shortcut
which is 1000x off on us-precision data.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

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
    stamp_identity: bool = True,
) -> list:
    """Build TickData list from SDK pool-state CSV (15-minute cadence).

    Each unique timestamp becomes one TickData containing every subnet's
    SubnetTick at that timestamp. SDK data lives at
    `/root/sdk_backfill/sdk_snapshots/sdk_pool_state.csv` on bot-vps; pull
    locally via rsync or use the existing `/tmp/autobot_paper_data/` cache.

    Columns expected: timestamp, netuid, price, tao_in, alpha_in, and
    alpha_out (staked alpha, used only to detect re-registrations). Other
    SDK columns (block, emission, k, ...) are ignored.

    With ``stamp_identity`` (default) every SubnetTick gets a
    ``generation`` so BacktestEngine can close a position whose netuid was
    re-registered (see bt_trading_tools.utils.lifecycle).

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
        usecols=["timestamp", "netuid", "price", "tao_in", "alpha_in", "alpha_out"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Detect on the window plus a day of lead-in, BEFORE the zero-pool filter
    # below (a dying pool's last rows are part of the signal).
    lead = df[(df["timestamp"] >= start_ts - pd.Timedelta(days=1))
              & (df["timestamp"] <= end_ts)]
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
    if stamp_identity:
        from bt_trading_tools.utils.lifecycle import (
            find_reregistrations, stamp_generations)
        stamp_generations(ticks, find_reregistrations(lead))
    return ticks


# ── Reserve / price consistency ───────────────────────────────────────


def rescale_reserves_to_price(tao_pool: float, alpha_pool: float, price: float) -> tuple[float, float]:
    """Rescale constant-product reserves to a target price, keeping k = tao x alpha.

    Returns ``(tao', alpha')`` with ``tao' x alpha' = tao x alpha`` and
    ``tao' / alpha' = price``: ``tao' = sqrt(k x price)``, ``alpha' = sqrt(k / price)``.
    Fixes a price/reserve mismatch (reserves older than the price); it cannot
    recover a change in pool depth (k) since the reserve snapshot.
    """
    k = tao_pool * alpha_pool
    return math.sqrt(k * price), math.sqrt(k / price)


def reserve_price_gap(ticks: Iterable) -> dict:
    """How far the pool-implied price (tao_pool / alpha_pool) is from each tick's own price.

    The engine fills from the reserves, so a large gap means mispriced fills.
    Accepts a list of ``TickData``. Returns ``{"n", "median_abs_gap",
    "share_over_2pct", "share_over_5pct", "max_abs_gap"}`` over every
    (tick, subnet) with positive price and reserves. On the SDK feed the
    median is about 0.005% and under 1% of subnet-ticks exceed 5%
    (measured 2026-10); a median above about 1% or a share over 5% above
    about 10% suggests stale reserves.
    """
    gaps = []
    for t in ticks:
        for st in t.subnets.values():
            if st.price > 0 and st.alpha_pool > 0 and st.tao_pool > 0:
                gaps.append(abs(st.tao_pool / st.alpha_pool / st.price - 1.0))
    if not gaps:
        return {"n": 0, "median_abs_gap": float("nan"), "share_over_2pct": float("nan"),
                "share_over_5pct": float("nan"), "max_abs_gap": float("nan")}
    s = pd.Series(gaps)
    return {"n": int(len(s)), "median_abs_gap": float(s.median()), "share_over_2pct": float((s > 0.02).mean()),
            "share_over_5pct": float((s > 0.05).mean()), "max_abs_gap": float(s.max())}


# ── Parquet (hourly OHLCV + daily pool) → TickData ────────────────────


def load_parquet_ticks(
    start, end,
    ohlcv_parquet: Union[str, Path] = DEFAULT_OHLCV_HOURLY_PARQUET,
    pool_parquet: Union[str, Path] = DEFAULT_POOL_HISTORY_PARQUET,
    pool_tolerance_days: int = 2,
    drop_startup_mode: bool = True,
    stamp_identity: bool = True,
    reserves: str = "rescale",
) -> list:
    """Build TickData list from delegation_ohlcv_hourly + pool_history parquet.

    Use for OOS windows where SDK pool_state.csv doesn't reach (SDK starts
    2026-02-11). Hourly cadence vs SDK's 15-min; pool data is daily,
    forward-filled to each hourly tick via `pd.merge_asof`.

    ``reserves`` (default ``"rescale"``): the engine fills from reserves, and
    daily reserves lag the hourly close, so by default the daily snapshot
    supplies only the invariant k = tao x alpha and the reserves are rescaled to
    the hourly close (``rescale_reserves_to_price``). ``"raw"`` keeps the daily
    reserves unchanged (the behavior before 2026-10) and warns: fills are then
    mispriced whenever price moved since the snapshot, most severely around
    surges. Rescaling does not capture changes in pool depth; prefer SDK ticks
    where depth matters. Each tick's ``signals`` carries ``reserve_gap_raw``
    (raw implied price / close - 1) and ``reserve_age_s`` (age of the daily
    snapshot).

    Subnets are emitted only in hours where they traded (no forward fill of
    quiet hours), so a strategy that needs a subnet present every hour must
    handle gaps.

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

    With ``stamp_identity`` (default) every SubnetTick gets a
    ``generation``, detected on the UNFILTERED daily pool history (startup
    rows kept), so BacktestEngine can close a position whose netuid was
    re-registered. Dropping startup rows otherwise hides the event and turns
    it into an apparent price jump across a data gap.
    """
    from bt_trading_tools.backtest.types import SubnetTick, TickData

    if reserves not in ("rescale", "raw"):
        raise ValueError(f"reserves must be 'rescale' or 'raw', got {reserves!r}")
    if reserves == "raw":
        warnings.warn(
            "load_parquet_ticks(reserves='raw') pairs hourly prices with daily reserves; the engine fills "
            "from the reserves, so fills are mispriced whenever price moved since the snapshot (about 8% "
            "cheap at surge entries in 2026-10 measurements). Use the default reserves='rescale' or SDK "
            "ticks. See docs/known_traps.md.",
            RuntimeWarning, stacklevel=2,
        )

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
    pool_unfiltered = pool
    if drop_startup_mode:
        pool = pool[pool["startup_mode"] == False]
    pool = pool.assign(
        total_tao=pool["total_tao"] / 1e9,
        alpha_in_pool=pool["alpha_in_pool"] / 1e9,
    )
    pool = pool.assign(pool_ts=pool["unix_ts"])[["netuid", "unix_ts", "pool_ts", "total_tao", "alpha_in_pool"]].sort_values(
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
            tao_r, alpha_r, close = float(row["total_tao"]), float(row["alpha_in_pool"]), float(row["close"])
            gap_raw = tao_r / alpha_r / close - 1.0
            if reserves == "rescale":
                tao_r, alpha_r = rescale_reserves_to_price(tao_r, alpha_r, close)
            subnets[int(row["netuid"])] = SubnetTick(
                netuid=int(row["netuid"]),
                price=close,
                tao_pool=tao_r,
                alpha_pool=alpha_r,
                signals={"reserve_gap_raw": gap_raw,
                         "reserve_age_s": float(row["unix_ts"] - row["pool_ts"])},
            )
        ticks.append(TickData(timestamp=int(ts), subnets=subnets))
    if stamp_identity:
        from bt_trading_tools.utils.lifecycle import (
            reregistrations_from_pool_history, stamp_generations)
        # Pool data is daily: floor events to midnight so no hourly tick on
        # the event day is attributed to the dead subnet.
        stamp_generations(
            ticks, reregistrations_from_pool_history(pool_unfiltered),
            floor_to_day=True)
    return ticks


__all__ = [
    "DEFAULT_OHLCV_HOURLY_PARQUET",
    "DEFAULT_POOL_HISTORY_PARQUET",
    "DEFAULT_SDK_POOL_STATE_CSV",
    "to_unix_seconds",
    "coerce_to_utc_timestamp",
    "rescale_reserves_to_price",
    "reserve_price_gap",
    "load_sdk_ticks",
    "load_parquet_ticks",
]
