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

KNOWN TRAPS in the parquet source (measured 2026-10; docs/known_traps.md):

1. Reserves vs price. BacktestEngine fills from the pool RESERVES
   (``tao_pool``, ``alpha_pool``), not from ``SubnetTick.price``. The parquet
   source pairs an HOURLY close with a DAILY reserve snapshot; carried forward,
   the reserves lag the price by up to a day (median error vs the true pool
   price: 0.0% right at the snapshot, 1.7% at 18 to 24 h old; p90 7.9%).
   Around surges the pool-implied price averaged 0.92 x the tick price at entry
   (160 surge episodes), lifting a 24-hour mean return from +7.8% (contemporaneous
   SDK reserves) to +11.2%. ``reserves="rescale"`` makes the reserves match the
   hourly close at constant k, but it is NOT a validated fix (see 2 and 3): it
   leaves depth changes uncorrected and on active strategies moved results by
   orders of magnitude (an autobot fade study: +275% to +25,962%).
2. Bar label lookahead. The OHLCV ``time`` is the bar START, the ``close`` is the
   pool price at the bar END (measured: the close matches the true pool price one
   hour after the label, median error 0.11%, versus 0.27% at the label). The
   default ``bar_label="start"`` therefore gives every tick a price from one hour
   in the FUTURE. Harmless for buy-and-hold between midnight endpoints, a
   lookahead for any strategy that reads the price at the tick. ``bar_label="end"``
   stamps each tick with the bar end so the close is known at its timestamp.
3. Re-registrations. The daily pool-history detector misses re-registrations that
   the 15-minute SDK detector finds (D1 window: SN82; cheapest-10 holdout: SN69;
   D2: SN90), so those positions are valued against the NEW subnet's pool and show
   fabricated profit (+19.9 and +34.9 TAO on 100 TAO of capital in two cases).
   Parquet results over windows with re-registrations are not reliable.
Where fills or entries depend on the price path (surges, hourly signals, large
clips) use SDK ticks: local from 2026-02-11 and, on bot-vps, a 5-minute file back
to 2025-08-18 (``/root/sdk_backfill/sdk_snapshots/sdk_5min.csv``, ends 2026-02-14).
Check any tick source with ``reserve_price_gap``.

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


_PARQUET_WARNED = False


def rescale_reserves_to_price(tao_pool: float, alpha_pool: float, price: float) -> tuple[float, float]:
    """Rescale constant-product reserves to a target price, keeping k = tao x alpha.

    Returns ``(tao', alpha')`` with ``tao' x alpha' = tao x alpha`` and
    ``tao' / alpha' = price``: ``tao' = sqrt(k x price)``, ``alpha' = sqrt(k / price)``.
    Fixes a price/reserve mismatch (reserves older than the price); it cannot
    recover a change in pool depth (k) since the reserve snapshot.
    """
    k = tao_pool * alpha_pool
    return math.sqrt(k * price), math.sqrt(k / price)


# Price-outlier guard (applies on the rescale path; see load_parquet_ticks).
# A subnet-tick whose price is more than this factor away, in either direction,
# from the price implied by the reserve snapshot paired with it is dropped.
# Judgment call, measured 2026-10-09 on 2025-11-14 to 2026-06-26 (651,298
# non-root subnet-hours from load_parquet_ticks, bar_label="start"):
# |log(close / implied)| has p99 = log(1.18), p99.9 = log(1.83),
# p99.99 = log(175). Of the 432 subnet-hours beyond 3x, 362 sit within 2 days of
# a netuid re-registration or of the end of a subnet's startup mode; of the 611
# between 1.5x and 3x, 447 are near neither (so a tighter cut would mostly
# remove ordinary fast moves). See docs/known_traps.md.
DEFAULT_OUTLIER_MAX_RATIO = 3.0
# netuid 0 (root) is not a constant-product alpha pool: its price is fixed at
# 1.0 and its pool_history reserves imply about 4 (ratio about 0.25 on every
# hour from 2026-04-13), so it is never flagged.
OUTLIER_EXEMPT_NETUIDS = (0,)


def _is_price_outlier(price: float, implied: float, max_ratio: Optional[float]) -> bool:
    if max_ratio is None:
        return False
    if not (price > 0 and implied > 0):
        return False
    return abs(math.log(price / implied)) > math.log(max_ratio)


def _check_max_ratio(max_ratio) -> None:
    if max_ratio is not None and not (isinstance(max_ratio, (int, float)) and max_ratio > 1.0):
        raise ValueError(f"outlier_max_ratio must be a number > 1 or None, got {max_ratio!r}")


def _warn_price_outliers(dropped: list, n_total: int, max_ratio: float, where: str) -> None:
    if not dropped:
        return
    nets = sorted({d[1] for d in dropped})
    worst = sorted(dropped, key=lambda d: -abs(math.log(d[2] / d[3])))[:3]
    ex = "; ".join(
        f"netuid {d[1]} at {pd.Timestamp(d[0], unit='s', tz='UTC'):%Y-%m-%d %H:%M} UTC "
        f"close/implied = {d[2] / d[3]:.4g}" for d in worst)
    warnings.warn(
        f"{where}: dropped {len(dropped)} of {n_total} subnet-ticks ({len(nets)} subnets) whose price is more "
        f"than {max_ratio:g}x away from the price implied by the paired reserve snapshot (worst: {ex}). "
        "Mostly netuid re-registrations and startup-mode exits, where the snapshot describes a different "
        "or much deeper pool than the one that printed the price; rescaling would make that price "
        "fillable. Pass outlier_max_ratio=None to keep them; pass outlier_report=[] to get the list. "
        "See docs/known_traps.md.",
        UserWarning, stacklevel=3,
    )


def rescale_tick_reserves(
    ticks: list,
    outlier_max_ratio: Optional[float] = DEFAULT_OUTLIER_MAX_RATIO,
    outlier_report: Optional[list] = None,
    outlier_exempt_netuids: Iterable[int] = OUTLIER_EXEMPT_NETUIDS,
) -> list:
    """Rescale every SubnetTick's reserves to its own ``price``, in place.

    For CUSTOM tick builders that pair an intra-day price (for example an
    hourly OHLCV close) with a less frequent pool snapshot (for example daily
    ``pool_history`` reserves, forward-filled). ``BacktestEngine`` fills from
    ``tao_pool`` / ``alpha_pool``, not from ``price``, so stale reserves
    misprice every fill. This applies the same per-subnet transform that
    ``load_parquet_ticks(reserves="rescale")`` applies
    (``rescale_reserves_to_price``: keep k = tao x alpha, move the implied
    price to ``price``).

    Depth caveat: rescaling matches PRICE, not DEPTH. k still comes from the
    old snapshot, so any change in pool depth since then (a surge of new TAO,
    a large unstake) is not captured, and price impact of a fill is computed
    against the old depth. Where depth matters (surges, large orders) use
    SDK ticks (``load_sdk_ticks``) instead. Like ``reserves="rescale"`` in
    the loader, this is an opt-in experiment, not a validated fix: on active
    strategies rescaling moved results by orders of magnitude (see
    docs/known_traps.md, "Why rescale is not the default").

    Each rescaled subnet's ``signals`` gets ``reserve_gap_raw`` (implied
    price before rescaling / price - 1) unless the builder already set it.
    Subnets with non-positive price or reserves are left unchanged. Run
    ``reserve_price_gap`` before calling this to see how stale the builder's
    reserves were. Returns the same list for chaining.

    Outlier guard (on by default, because rescaling is what makes a bad price
    fillable): a subnet whose ``price`` is more than ``outlier_max_ratio``
    times away, in either direction, from the price its own reserves imply is
    REMOVED from that tick instead of rescaled, and a ``UserWarning`` reports
    the count. Rescaling such a tick would turn a price printed by a different
    pool (typically the new subnet's tiny bootstrap pool right after a netuid
    re-registration, paired with the old subnet's deep snapshot) into a deep
    pool the engine fills and marks at. The test uses only the tick's own
    price and reserves, so it is causal when the reserves are a snapshot at
    or before the tick. ``None`` turns it off; ``outlier_report`` (a list)
    receives one ``(timestamp, netuid, price, implied_price, "ratio")`` tuple
    per removed subnet; netuids in ``outlier_exempt_netuids`` (default: root,
    netuid 0) are never removed. The default 3.0 is a judgment call; see
    ``DEFAULT_OUTLIER_MAX_RATIO``.
    """
    _check_max_ratio(outlier_max_ratio)
    exempt = set(outlier_exempt_netuids)
    dropped: list = []
    n_total = 0
    for t in ticks:
        for nid in list(t.subnets):
            st = t.subnets[nid]
            n_total += 1
            if not (st.price > 0 and st.tao_pool > 0 and st.alpha_pool > 0):
                continue
            implied = st.tao_pool / st.alpha_pool
            if nid not in exempt and _is_price_outlier(st.price, implied, outlier_max_ratio):
                dropped.append((int(t.timestamp), int(nid), float(st.price), float(implied), "ratio"))
                del t.subnets[nid]
                continue
            gap_raw = implied / st.price - 1.0
            st.tao_pool, st.alpha_pool = rescale_reserves_to_price(
                st.tao_pool, st.alpha_pool, st.price)
            if st.signals is None:
                st.signals = {}
            st.signals.setdefault("reserve_gap_raw", gap_raw)
    if outlier_report is not None:
        outlier_report.extend(dropped)
    _warn_price_outliers(dropped, n_total, outlier_max_ratio, "rescale_tick_reserves")
    return ticks


def reserve_price_gap(ticks: Iterable) -> dict:
    """How far the pool-implied price (tao_pool / alpha_pool) is from each tick's own price.

    The engine fills from the reserves, so a large gap means mispriced fills.
    Accepts a list of ``TickData``. Returns ``{"n", "median_abs_gap",
    "p90_abs_gap", "share_over_2pct", "share_over_5pct", "max_abs_gap"}`` over every
    (tick, subnet) with positive price and reserves. Measured 2026-10-08:
    on the SDK feed the median is about 0.05% for 2026-02 to 2026-06 and
    about 0.00001% from 2026-07; hourly prices with raw daily reserves gave
    a median of 0.5% to 1% and 7% to 15% of subnet-ticks over 5% on two
    14-day windows. A median above about 0.5% or a share over 5% above a few
    percent suggests stale reserves (see docs/known_traps.md).
    """
    gaps = []
    for t in ticks:
        for st in t.subnets.values():
            if st.price > 0 and st.alpha_pool > 0 and st.tao_pool > 0:
                gaps.append(abs(st.tao_pool / st.alpha_pool / st.price - 1.0))
    if not gaps:
        return {"n": 0, "median_abs_gap": float("nan"), "p90_abs_gap": float("nan"),
                "share_over_2pct": float("nan"),
                "share_over_5pct": float("nan"), "max_abs_gap": float("nan")}
    s = pd.Series(gaps)
    return {"n": int(len(s)), "median_abs_gap": float(s.median()),
            "p90_abs_gap": float(s.quantile(0.90)), "share_over_2pct": float((s > 0.02).mean()),
            "share_over_5pct": float((s > 0.05).mean()), "max_abs_gap": float(s.max())}


# ── Parquet (hourly OHLCV + daily pool) → TickData ────────────────────


def load_parquet_ticks(
    start, end,
    ohlcv_parquet: Union[str, Path] = DEFAULT_OHLCV_HOURLY_PARQUET,
    pool_parquet: Union[str, Path] = DEFAULT_POOL_HISTORY_PARQUET,
    pool_tolerance_days: int = 2,
    drop_startup_mode: bool = True,
    stamp_identity: bool = True,
    reserves: str = "raw",
    bar_label: str = "start",
    reregistrations=None,
    outlier_max_ratio: Union[float, None, str] = "default",
    outlier_report: Optional[list] = None,
    outlier_exempt_netuids: Iterable[int] = OUTLIER_EXEMPT_NETUIDS,
    drop_pre_rebirth_reserves: bool = False,
) -> list:
    """Build TickData list from delegation_ohlcv_hourly + pool_history parquet.

    Use for OOS windows where SDK pool_state.csv doesn't reach (SDK starts
    2026-02-11). Hourly cadence vs SDK's 15-min; pool data is daily,
    forward-filled to each hourly tick via `pd.merge_asof`.

    READ docs/known_traps.md before using this for anything that trades or
    decides at the tick: three measured problems are documented in the module
    docstring (stale daily reserves; the bar-label lookahead; missed
    re-registrations). The defaults below keep the historical behavior so old
    results stay reproducible, and warn once per process.

    ``reserves`` (default ``"raw"``): the engine fills from reserves; ``"raw"``
    keeps the daily reserves unchanged (stale by up to a day). ``"rescale"``
    keeps only the invariant k = tao x alpha from the daily snapshot and rescales
    the reserves to the tick price (``rescale_reserves_to_price``): it removes
    the price/reserve mismatch but not depth changes, was not validated against
    contemporaneous reserves for active strategies, and on an autobot fade study
    moved results by orders of magnitude; treat it as an experiment, not a fix.

    ``bar_label`` (default ``"start"``): the OHLCV ``time`` is the bar START and
    its ``close`` is the pool price at the bar END. ``"start"`` stamps the tick
    with the bar start, so each tick carries a price one hour in the future (a
    lookahead for any strategy reading the price at the tick). ``"end"`` stamps it
    with the bar end (``time`` + 1 hour) so the close is known at its timestamp;
    the pool-history reserve snapshot is then matched at that later time too.

    Each tick's ``signals`` carries ``reserve_gap_raw`` (raw implied price / close
    - 1) and ``reserve_age_s`` (age of the daily snapshot).

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

    ``reregistrations`` ({netuid: [UTC timestamps]}) replaces the daily
    detector with exact events, e.g. ``utils.lifecycle.reregistrations_from_csvs``
    on SDK snapshots where they cover the window. The daily detector dates an
    event to its day and floors it to midnight, so a position opened earlier on
    the event day is stamped as the NEW subnet while sitting on the old pool,
    and the engine never closes it. Measured 2026-10-09 (11 strategy-window
    cases, HODL depth band, market, cheapest-10, against SDK ticks): mean
    absolute return error 6.3 points with the daily detector, 2.5 points with
    SDK events (e.g. SN82, window starting on its event day: +14.5% vs SDK
    -6.0%, fixed to -5.4%). An event-day blackout (drop the netuid's ticks that
    day) was tried and did not help (D1 error 29.6 points), so it is not offered.

    ``outlier_max_ratio`` (default ``"default"``: ``DEFAULT_OUTLIER_MAX_RATIO``
    = 3.0 with ``reserves="rescale"``, off with ``reserves="raw"`` so the
    default path stays the historical one): a subnet-hour whose close is more
    than this factor away, in either direction, from the price implied by the
    paired daily snapshot is DROPPED (the subnet is absent from that tick),
    and a ``UserWarning`` reports the count. On by default when rescaling
    because rescaling is what turns such a close into a pool the engine fills
    and marks at: measured 2026-10-09, netuid 97 at 2026-03-13 09:00 UTC
    closed at 40.09 TAO per alpha (the new subnet's bootstrap pool, after the
    old subnet deregistered at 09:19) against a snapshot-implied 0.00266, and
    one AutoBot CV fold under rescaled reserves returned +64,298%. Under raw
    reserves the engine fills and marks at the snapshot, so such a close
    misleads a strategy's decision but not its fills or marks. Dropping is used
    rather than raw reserves (the old pool no longer exists) or clipping (it
    invents a price). Causal: uses only the close and a snapshot at or before
    the tick. With exact ``reregistrations`` and ``bar_label="start"``, the
    hour whose bar contains the event is stamped as the OLD subnet but closes
    at the new pool's price; this guard is what removes it on the rescale
    path. ``None`` keeps every hour; a number > 1 sets the factor (also on
    raw); ``outlier_report`` (a list) receives ``(timestamp, netuid, close,
    implied_price, "ratio")`` per dropped hour; netuids in
    ``outlier_exempt_netuids`` (default root, netuid 0) are never dropped.

    ``drop_pre_rebirth_reserves`` (default ``False``, any ``reserves`` mode):
    drop a subnet-hour at or after a re-registration of its netuid whose
    paired snapshot predates that event, i.e. reserves that describe the dead
    subnet. Events are the ones used for stamping: ``reregistrations`` when
    given (exact times), else the daily detector's, floored to midnight UTC.
    With the daily detector this drops the event day's hours up to the
    end-of-day snapshot, which are the old subnet's hours that the floored
    stamping labels as the NEW subnet (a position bought then is never closed
    and is later marked at the new subnet). That is up to one day of
    lookahead and close to the event-day blackout described above, which did
    not help on a HODL window, so it is opt-in. Measured 2026-10-10 on one
    AutoBot V2-moderate CV fold (2026-02-11 to 2026-03-14, rescale, ratio
    guard on, one seed): daily detector +19.62% with a 41% drawdown from the
    netuid 76 rebirth day; with this option +5.01% (3.1%); with exact SDK
    events instead +4.88% (3.2%). Prefer exact events where SDK snapshots
    reach. Dropped hours go to ``outlier_report`` with reason
    ``"pre_rebirth_reserves"``.
    """
    from bt_trading_tools.backtest.types import SubnetTick, TickData

    if reserves not in ("rescale", "raw"):
        raise ValueError(f"reserves must be 'rescale' or 'raw', got {reserves!r}")
    if bar_label not in ("start", "end"):
        raise ValueError(f"bar_label must be 'start' or 'end', got {bar_label!r}")
    if isinstance(outlier_max_ratio, str):
        if outlier_max_ratio != "default":
            raise ValueError(f"outlier_max_ratio must be a number > 1, None or 'default', got {outlier_max_ratio!r}")
        outlier_max_ratio = DEFAULT_OUTLIER_MAX_RATIO if reserves == "rescale" else None
    _check_max_ratio(outlier_max_ratio)
    global _PARQUET_WARNED
    if not _PARQUET_WARNED:
        _PARQUET_WARNED = True
        issues = []
        if reserves == "raw":
            issues.append("daily reserves carried forward under hourly prices (engine fills from the reserves)")
        if bar_label == "start":
            issues.append("each tick carries the price from one hour LATER than its timestamp (bar_label='start')")
        issues.append("re-registrations are dated to the day (floored to midnight): positions opened earlier that day on a re-registered subnet are not closed; pass exact events via reregistrations=")
        warnings.warn(
            "load_parquet_ticks known issues (measured 2026-10): " + "; ".join(issues) + ". "
            "Results from this source can be mispriced or look-ahead biased; prefer SDK ticks where they reach. "
            "See docs/known_traps.md. (Shown once per process.)",
            RuntimeWarning, stacklevel=2,
        )

    start_ts = coerce_to_utc_timestamp(start)
    end_ts = coerce_to_utc_timestamp(end)

    ohlcv = pd.read_parquet(ohlcv_parquet)
    ohlcv["unix_ts"] = to_unix_seconds(ohlcv["time"])
    if bar_label == "end":
        ohlcv["unix_ts"] = ohlcv["unix_ts"] + 3600
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
    n_total = len(merged)
    stamp_events = None
    if stamp_identity or drop_pre_rebirth_reserves:
        from bt_trading_tools.utils.lifecycle import reregistrations_from_pool_history
        if reregistrations is not None:
            stamp_events, floored = reregistrations, False
        elif not stamp_identity and "alpha_staked" not in pool_unfiltered.columns:
            stamp_events, floored = {}, True   # cannot detect events (synthetic pool files)
        else:
            stamp_events, floored = reregistrations_from_pool_history(pool_unfiltered), True
    if drop_pre_rebirth_reserves and len(merged):
        stale = pd.Series(False, index=merged.index)
        for nid, events in stamp_events.items():
            sel = merged["netuid"] == nid
            if not sel.any():
                continue
            for ev in events:
                ev = pd.Timestamp(ev)
                ev = ev.tz_localize("UTC") if ev.tzinfo is None else ev
                cut = (ev.normalize() if floored else ev).timestamp()
                stale |= sel & (merged["pool_ts"] < ev.timestamp()) & (merged["unix_ts"] >= cut)
        dropped = [(int(r.unix_ts), int(r.netuid), float(r.close), float(r.total_tao / r.alpha_in_pool),
                    "pre_rebirth_reserves") for r in merged[stale].itertuples(index=False)]
        merged = merged[~stale]
        if outlier_report is not None:
            outlier_report.extend(dropped)
        if dropped:
            warnings.warn(
                f"load_parquet_ticks: dropped {len(dropped)} of {n_total} subnet-hours "
                f"({len({d[1] for d in dropped})} subnets) whose reserve snapshot predates a re-registration "
                "of that netuid (the reserves describe the dead subnet). See docs/known_traps.md.",
                UserWarning, stacklevel=2,
            )
    if outlier_max_ratio is not None and len(merged):
        implied = merged["total_tao"] / merged["alpha_in_pool"]
        bad = ((merged["close"] / implied).map(math.log).abs() > math.log(outlier_max_ratio)) \
            & ~merged["netuid"].isin(list(outlier_exempt_netuids))
        dropped = [(int(r.unix_ts), int(r.netuid), float(r.close), float(r.total_tao / r.alpha_in_pool), "ratio")
                   for r in merged[bad].itertuples(index=False)]
        merged = merged[~bad]
        if outlier_report is not None:
            outlier_report.extend(dropped)
        _warn_price_outliers(dropped, n_total, outlier_max_ratio, "load_parquet_ticks")

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
        from bt_trading_tools.utils.lifecycle import stamp_generations
        # Exact events stamp at their time; daily-detector events are floored
        # to midnight (see docstring for the failure this leaves; pass SDK
        # events via `reregistrations`).
        stamp_generations(ticks, stamp_events, floor_to_day=floored)
    return ticks


__all__ = [
    "DEFAULT_OHLCV_HOURLY_PARQUET",
    "DEFAULT_POOL_HISTORY_PARQUET",
    "DEFAULT_SDK_POOL_STATE_CSV",
    "to_unix_seconds",
    "coerce_to_utc_timestamp",
    "DEFAULT_OUTLIER_MAX_RATIO",
    "OUTLIER_EXEMPT_NETUIDS",
    "rescale_reserves_to_price",
    "rescale_tick_reserves",
    "reserve_price_gap",
    "load_sdk_ticks",
    "load_parquet_ticks",
]
