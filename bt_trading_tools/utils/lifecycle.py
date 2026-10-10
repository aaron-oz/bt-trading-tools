"""
Subnet lifecycle detection — handles deregistration and rebirth.

Bittensor subnets can be deregistered, and their netuid reused by a
completely different subnet. SN47 in January may be an entirely different
entity than SN47 in March. We detect these boundaries and mask data
across them so features/models never see cross-lifecycle data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def detect_lifecycle_boundaries(
    pool_history: pd.DataFrame,
    subnet_ids: list[int],
    timestamps: np.ndarray,
    margin_hours: int = 168,  # 7 days buffer after rebirth
) -> np.ndarray:
    """
    Detect subnet lifecycle boundaries from pool_history startup_mode transitions.

    Returns:
        valid_mask: (n_times, n_subnets) bool array.
            True = this (time, subnet) pair is safe to use.
            False = subnet is in startup, recently reborn, or inactive.
    """
    nt = len(timestamps)
    ns = len(subnet_ids)
    valid = np.ones((nt, ns), dtype=bool)

    if "startup_mode" not in pool_history.columns:
        return valid

    ts_series = pd.Series(timestamps)

    for si, nid in enumerate(subnet_ids):
        sub = pool_history[pool_history["netuid"] == nid].sort_values("timestamp")
        if len(sub) == 0:
            valid[:, si] = False
            continue

        sm = sub["startup_mode"].fillna(False).astype(bool)
        sub_ts = sub["timestamp"].values

        # Find all startup_mode=True periods
        startup_ranges = []
        in_startup = False
        start = None
        for idx in range(len(sub)):
            if sm.iloc[idx] and not in_startup:
                start = sub_ts[idx]
                in_startup = True
            elif not sm.iloc[idx] and in_startup:
                end = sub_ts[idx]
                startup_ranges.append((start, end))
                in_startup = False
        if in_startup:
            startup_ranges.append((start, sub_ts[-1]))

        # Mask startup periods + margin_hours after each startup ends
        for start, end in startup_ranges:
            end_with_margin = end + np.timedelta64(margin_hours, "h")
            mask = (timestamps >= np.datetime64(start)) & (timestamps <= np.datetime64(end_with_margin))
            valid[mask, si] = False

        # Also mask before the subnet's first appearance
        first_valid = sub_ts[0]
        valid[timestamps < np.datetime64(first_valid), si] = False

    return valid


def apply_lifecycle_mask(
    prices: np.ndarray,
    valid_mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Set invalid (cross-lifecycle) prices to fill_value."""
    out = prices.copy()
    out[~valid_mask] = fill_value
    return out


def get_lifecycle_segments(
    pool_history: pd.DataFrame,
    netuid: int,
) -> list[dict]:
    """
    Get all lifecycle segments for a subnet.

    Returns list of dicts with:
        - start: first valid timestamp
        - end: last valid timestamp (or ongoing)
        - is_current: whether this is the most recent lifecycle
    """
    sub = pool_history[pool_history["netuid"] == netuid].sort_values("timestamp")
    if len(sub) == 0:
        return []

    sm = sub["startup_mode"].fillna(False).astype(bool)
    segments = []
    in_startup = True
    seg_start = None

    for idx in range(len(sub)):
        if not sm.iloc[idx] and in_startup:
            # Transition from startup to active
            seg_start = sub.iloc[idx]["timestamp"]
            in_startup = False
        elif sm.iloc[idx] and not in_startup:
            # Transition from active to startup (rebirth happening)
            if seg_start is not None:
                segments.append({
                    "start": seg_start,
                    "end": sub.iloc[idx - 1]["timestamp"],
                    "is_current": False,
                })
            in_startup = True
            seg_start = None

    # Current segment
    if not in_startup and seg_start is not None:
        segments.append({
            "start": seg_start,
            "end": sub.iloc[-1]["timestamp"],
            "is_current": True,
        })

    return segments


# ── Re-registration detection and generation stamping (2026-10-01) ──────
#
# A netuid is a slot, not an identity. ``find_reregistrations`` locates the
# moments a new subnet took over a slot; ``stamp_generations`` writes a
# per-slot generation counter onto ``SubnetTick.generation`` so that
# ``BacktestEngine`` can close a position whose subnet was replaced instead of
# selling it into the new subnet's pool.
#
# Detection rule (AMM mechanics, not a tuned threshold): between consecutive
# snapshots of one netuid, flag a re-registration when staked alpha
# (``alpha_out``) collapses by more than ``collapse_frac`` while pool alpha
# (``alpha_in``) also falls. Unstaked alpha is sold INTO the pool, so a genuine
# mass exit raises ``alpha_in``; both sides falling at once means both were
# replaced. Where a ``startup_mode`` column exists, a False -> True transition
# is also a re-registration (a subnet cannot re-enter startup mode).
#
# Detect on UNFILTERED data. Dropping startup rows (as many tick builders do)
# hides the event and turns it into an apparent multi-x price jump.


def find_reregistrations(
    snapshots: pd.DataFrame,
    collapse_frac: float = 0.80,
    merge_within_s: int = 7200,
) -> dict[int, list[pd.Timestamp]]:
    """Return {netuid: [UTC timestamps at which a new subnet took the slot]}.

    Args:
        snapshots: one row per (netuid, snapshot) with columns ``timestamp``,
            ``netuid``, ``alpha_in`` (pool alpha), ``alpha_out`` (staked
            alpha), optionally ``startup_mode``. Units do not matter (ratios).
            For taostats ``pool_history`` use
            :func:`reregistrations_from_pool_history`, which renames columns
            and collapses multiple rows per day first.
        collapse_frac: staked alpha must fall by more than this fraction.
        merge_within_s: events at one netuid closer than this are one
            registration observed across adjacent snapshots.
    """
    cols = ["timestamp", "netuid", "alpha_in", "alpha_out"]
    has_startup = "startup_mode" in snapshots.columns
    if has_startup:
        cols.append("startup_mode")
    df = snapshots[cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out: dict[int, list[pd.Timestamp]] = {}
    for netuid, g in df.groupby("netuid"):
        g = g.sort_values("timestamp")
        a_in, a_out = g["alpha_in"], g["alpha_out"]
        hit = (
            (a_out < (1 - collapse_frac) * a_out.shift(1))
            & (a_in < a_in.shift(1))
            & (a_out.shift(1) > 0)
        )
        if has_startup:
            sm = g["startup_mode"].astype(str).str.lower().eq("true")
            # shift() on bool gives object dtype; `~` on object is integer
            # negation (~True == -2, truthy). Cast explicitly.
            prev = sm.shift(1, fill_value=True).astype(bool)
            hit = hit | (sm & ~prev)
        merged: list[pd.Timestamp] = []
        for t in g.loc[hit, "timestamp"]:
            if merged and (t - merged[-1]).total_seconds() <= merge_within_s:
                continue
            merged.append(t)
        if merged:
            out[int(netuid)] = merged
    return out


def reregistrations_from_pool_history(
    pool_history: pd.DataFrame, collapse_frac: float = 0.80,
) -> dict[int, list[pd.Timestamp]]:
    """:func:`find_reregistrations` for the taostats daily ``pool_history``.

    Pass the UNFILTERED frame (startup rows kept). pool_history carries
    several rows per netuid per day and ``startup_mode`` is not constant
    within a day, so rows are collapsed to daily-last first.
    """
    ph = pool_history.rename(
        columns={"alpha_in_pool": "alpha_in", "alpha_staked": "alpha_out"})
    ph = ph.copy()
    ph["timestamp"] = pd.to_datetime(ph["timestamp"], utc=True, format="ISO8601")
    ph["_date"] = ph["timestamp"].dt.normalize()
    agg = {"timestamp": ("timestamp", "last"),
           "alpha_in": ("alpha_in", "last"),
           "alpha_out": ("alpha_out", "last")}
    if "startup_mode" in ph.columns:
        agg["startup_mode"] = ("startup_mode", "last")
    daily = (ph.sort_values("timestamp")
               .groupby(["netuid", "_date"], as_index=False).agg(**agg))
    return find_reregistrations(daily, collapse_frac=collapse_frac, merge_within_s=0)


def reregistrations_from_csvs(
    paths, collapse_frac: float = 0.80,
) -> dict[int, list[pd.Timestamp]]:
    """:func:`find_reregistrations` on SDK pool-state CSV snapshots.

    ``paths`` is one path or a list; files are concatenated and de-duplicated
    on (timestamp, netuid). Needs columns ``timestamp, netuid, alpha_in,
    alpha_out``. Event times have the snapshot resolution (15 minutes or
    finer), much tighter than the daily pool_history rule.
    """
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        paths = [paths]
    cols = ["timestamp", "netuid", "alpha_in", "alpha_out"]
    df = pd.concat([pd.read_csv(p, usecols=cols) for p in paths])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(["timestamp", "netuid"]).sort_values("timestamp")
    return find_reregistrations(df, collapse_frac=collapse_frac)


def stamp_generations(
    ticks: list,
    reregistrations: dict[int, list[pd.Timestamp]],
    floor_to_day: bool = False,
) -> list:
    """Set ``SubnetTick.generation`` on every subnet of every tick, in place.

    Generation = number of re-registrations of that netuid at or before the
    tick (so the first subnet seen in the data is 0). Netuids with no events
    get 0, which is still a real stamp: the engine treats a stamped tick as
    "identity known".

    Args:
        floor_to_day: push each event back to midnight UTC. Use with DAILY
            ticks normalized to midnight, where an event at 08:41 would
            otherwise leave that day's tick in the old generation.

    Returns the same list for chaining.
    """
    cut = {n: sorted(int((t.normalize() if floor_to_day else t).timestamp())
                     for t in ts)
           for n, ts in reregistrations.items()}
    for tick in ticks:
        for netuid, st in tick.subnets.items():
            cs = cut.get(int(netuid))
            st.generation = (sum(1 for c in cs if tick.timestamp >= c)
                             if cs else 0)
    return ticks
