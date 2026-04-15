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
