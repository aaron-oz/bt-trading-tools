"""Point-in-time alpha yield rates from the taostats validator yield history.

Why this exists. ``BacktestEngine`` credits alpha yield through a yield model
whose rate source decides the result. The default cascade in a backtest has no
history behind it: with the validator-selection cache pinned off (to avoid
using today's cache for past dates) and no API key, only the empirical tier is
live, and it is not point-in-time. The paper bots use the taostats live tier
and the validator-selection cache. Measured on the emission-drought bot's
2026-06-30 to 2026-09-23 window, paper's logged rate at a sale was 2.4 to 9.5
times the backtest's (median 4.2). This module rebuilds the cache's rate for any
past date from the recorded history, so a backtest can use the SAME rates as
paper, as of the date being simulated.

Rate definition (identical to ``scripts/update_validator_selection.py`` and
``ValidatorCacheYieldProvider``): per subnet and date, take the validators with
``thirty_day_epoch_participation >= min_uptime`` and a stake share of the
subnet's recorded stake of at least ``min_stake_share``, keep the top ``top_k``
by ``thirty_day_apy``, and return the stake-weighted mean of
``thirty_day_apy`` divided by 365 (a simple-interest daily rate).

Two accrual conventions, selectable on ``HistoricalYieldModel``:

``"sale_rate"`` (default, what ``AlphaYieldModel`` does): the rate on the date
    of ``now`` is applied to the whole hold, ``alpha * rate(now) * days``.
    Matches how paper credits yield, including its bias when the rate at the
    sale differs from the rate during the hold.
``"integrated"``: the sum over the hold of ``alpha * rate(day) * day_fraction``
    using each day's own rate; closer to a time-varying yield, but NOT what
    paper does.

Rates above ``MAX_PLAUSIBLE_RATE_PER_DAY`` are treated as data errors (0.0),
as in ``AlphaYieldModel.rate``. A date or subnet with no history uses the most
recent earlier rate for that subnet (no look-ahead); with none, 0.0.
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from bt_trading_tools.alpha_yield import (
    MAX_PLAUSIBLE_RATE_PER_DAY,
    _warn_rejected,
    generation_end,
    implausible_rate,
    normalize_reregistrations,
)

DAY_S = 86400.0
HISTORY_COLUMNS = ["date", "netuid", "hotkey", "stake_rao", "thirty_day_apy",
                   "thirty_day_epoch_participation", "timestamp"]


def load_validator_yield_history(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> pd.DataFrame:
    """Read one or more history CSVs (same schema) into one frame."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [pd.read_csv(p, usecols=HISTORY_COLUMNS) for p in paths]
    return pd.concat(frames, ignore_index=True)


def build_rate_table(
    history: pd.DataFrame,
    min_uptime: float = 0.95,
    min_stake_share: float = 0.02,
    top_k: int = 20,
) -> pd.DataFrame:
    """Per (netuid, date) rate per day, by the validator-selection cache's rules.

    The LAST snapshot of each (netuid, date) is used (history files can hold
    several snapshots per day). Returns columns ``netuid``, ``date`` (midnight
    UTC, ``datetime64``), ``rate_per_day``.
    """
    h = history.dropna(subset=["thirty_day_apy", "stake_rao"]).copy()
    h = h[h["stake_rao"] > 0]
    last = h.groupby(["netuid", "date"])["timestamp"].transform("max")
    h = h[h["timestamp"] == last]
    total = h.groupby(["netuid", "date"])["stake_rao"].transform("sum")
    h["share"] = h["stake_rao"] / total
    h = h[(h["thirty_day_epoch_participation"] >= min_uptime) & (h["share"] >= min_stake_share)]
    h = h.sort_values(["netuid", "date", "thirty_day_apy"], ascending=[True, True, False])
    h = h.groupby(["netuid", "date"]).head(top_k)
    h["w"] = h["stake_rao"] * h["thirty_day_apy"]
    g = h.groupby(["netuid", "date"]).agg(w=("w", "sum"), s=("stake_rao", "sum")).reset_index()
    g["rate_per_day"] = (g["w"] / g["s"]) / 365.0
    g["date"] = pd.to_datetime(g["date"], utc=True)
    return g[["netuid", "date", "rate_per_day"]]


class HistoricalYieldModel:
    """Duck-typed yield model for ``BacktestEngine`` (``accrued_yield``).

    Args:
        rate_table: output of :func:`build_rate_table`.
        convention: ``"sale_rate"`` or ``"integrated"`` (module docstring).
        lag_days: use the rate from ``lag_days`` before the simulated date
            (the validator-selection cache is refreshed daily and may be up to
            a day old). Default 0.
        max_rate_per_day: rates above this are treated as data errors (0.0).
            The number of such rows is in ``self.rejected_rate_rows`` and is
            reported once as a ``UserWarning`` at construction.
        reregistrations: optional ``{netuid: [timestamps]}`` (output of
            ``bt_trading_tools.utils.lifecycle.reregistrations_from_pool_history``).
            When given, accrual stops at the first re-registration of the
            netuid after ``entry_time``, and the rate used is the one in force
            the day before that re-registration, so a position never earns the
            rate of the unrelated subnet that later took its slot.
    """

    def __init__(self, rate_table: pd.DataFrame, convention: str = "sale_rate",
                 lag_days: int = 0, max_rate_per_day: float = MAX_PLAUSIBLE_RATE_PER_DAY,
                 reregistrations=None):
        if convention not in ("sale_rate", "integrated"):
            raise ValueError(f"convention must be 'sale_rate' or 'integrated', got {convention!r}")
        self.convention = convention
        self.lag_days = int(lag_days)
        self.max_rate = max_rate_per_day
        self._days: dict[int, list[int]] = {}
        self._rates: dict[int, list[float]] = {}
        self.rejected_rate_rows = 0
        self.rejected_examples: list = []
        for netuid, g in rate_table.sort_values(["netuid", "date"]).groupby("netuid"):
            self._days[int(netuid)] = [int(d.timestamp() // DAY_S) for d in g["date"]]
            self._rates[int(netuid)] = [float(r) for r in g["rate_per_day"]]
            for d, r in zip(g["date"], self._rates[int(netuid)]):
                if implausible_rate(r, self.max_rate):
                    self.rejected_rate_rows += 1
                    if len(self.rejected_examples) < 20:
                        self.rejected_examples.append((int(netuid), str(d.date()), r))
        self._reregs = normalize_reregistrations(reregistrations)
        _warn_rejected("HistoricalYieldModel", self.rejected_rate_rows,
                       self.rejected_examples)

    def rate_on(self, netuid: int, unix_ts: float) -> float:
        """The rate in force on the date of ``unix_ts`` (minus ``lag_days``);
        the latest earlier rate when that date has none; 0.0 when none or implausible."""
        days = self._days.get(int(netuid))
        if not days:
            return 0.0
        day = int(unix_ts // DAY_S) - self.lag_days
        i = bisect.bisect_right(days, day) - 1
        if i < 0:
            return 0.0
        r = self._rates[int(netuid)][i]
        if implausible_rate(r, self.max_rate):
            return 0.0
        return r

    def accrued_yield(self, netuid: int, alpha_qty: float, entry_time: float, now: float) -> float:
        """Alpha accrued on ``alpha_qty`` held from ``entry_time`` to ``now`` (unix seconds)."""
        if alpha_qty <= 0 or not math.isfinite(alpha_qty) or now <= entry_time:
            return 0.0
        end = generation_end(self._reregs, netuid, float(entry_time), float(now))
        # Last instant whose daily rate certainly belongs to the entry's
        # generation: the end itself, or the day before the rebirth day.
        last_safe = end if end >= now else (end // DAY_S) * DAY_S - 1.0
        if end <= entry_time:
            return 0.0
        if self.convention == "sale_rate":
            return alpha_qty * self.rate_on(netuid, last_safe) * (end - entry_time) / DAY_S
        total, t = 0.0, float(entry_time)
        while t < end:                                   # walk day boundaries
            nxt = min(end, (t // DAY_S + 1) * DAY_S)
            total += self.rate_on(netuid, min(t, last_safe)) * (nxt - t) / DAY_S
            t = nxt
        return alpha_qty * total
