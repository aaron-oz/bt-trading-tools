"""Realized on-chain staking yield for paper and backtest: per validator, causal, trued up.

Why this exists. Until 2026-10-12 both paper and the backtest credited staking
yield at a MODELED rate (a taostats 30-day trailing APY, pooled over a set of
validators). This module credits the yield the chain actually paid, through
the specific validator a position stakes with, and gives decisions only the
part of that record that was observable at decision time.

Input: the daily on-chain table (``alpha-trading/data/backfill_onchain_yield.py
build``; on bot-vps the daily job ``data/onchain_yield_daily_job.py``). One row
per (UTC date d, netuid, validator hotkey):

    rate_per_day(d) = (P(end of d) / P(start of d) - 1) * 86400 / seconds(d),
    P = TotalHotkeyAlpha / TotalHotkeyShares

which is what an unmoved staker on that validator earned during d, after the
validator's take. Rows with ``valid`` False (missing endpoint, re-registration
inside d, negative or implausible growth) carry no rate.

Observability (the no-lookahead rule). Day d's rate is known once the boundary
snapshot at the end of d has been read. The model treats it as observable from
``end_of(d) + observe_lag_s`` (default 2 hours: the bot-vps job runs a few
minutes after midnight UTC and retries hourly) AND only if the table contains
date d at all (in paper the table holds only what the job has written). So at
time T the last observable day is

    obs_day(T) = min(floor((T - observe_lag_s) / 86400) - 1, last date in table)

Everything a decision reads (validator ranking, trailing yields) uses days
``<= obs_day(T)`` only.

Crediting (provisional credit, then true-up). Yield over an interval is
integrated day by day. For a day ``d <= obs_day(as_of)`` the realized rate of
d is used. For a later day (today, or yesterday before the job ran) the rate
is PROVISIONAL: the last realized rate known at ``as_of``. The ledger keeps a
committed total only up to ``final_through(as_of) = end_of(obs_day(as_of))``
and recomputes the provisional tail at every tick
(``ledger.accrue_yield(..., commit_to=...)``), so once the day's realized
rate becomes observable the tail is re-credited at that rate: the true-up.
Committed days are never revalued. The one place a provisional credit is not
trued up is a fold (a top-up, partial sell or full sell): the alpha folded
or sold includes the provisional tail at that moment, as a real staker's
balance would include whatever the chain had paid, which paper cannot know
yet. Paper and the backtest share this code, so both behave the same way.

Rate per day d for a position (``mode="validator"``, the default):

1. the position's validator hotkey has a valid row for d: that rate;
2. otherwise the pooled subnet rate for d (stake-weighted mean over the
   pooled set: permit holders with stake share >= ``pooled_min_stake_share``,
   top ``pooled_top_k`` by stake; :func:`bt_trading_tools.onchain_yield.build_onchain_rate_table`);
3. otherwise 0.0.

``mode="pooled"`` skips step 1: every position earns the pooled rate. It
exists to compare the validator rule against the pooled rate with identical
decisions.

Validator selection rule (:class:`ValidatorRule`), evaluated at decision time
T on ``obs_day(T)``: among the subnet's validators with a permit on that day,
stake share >= ``min_stake_share``, delegate take <= ``max_take`` (unknown take
passes unless ``unknown_take_passes`` is False), and at least
``min_valid_days`` valid daily rates in the trailing ``window_days`` days,
pick the highest mean realized rate over that window; ties go to the
lexicographically smaller hotkey, so the choice is deterministic. If no
validator passes, the position gets no hotkey (pooled rate) and the choice is
retried at later ticks.

A re-registration of the netuid (``reregistrations``) stops accrual at the
first rebirth after the position's yield origin, and no day on or after the
rebirth day is paid, as in ``HistoricalYieldModel``.
"""
from __future__ import annotations

import bisect
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from bt_trading_tools.alpha_yield import (
    MAX_PLAUSIBLE_RATE_PER_DAY,
    YieldRate,
    generation_end,
    normalize_reregistrations,
)
from bt_trading_tools.tracking.schema import AlphaYieldSource

DAY_S = 86400
DEFAULT_OBSERVE_LAG_S = 7200.0

_COLUMNS = ["date", "netuid", "hotkey", "rate_per_day", "valid", "permit",
            "stake_share", "total_hotkey_alpha"]


@dataclass(frozen=True)
class ValidatorRule:
    """Explicit validator selection rule (module docstring).

    Defaults: 7-day trailing window, at least 5 valid days in it, take at most
    0.18 (the chain's default and maximum delegate take, so the cap is
    inactive by default), stake share at least 2% of the subnet's permit
    holders' stake, permit required.
    """
    window_days: int = 7
    min_valid_days: int = 5
    max_take: float = 0.18
    min_stake_share: float = 0.02
    require_permit: bool = True
    unknown_take_passes: bool = True


@dataclass(frozen=True)
class ValidatorPick:
    """Result of :meth:`RealizedYieldModel.choose_validator`."""
    netuid: int
    hotkey: str
    trailing_rate: float          # mean realized rate per day over the rule window
    trailing_rate_30d: Optional[float]
    n_valid_days: int
    n_eligible: int               # validators passing every filter
    take: Optional[float]
    stake_share: float
    obs_day: int                  # unix day number the choice was made on


def _day(ts: float) -> int:
    return int(math.floor(float(ts) / DAY_S))


def _trailing(days: np.ndarray, rates: np.ndarray, window: int):
    """Per row i: mean and count of valid rates over days in [d_i - window + 1, d_i]."""
    ok = np.isfinite(rates)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, rates, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    lo = np.searchsorted(days, days - window + 1, side="left")
    hi = np.arange(1, len(days) + 1)
    n = cn[hi] - cn[lo]
    s = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return mean, n


class RealizedYieldModel:
    """Duck-typed yield model for ``BacktestEngine`` and ``PaperBotBase``.

    Args:
        daily: the daily on-chain table (columns in the module docstring;
            ``take`` optional), or a path to it (parquet or csv).
        rule: the validator selection rule.
        mode: ``"validator"`` (positions earn their validator's rate) or
            ``"pooled"`` (every position earns the pooled subnet rate).
        observe_lag_s: seconds after the end of a UTC day before its rate is
            treated as known.
        reregistrations: optional ``{netuid: [timestamps]}``; accrual stops at
            the first re-registration after the position's yield origin.
        keep_days: keep only the last ``keep_days`` dates of the table (paper
            uses this to bound memory); None keeps all.
        reload_check_s: when built from a path, re-read the file if its
            modification time changed, checked at most this often (paper:
            the daily job appends while the bot runs). None disables.
    """

    supports_hotkey = True

    def __init__(self, daily: Union[pd.DataFrame, str, Path], rule: ValidatorRule = ValidatorRule(),
                 mode: str = "validator", observe_lag_s: float = DEFAULT_OBSERVE_LAG_S,
                 reregistrations=None, pooled_min_stake_share: float = 0.02,
                 pooled_top_k: int = 20, keep_days: Optional[int] = None,
                 reload_check_s: Optional[float] = 60.0,
                 max_rate_per_day: float = MAX_PLAUSIBLE_RATE_PER_DAY):
        if mode not in ("validator", "pooled"):
            raise ValueError(f"mode must be 'validator' or 'pooled', got {mode!r}")
        self.rule = rule
        self.mode = mode
        self.observe_lag_s = float(observe_lag_s)
        self.pooled_min_stake_share = pooled_min_stake_share
        self.pooled_top_k = pooled_top_k
        self.keep_days = keep_days
        self.max_rate = max_rate_per_day
        self._reregs = normalize_reregistrations(reregistrations)
        self._path: Optional[Path] = None
        self._mtime = 0.0
        self._next_check = 0.0
        self._reload_check_s = reload_check_s
        if isinstance(daily, (str, Path)):
            self._path = Path(daily)
            self._mtime = self._path.stat().st_mtime
            daily = self._read(self._path)
        self._ingest(daily)

    # ------------------------------------------------------------------ data
    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            names = set(pq.ParquetFile(path).schema.names)
            cols = [c for c in _COLUMNS + ["take"] if c in names]
            return pd.read_parquet(path, columns=cols)
        return pd.read_csv(path)

    def _ingest(self, daily: pd.DataFrame) -> None:
        from bt_trading_tools.onchain_yield import build_onchain_rate_table
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"], utc=True).dt.normalize()
        if self.keep_days is not None and len(d):
            d = d[d["date"] > d["date"].max() - pd.Timedelta(days=int(self.keep_days))]
        d["netuid"] = d["netuid"].astype(int)
        d["day"] = ((d["date"] - pd.Timestamp("1970-01-01", tz="UTC"))
                    // pd.Timedelta(days=1)).astype(int)
        if "take" not in d.columns:
            d["take"] = np.nan
        valid = d["valid"].fillna(False).astype(bool)
        r = d["rate_per_day"].astype(float)
        bad = ~np.isfinite(r) | (r < 0) | (r > self.max_rate)
        d["r"] = np.where(valid & ~bad, r, np.nan)
        self.first_day = int(d["day"].min()) if len(d) else None
        self.last_day = int(d["day"].max()) if len(d) else None

        # per (netuid, hotkey): realized rates and trailing stats per sampled day
        d = d.sort_values(["netuid", "hotkey", "day"]).reset_index(drop=True)
        W, W30 = int(self.rule.window_days), 30
        mean_w = np.full(len(d), np.nan)
        n_w = np.zeros(len(d), dtype=np.int64)
        mean_30 = np.full(len(d), np.nan)
        self._hk: dict[tuple[int, str], tuple[list[int], list[float]]] = {}
        keys = d[["netuid", "hotkey"]].to_numpy()
        days_all = d["day"].to_numpy()
        r_all = d["r"].to_numpy()
        starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
        ends = np.r_[starts[1:], len(d)]
        for a, b in zip(starts, ends):
            days, rates = days_all[a:b], r_all[a:b]
            m, n = _trailing(days, rates, W)
            mean_w[a:b], n_w[a:b] = m, n
            mean_30[a:b] = _trailing(days, rates, W30)[0]
            ok = np.isfinite(rates)
            self._hk[(int(keys[a][0]), str(keys[a][1]))] = (
                days[ok].tolist(), rates[ok].tolist())
        d["trail"], d["n_valid"], d["trail30"] = mean_w, n_w, mean_30

        # candidates per (netuid, day): every row with its filters precomputed
        self._cand: dict[tuple[int, int], pd.DataFrame] = {}
        cand = d[["netuid", "day", "hotkey", "permit", "stake_share", "take", "trail",
                  "n_valid", "trail30"]]
        for (n, day), g in cand.groupby(["netuid", "day"], sort=False):
            self._cand[(int(n), int(day))] = g
        self._pick_cache: dict[tuple[int, int], Optional[ValidatorPick]] = {}

        # pooled subnet rate per day
        pooled = build_onchain_rate_table(
            d.drop(columns=["r", "trail", "n_valid", "trail30", "day"]),
            min_stake_share=self.pooled_min_stake_share, top_k=self.pooled_top_k)
        self._pool: dict[int, tuple[list[int], list[float]]] = {}
        for n, g in pooled.groupby("netuid"):
            days = [int(x.timestamp()) // DAY_S for x in g["date"]]
            rates = [float(x) for x in g["rate_per_day"]]
            keep = [(dd, rr) for dd, rr in zip(days, rates)
                    if math.isfinite(rr) and 0 <= rr <= self.max_rate]
            self._pool[int(n)] = ([k[0] for k in keep], [k[1] for k in keep])

    def maybe_reload(self, now: Optional[float] = None) -> bool:
        """Re-read the table file if it changed (paper). Returns True on reload.
        A failed read keeps the previous table."""
        if self._path is None or self._reload_check_s is None:
            return False
        t = time.time() if now is None else now
        if t < self._next_check:
            return False
        self._next_check = t + self._reload_check_s
        try:
            m = self._path.stat().st_mtime
            if m == self._mtime:
                return False
            daily = self._read(self._path)
            self._ingest(daily)
            self._mtime = m
            return True
        except Exception:
            return False

    # ------------------------------------------------------------- time rule
    def obs_day(self, as_of: float) -> Optional[int]:
        """Last UTC day (unix day number) whose realized rate is known at ``as_of``."""
        if self.last_day is None:
            return None
        return min(_day(float(as_of) - self.observe_lag_s) - 1, self.last_day)

    def final_through(self, as_of: float) -> float:
        """End of the last observable day: yield before this is final at ``as_of``.
        The ledger commits accrual only up to here (``accrue_yield(commit_to=...)``)."""
        od = self.obs_day(as_of)
        return float("-inf") if od is None else float((od + 1) * DAY_S)

    # ----------------------------------------------------------------- rates
    @staticmethod
    def _lookup(series, day: int) -> Optional[float]:
        if series is None:
            return None
        days, rates = series
        i = bisect.bisect_left(days, day)
        if i < len(days) and days[i] == day:
            return rates[i]
        return None

    def realized_rate(self, netuid: int, day: int, hotkey: Optional[str] = None) -> Optional[float]:
        """Rate for one day by the crediting rule (validator, else pooled); None if neither."""
        if self.mode == "validator" and hotkey:
            r = self._lookup(self._hk.get((int(netuid), hotkey)), day)
            if r is not None:
                return r
        return self._lookup(self._pool.get(int(netuid)), day)

    def known_rate(self, netuid: int, day: int, hotkey: Optional[str], as_of: float) -> float:
        """Rate credited for ``day`` as known at ``as_of``: realized if observable,
        else the last realized rate up to ``obs_day(as_of)`` (provisional)."""
        od = self.obs_day(as_of)
        if od is None:
            return 0.0
        if day <= od:
            r = self.realized_rate(netuid, day, hotkey)
            return 0.0 if r is None else r
        for back in range(0, 8):          # last known: walk back from obs_day
            r = self.realized_rate(netuid, od - back, hotkey)
            if r is not None:
                return r
        return 0.0

    def accrued_yield(self, netuid: int, alpha_qty: float, entry_time: float, now: float,
                      generation_origin: Optional[float] = None,
                      hotkey: Optional[str] = None, as_of: Optional[float] = None) -> float:
        """Alpha earned on ``alpha_qty`` over ``[entry_time, now]``, each day at the
        rate known at ``as_of`` (default ``now``)."""
        if alpha_qty <= 0 or not math.isfinite(alpha_qty) or now <= entry_time:
            return 0.0
        self.maybe_reload()
        as_of = float(now) if as_of is None else float(as_of)
        origin = float(entry_time) if generation_origin is None else min(
            float(generation_origin), float(entry_time))
        end = generation_end(self._reregs, netuid, origin, float(now))
        if end <= entry_time:
            return 0.0
        last_paid_day = _day(end) if end >= now else _day(end) - 1
        total, t = 0.0, float(entry_time)
        while t < end:
            dn = _day(t)
            nxt = min(end, float((dn + 1) * DAY_S))
            if dn <= last_paid_day:
                total += self.known_rate(netuid, dn, hotkey, as_of) * (nxt - t) / DAY_S
            t = nxt
        return alpha_qty * total

    def rate(self, netuid: int, hotkey: Optional[str] = None,
             as_of: Optional[float] = None) -> YieldRate:
        """The rate in force now (provisional), for logging (``YieldRate``)."""
        self.maybe_reload()
        t = time.time() if as_of is None else float(as_of)
        r = self.known_rate(netuid, _day(t), hotkey, t)
        return YieldRate(netuid=int(netuid), rate_per_day=r,
                         source=AlphaYieldSource.ONCHAIN_REALIZED, observed_at=t)

    # ------------------------------------------------------------- decisions
    def trailing_rate(self, netuid: int, as_of: float, hotkey: Optional[str] = None,
                      window_days: Optional[int] = None) -> tuple[Optional[float], int]:
        """Mean realized rate per day over the ``window_days`` days ending at
        ``obs_day(as_of)`` and the number of valid days. ``hotkey`` None gives
        the pooled subnet rate."""
        self.maybe_reload()
        od = self.obs_day(as_of)
        if od is None:
            return None, 0
        w = int(window_days or self.rule.window_days)
        series = self._pool.get(int(netuid)) if hotkey is None else self._hk.get((int(netuid), hotkey))
        if not series:
            return None, 0
        days, rates = series
        lo = bisect.bisect_left(days, od - w + 1)
        hi = bisect.bisect_right(days, od)
        vals = rates[lo:hi]
        if not vals:
            return None, 0
        return float(sum(vals) / len(vals)), len(vals)

    def eligible(self, netuid: int, as_of: float) -> pd.DataFrame:
        """Validators passing the rule at ``as_of``, best first."""
        od = self.obs_day(as_of)
        g = self._cand.get((int(netuid), od)) if od is not None else None
        if g is None or not len(g):
            return pd.DataFrame(columns=["hotkey", "trail", "n_valid", "trail30", "take",
                                         "stake_share"])
        r = self.rule
        m = (g["n_valid"] >= r.min_valid_days) & g["trail"].notna()
        if r.require_permit:
            m &= g["permit"].fillna(False).astype(bool)
        m &= g["stake_share"].fillna(0.0) >= r.min_stake_share
        take_ok = g["take"] <= r.max_take
        m &= take_ok | (g["take"].isna() & r.unknown_take_passes)
        e = g[m]
        return e.sort_values(["trail", "hotkey"], ascending=[False, True])

    def choose_validator(self, netuid: int, as_of: float) -> Optional[ValidatorPick]:
        """The rule's validator for ``netuid`` at ``as_of`` (None if none passes)."""
        self.maybe_reload()
        od = self.obs_day(as_of)
        if od is None:
            return None
        key = (int(netuid), od)
        if key in self._pick_cache:
            return self._pick_cache[key]
        e = self.eligible(netuid, as_of)
        pick = None
        if len(e):
            b = e.iloc[0]
            pick = ValidatorPick(
                netuid=int(netuid), hotkey=str(b["hotkey"]), trailing_rate=float(b["trail"]),
                trailing_rate_30d=None if pd.isna(b["trail30"]) else float(b["trail30"]),
                n_valid_days=int(b["n_valid"]), n_eligible=int(len(e)),
                take=None if pd.isna(b["take"]) else float(b["take"]),
                stake_share=float(b["stake_share"]), obs_day=od)
        self._pick_cache[key] = pick
        return pick

    def choose_hotkey(self, netuid: int, as_of: float) -> Optional[str]:
        """Hotkey of :meth:`choose_validator` (the hook the engine and paper call)."""
        p = self.choose_validator(netuid, as_of)
        return None if p is None else p.hotkey


def onchain_model_from_env(rule: ValidatorRule = ValidatorRule(), **kw) -> Optional[RealizedYieldModel]:
    """``RealizedYieldModel`` from ``ONCHAIN_YIELD_TABLE`` (path), or None when unset.

    ``ONCHAIN_YIELD_MODE`` (validator|pooled) and ``ONCHAIN_YIELD_OBSERVE_LAG_S``
    override the defaults.
    """
    path = os.environ.get("ONCHAIN_YIELD_TABLE")
    if not path:
        return None
    kw.setdefault("mode", os.environ.get("ONCHAIN_YIELD_MODE", "validator"))
    if os.environ.get("ONCHAIN_YIELD_OBSERVE_LAG_S"):
        kw.setdefault("observe_lag_s", float(os.environ["ONCHAIN_YIELD_OBSERVE_LAG_S"]))
    kw.setdefault("keep_days", 60)
    return RealizedYieldModel(path, rule=rule, **kw)
