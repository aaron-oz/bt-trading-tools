"""Daily alpha yield rates measured on chain, as a rate table for ``HistoricalYieldModel``.

Input: a daily table with one row per (date, netuid, hotkey) holding the
realized growth of the validator's share price over that UTC day,

    P = TotalHotkeyAlpha / TotalHotkeyShares            (SubtensorModule storage)
    rate_per_day(d) = (P(end of d) / P(start of d) - 1) * 86400 / seconds(d)

Staking and unstaking move alpha and shares together, so P moves only when
dividends (after the validator's take) are credited; ``rate_per_day`` is what
an unmoved staker on that validator earned that day. Required columns:
``date``, ``netuid``, ``hotkey``, ``rate_per_day``, ``total_hotkey_alpha``
(stake at the start of the day, used as the weight) and ``valid`` (bool; rows
with ``valid`` False are ignored: missing endpoints, a subnet re-registration
inside the day, negative or implausible growth). Optional: ``permit`` and
``stake_share`` (used only by the on-chain selection rule).

The output has the columns of :func:`bt_trading_tools.alpha_yield_history.build_rate_table`
(``netuid``, ``date``, ``rate_per_day``) plus diagnostics, so it plugs into
``HistoricalYieldModel`` unchanged.

Timing. A modeled history rate on date d is a trailing APY known on d. An
on-chain rate on date d is the yield actually credited DURING d. For crediting
accrual over an interval inside d that is the right number; used as a trading
signal it would look ahead by up to one day. Use it for accrual only.

Selection (which validators are pooled per subnet and date):

``validator_sets`` given (rows ``netuid``, ``date``, ``hotkey``, e.g. from
    :func:`bt_trading_tools.alpha_yield_history.select_validators`): exactly
    those validators. Dates or subnets with no set fall back to the on-chain
    rule only if ``fallback_onchain`` is True.
on-chain rule: validators with a permit and ``stake_share >= min_stake_share``,
    the ``top_k`` by stake. The history rule ranks by 30-day APY and filters on
    uptime, which the chain does not store directly; the two sets differ.

Pooling: stake-weighted mean of ``rate_per_day``, weights ``total_hotkey_alpha``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

REQUIRED_COLUMNS = ["date", "netuid", "hotkey", "rate_per_day", "total_hotkey_alpha", "valid"]


def load_onchain_yield_daily(path: Union[str, Path]) -> pd.DataFrame:
    """Read a daily on-chain yield table (parquet or csv); ``date`` as UTC midnight."""
    path = Path(path)
    d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"on-chain yield table lacks columns {missing}")
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.normalize()
    return d


def _norm_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.normalize()


def onchain_selection(daily: pd.DataFrame, min_stake_share: float = 0.02,
                      top_k: int = 20) -> pd.DataFrame:
    """(netuid, date, hotkey) chosen by the on-chain rule (module docstring)."""
    d = daily
    if "permit" in d.columns:
        d = d[d["permit"].fillna(False).astype(bool)]
    if "stake_share" in d.columns:
        d = d[d["stake_share"] >= min_stake_share]
    d = d.sort_values(["netuid", "date", "total_hotkey_alpha"], ascending=[True, True, False])
    return d.groupby(["netuid", "date"]).head(top_k)[["netuid", "date", "hotkey"]]


def build_onchain_rate_table(
    daily: pd.DataFrame,
    validator_sets: Optional[pd.DataFrame] = None,
    fallback_onchain: bool = True,
    min_stake_share: float = 0.02,
    top_k: int = 20,
) -> pd.DataFrame:
    """Per (netuid, date) stake-weighted on-chain rate per day.

    Returns ``netuid``, ``date``, ``rate_per_day`` and diagnostics
    ``n_validators`` (pooled), ``n_selected`` (selected, valid or not),
    ``alpha_pooled`` (sum of weights) and ``selection`` (``"given"`` or
    ``"onchain"``). A (netuid, date) whose selected validators have no valid
    row is absent (``HistoricalYieldModel`` then carries the last earlier rate).
    """
    d = daily.copy()
    d["date"] = _norm_dates(d["date"])
    d["netuid"] = d["netuid"].astype(int)
    parts = []
    if validator_sets is not None:
        vs = validator_sets[["netuid", "date", "hotkey"]].copy()
        vs["date"] = _norm_dates(vs["date"])
        vs["netuid"] = vs["netuid"].astype(int)
        vs = vs.drop_duplicates()
        given = vs.assign(selection="given")
        parts.append(given)
    if validator_sets is None or fallback_onchain:
        oc = onchain_selection(d, min_stake_share, top_k).assign(selection="onchain")
        if validator_sets is not None:
            have = vs[["netuid", "date"]].drop_duplicates().assign(_h=1)
            oc = oc.merge(have, on=["netuid", "date"], how="left")
            oc = oc[oc["_h"].isna()].drop(columns="_h")
        parts.append(oc)
    sel = pd.concat(parts, ignore_index=True)
    n_sel = sel.groupby(["netuid", "date"]).size().rename("n_selected")
    m = sel.merge(d, on=["netuid", "date", "hotkey"], how="inner")
    m = m[m["valid"].astype(bool) & (m["total_hotkey_alpha"] > 0)]
    m = m.assign(w=m["total_hotkey_alpha"] * m["rate_per_day"])
    g = m.groupby(["netuid", "date"]).agg(
        w=("w", "sum"), alpha_pooled=("total_hotkey_alpha", "sum"),
        n_validators=("hotkey", "size"), selection=("selection", "first")).reset_index()
    g["rate_per_day"] = g["w"] / g["alpha_pooled"]
    g = g.merge(n_sel, on=["netuid", "date"], how="left")
    return g[["netuid", "date", "rate_per_day", "n_validators", "n_selected",
              "alpha_pooled", "selection"]].sort_values(["netuid", "date"]).reset_index(drop=True)


def hotkey_rate_table(daily: pd.DataFrame, hotkey: str) -> pd.DataFrame:
    """Rate table for staking through one validator ``hotkey`` on every subnet."""
    d = daily[(daily["hotkey"] == hotkey) & daily["valid"].astype(bool)].copy()
    d["date"] = _norm_dates(d["date"])
    return d[["netuid", "date", "rate_per_day"]].sort_values(["netuid", "date"]).reset_index(drop=True)
