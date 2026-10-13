"""RealizedYieldModel: causal decisions, provisional credit with true-up,
validator selection determinism, engine integration (2026-10-12)."""
import math

import numpy as np
import pandas as pd
import pytest

from bt_trading_tools import ledger
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, Position, SubnetTick, TickData
from bt_trading_tools.realized_yield import RealizedYieldModel, ValidatorRule

DAY = 86400
D0 = pd.Timestamp("2026-01-01", tz="UTC")
T0 = int(D0.timestamp())
LAG = 7200.0


def table(n_days=10, rates=None, netuid=1, takes=None, drop=()):
    """Daily table: validators A, B, C on ``netuid``; ``rates[hk][i]`` per day."""
    rates = rates or {"A": [0.001] * n_days, "B": [0.002] * n_days, "C": [0.0015] * n_days}
    rows = []
    for i in range(n_days):
        for hk, rs in rates.items():
            if (hk, i) in drop:
                continue
            r = rs[i]
            rows.append({
                "date": D0 + pd.Timedelta(days=i), "netuid": netuid, "hotkey": hk,
                "rate_per_day": r, "valid": r is not None and np.isfinite(r),
                "permit": True, "stake_share": 1 / len(rates),
                "total_hotkey_alpha": {"A": 3000.0, "B": 1000.0, "C": 2000.0}.get(hk, 1000.0),
                "take": (takes or {}).get(hk, 0.09),
            })
    return pd.DataFrame(rows)


def model(df, **kw):
    kw.setdefault("observe_lag_s", LAG)
    kw.setdefault("rule", ValidatorRule(window_days=3, min_valid_days=2))
    return RealizedYieldModel(df, reload_check_s=None, **kw)


# ---------------------------------------------------------------- timing rule
def test_obs_day_and_final_through():
    m = model(table())
    d0 = T0 // DAY
    # day 2 ends at T0 + 3 days; observable from +2 h
    assert m.obs_day(T0 + 3 * DAY + LAG - 1) == d0 + 1
    assert m.obs_day(T0 + 3 * DAY + LAG) == d0 + 2
    assert m.final_through(T0 + 3 * DAY + LAG) == T0 + 3 * DAY
    # capped by the table's last date (day 9)
    assert m.obs_day(T0 + 100 * DAY) == d0 + 9


# ---------------------------------------------------------------- no lookahead
@pytest.mark.parametrize("t_off", [3 * DAY + LAG, 5 * DAY + 3600, 6 * DAY + LAG + 10])
def test_decisions_ignore_unobservable_rows(t_off):
    """Changing or removing every row after obs_day(T) changes no decision at T."""
    base = table()
    T = T0 + t_off
    m1 = model(base)
    od = m1.obs_day(T)
    future = (base["date"] - D0).dt.days + T0 // DAY > od
    perturbed = base.copy()
    perturbed.loc[future, "rate_per_day"] = perturbed.loc[future, "rate_per_day"] * 50 + 0.01
    perturbed.loc[future & (perturbed["hotkey"] == "A"), "rate_per_day"] = 0.019
    m2 = model(perturbed)
    m3 = model(base[~future])
    for m in (m2, m3):
        assert m.choose_validator(1, T) == m1.choose_validator(1, T)
        for hk in (None, "A", "B", "C"):
            assert m.trailing_rate(1, T, hk) == m1.trailing_rate(1, T, hk)
        # accrual as of T also only uses observable days
        a = m.accrued_yield(1, 100.0, T0 + DAY, T, hotkey="A", as_of=T)
        assert a == pytest.approx(m1.accrued_yield(1, 100.0, T0 + DAY, T, hotkey="A", as_of=T))


def test_trailing_rate_window():
    rates = {"A": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006], "B": [0.0] * 6}
    m = model(table(6, rates))
    T = T0 + 5 * DAY + LAG          # obs_day = day 4
    mean, n = m.trailing_rate(1, T, "A")          # days 2..4
    assert n == 3 and mean == pytest.approx((0.003 + 0.004 + 0.005) / 3)


# ---------------------------------------------------------------- selection rule
def test_selection_picks_highest_trailing_and_is_deterministic():
    m = model(table())
    T = T0 + 5 * DAY + LAG
    p = m.choose_validator(1, T)
    assert p.hotkey == "B" and p.n_eligible == 3
    # row order does not matter
    shuffled = table().sample(frac=1.0, random_state=7)
    assert model(shuffled).choose_validator(1, T) == p


def test_selection_tie_breaks_on_hotkey():
    rates = {"Z": [0.002] * 6, "B": [0.002] * 6, "A": [0.001] * 6}
    m = model(table(6, rates))
    assert m.choose_hotkey(1, T0 + 5 * DAY + LAG) == "B"


def test_selection_filters_take_permit_stake_and_valid_days():
    df = table(takes={"B": 0.18})
    m = model(df, rule=ValidatorRule(window_days=3, min_valid_days=2, max_take=0.12))
    T = T0 + 5 * DAY + LAG
    assert m.choose_hotkey(1, T) == "C"           # B fails the take cap
    df2 = table(drop={("B", 2), ("B", 3), ("B", 4)})   # B has 0 valid days in window
    assert model(df2).choose_hotkey(1, T) == "C"
    df3 = table()
    df3.loc[df3.hotkey == "B", "permit"] = False
    assert model(df3).choose_hotkey(1, T) == "C"
    df4 = table()
    df4.loc[df4.hotkey == "B", "stake_share"] = 0.01
    assert model(df4).choose_hotkey(1, T) == "C"


# ---------------------------------------------------------------- crediting
def test_validator_vs_pooled_and_fallback():
    m = model(table())
    a = m.accrued_yield(1, 1000.0, T0 + DAY, T0 + 2 * DAY, hotkey="B", as_of=T0 + 9 * DAY)
    assert a == pytest.approx(1000.0 * 0.002)
    # pooled: stake-weighted over A, B, C (3000, 1000, 2000)
    pooled = (3000 * 0.001 + 1000 * 0.002 + 2000 * 0.0015) / 6000
    p = m.accrued_yield(1, 1000.0, T0 + DAY, T0 + 2 * DAY, as_of=T0 + 9 * DAY)
    assert p == pytest.approx(1000.0 * pooled)
    mp = model(table(), mode="pooled")
    assert mp.accrued_yield(1, 1000.0, T0 + DAY, T0 + 2 * DAY, hotkey="B",
                            as_of=T0 + 9 * DAY) == pytest.approx(p)
    # a hotkey with no row that day falls back to the pooled rate
    m2 = model(table(drop={("B", 1)}))
    pooled2 = (3000 * 0.001 + 2000 * 0.0015) / 5000
    assert m2.accrued_yield(1, 1000.0, T0 + DAY, T0 + 2 * DAY, hotkey="B",
                            as_of=T0 + 9 * DAY) == pytest.approx(1000.0 * pooled2)


def test_provisional_credit_uses_last_known_rate():
    rates = {"A": [0.001, 0.001, 0.001, 0.005, 0.005], "B": [0.0] * 5}
    m = model(table(5, rates))
    # as of day 3 at noon: obs_day = 2, so day 3 is provisional at 0.001
    T = T0 + 3 * DAY + DAY // 2
    a = m.accrued_yield(1, 1000.0, T0 + 3 * DAY, T, hotkey="A", as_of=T)
    assert a == pytest.approx(1000.0 * 0.001 * 0.5)
    # once day 3 is observable the same interval is credited at 0.005
    late = T0 + 4 * DAY + LAG
    assert m.accrued_yield(1, 1000.0, T0 + 3 * DAY, T, hotkey="A", as_of=late) == \
        pytest.approx(1000.0 * 0.005 * 0.5)


def _ledger_walk(m, ticks, hotkey="A", alpha=1000.0, entry=None):
    pos = Position(netuid=1, entry_price=1.0, alpha_qty=alpha, tao_cost=alpha,
                   entry_time=entry if entry is not None else ticks[0],
                   yield_anchor_time=None, yield_accrued=0.0,
                   yield_accrued_to=entry if entry is not None else ticks[0],
                   validator_hotkey=hotkey)

    def yfn(netuid, a, since, now, origin=None, hotkey=None, as_of=None):
        return ledger.call_yield_model(m, netuid, a, since, now, origin=origin,
                                       hotkey=hotkey, as_of=as_of)

    marks = []
    for t in ticks:
        ledger.accrue_yield(pos, t, yfn, commit_to=m.final_through(t))
        marks.append((t, ledger.accrued_yield(pos, t, yfn), pos.yield_accrued,
                      pos.yield_accrued_to))
    return pos, marks, yfn


def test_true_up_converges_to_realized_and_never_revalues_committed():
    rates = {"A": [0.001, 0.003, 0.0005, 0.004, 0.002, 0.001, 0.002, 0.003],
             "B": [0.0] * 8}
    m = model(table(8, rates))
    ticks = [T0 + DAY + k * 3600 for k in range(0, 6 * 24 + 1)]     # hourly, 6 days
    pos, marks, _ = _ledger_walk(m, ticks)
    end = ticks[-1]
    exact = 1000.0 * sum(rates["A"][1:7])                         # days 1..6 fully held
    # at the end, days 1..5 are realized (obs_day = 5 at day 7 00:00 + LAG? no:
    # end = day 7 00:00, obs_day = 5); day 6 is provisional at day 5's rate
    od = m.obs_day(end) - T0 // DAY
    assert od == 5
    assert marks[-1][1] == pytest.approx(1000.0 * (sum(rates["A"][1:6]) + rates["A"][5]))
    # after every day is observable, the same position reads exactly realized
    late = end + 2 * DAY
    assert ledger.accrued_yield(pos, end, lambda *a, **k: m.accrued_yield(
        a[0], a[1], a[2], a[3], hotkey=k.get("hotkey"), as_of=late,
        generation_origin=k.get("origin"))) == pytest.approx(exact)
    # committed total only ever grows and is never revalued
    committed = [(c, to) for _, _, c, to in marks]
    for (c1, t1), (c2, t2) in zip(committed, committed[1:]):
        assert c2 >= c1 - 1e-15 and t2 >= t1
        if t2 == t1:
            assert c2 == c1
    # each committed stretch equals realized rates exactly
    for _, _, c, to in marks:
        if to > T0 + DAY:
            k = int((to - T0) // DAY)
            assert c == pytest.approx(1000.0 * sum(rates["A"][1:k]))


def test_true_up_path_equals_engine_without_lag_when_all_known():
    """With every day observable at all times (lag far negative), path accrual
    with commit equals the plain integral."""
    rates = {"A": [0.001, 0.003, 0.0005, 0.004, 0.002], "B": [0.0] * 5}
    m = model(table(5, rates), observe_lag_s=-30 * DAY)
    ticks = [T0 + k * 1800 for k in range(0, 4 * 48 + 1)]
    pos, marks, _ = _ledger_walk(m, ticks)
    assert marks[-1][1] == pytest.approx(1000.0 * sum(rates["A"][0:4]))


def test_legacy_position_keeps_its_old_total_at_switch():
    """A position whose path total was built by another model keeps that total;
    the new model only adds from yield_accrued_to (no equity jump at restart)."""
    m = model(table())
    t_switch = T0 + 5 * DAY + 3 * 3600
    pos = Position(netuid=1, entry_price=1.0, alpha_qty=1000.0, tao_cost=1000.0,
                   entry_time=T0, yield_anchor_time=T0, yield_accrued=7.5,
                   yield_accrued_to=t_switch)

    def yfn(netuid, a, since, now, origin=None, hotkey=None, as_of=None):
        return ledger.call_yield_model(m, netuid, a, since, now, origin=origin,
                                       hotkey=hotkey, as_of=as_of)

    ledger.assign_validator(pos, t_switch, m.choose_hotkey)
    assert pos.validator_hotkey == "B"
    ledger.accrue_yield(pos, t_switch, yfn, commit_to=m.final_through(t_switch))
    assert ledger.accrued_yield(pos, t_switch, yfn) == pytest.approx(7.5)
    t1 = t_switch + 3600
    ledger.accrue_yield(pos, t1, yfn, commit_to=m.final_through(t1))
    assert ledger.accrued_yield(pos, t1, yfn) == pytest.approx(7.5 + 1000 * 0.002 / 24)


def test_reregistration_stops_accrual():
    rereg = {1: [T0 + 3 * DAY + 3600]}
    m = model(table(), reregistrations=rereg)
    a = m.accrued_yield(1, 1000.0, T0 + DAY, T0 + 6 * DAY, hotkey="A", as_of=T0 + 9 * DAY)
    assert a == pytest.approx(1000.0 * 0.001 * 2)      # days 1 and 2 only


# ---------------------------------------------------------------- engine integration
def _ticks(n_hours, start):
    out = []
    for k in range(n_hours):
        ts = start + k * 3600
        out.append(TickData(timestamp=ts, subnets={
            1: SubnetTick(netuid=1, price=0.01, tao_pool=10000.0, alpha_pool=1e6,
                          generation=1)}))
    return out


class _BuyOnce:
    def __init__(self, hotkey=None):
        self.done = False
        self.hotkey = hotkey

    def on_tick(self, tick, positions, capital, portfolio_value):
        import copy
        if 1 in positions:
            self.seen = (tick.timestamp, copy.copy(positions[1]))
        if self.done:
            return []
        self.done = True
        sd = {"validator_hotkey": self.hotkey} if self.hotkey else None
        return [Order(netuid=1, side="buy", tao_amount=1.0, signal_data=sd)]


@pytest.mark.parametrize("hk,expect", [(None, "B"), ("C", "C")])
def test_engine_assigns_validator_and_credits_its_rate(hk, expect):
    m = model(table())
    eng = BacktestEngine(capital=10.0, yield_model=m, realism_rng_seed=1)
    ticks = _ticks(48, T0 + 5 * DAY)
    strat = _BuyOnce(hk)
    eng.run(ticks, strat)
    ts, pos = strat.seen
    assert ts == ticks[-1].timestamp
    assert pos.validator_hotkey == expect
    rate = {"B": 0.002, "C": 0.0015}[expect]
    entry = pos.entry_time
    acc = ledger.accrued_yield(pos, ts, eng._yield_fn)
    assert acc == pytest.approx(pos.alpha_qty * rate * (ts - entry) / DAY)
