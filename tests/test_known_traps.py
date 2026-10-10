"""Regression tests for two measured, silent backtest traps (docs/known_traps.md).

1. Same realism seed across independent engines (panel windows replay one draw).
2. Pool reserves that disagree with the tick price (hourly price + daily reserves).
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from bt_trading_tools.backtest import (
    BacktestEngine,
    Order,
    SubnetTick,
    TickData,
    derive_window_seed,
    panel_forward_returns,
    run_basket_window,
    seeded_engine_factory,
)
from bt_trading_tools.backtest import engine as engine_mod
from bt_trading_tools.data.ticks import reserve_price_gap, rescale_reserves_to_price


def flat_ticks(n=60, step=900, tao=5000.0, price=0.01, netuid=7, gap=0.0, t0=1_700_000_000):
    """n ticks of one subnet at an unchanging price; reserves imply price*(1+gap)."""
    alpha = tao / (price * (1.0 + gap))
    return [TickData(timestamp=t0 + i * step,
                     subnets={netuid: SubnetTick(netuid=netuid, price=price, tao_pool=tao, alpha_pool=alpha, generation=0)})
            for i in range(n)]


class BuyEveryTick:
    def __init__(self, netuid, spend=0.05):
        self.netuid, self.spend = netuid, spend

    def on_tick(self, tick, positions, capital, portfolio_value):
        return [Order(self.netuid, "buy", tao_amount=self.spend)] if capital >= self.spend else []


# ---------- trap 1: realism seeds ----------


def test_derive_window_seed_is_deterministic_and_window_specific():
    a = derive_window_seed(0, 1_700_000_000, [5, 7])
    assert a == derive_window_seed(0, 1_700_000_000, [7, 5])            # order of netuids does not matter
    assert a == derive_window_seed(0, 1_700_000_000, [5, 7])            # reproducible
    assert a != derive_window_seed(0, 1_700_000_900, [5, 7])            # different entry time
    assert a != derive_window_seed(1, 1_700_000_000, [5, 7])            # different base seed
    assert a != derive_window_seed(0, 1_700_000_000, [5])              # different basket


def test_panel_windows_do_not_share_one_realism_draw():
    """With the old behavior every window got seed 0 and identical noise, so identical
    windows returned identical net returns. Now they must differ, and reruns must match."""
    ticks = flat_ticks()
    entries = [(ticks[i].timestamp, {7: 1.0}) for i in range(0, 40, 2)]
    r1 = [r["net_return"] for r in panel_forward_returns(ticks, entries, 1800) if r["net_return"] is not None]
    r2 = [r["net_return"] for r in panel_forward_returns(ticks, entries, 1800) if r["net_return"] is not None]
    assert len(r1) >= 15
    assert r1 == r2                                                      # deterministic
    assert np.std(r1) > 1e-4                                             # windows differ (old: std == 0)
    assert len(set(round(x, 10) for x in r1)) > len(r1) // 2


def test_explicit_base_seed_is_reproducible_and_changes_the_draws():
    ticks = flat_ticks()
    entries = [(ticks[i].timestamp, {7: 1.0}) for i in range(0, 30, 3)]
    a = panel_forward_returns(ticks, entries, 1800, realism_rng_seed=0)
    b = panel_forward_returns(ticks, entries, 1800, realism_rng_seed=123)
    assert [r["net_return"] for r in a] != [r["net_return"] for r in b]


def test_mean_of_one_trade_windows_matches_a_seed_sweep_not_one_draw():
    """The mean over many windows should sit near the mean over many seeds of one window,
    not at the value of a single (seed 0) draw."""
    ticks = flat_ticks(n=120)
    entry = ticks[3].timestamp
    seed_sweep = []
    for s in range(60):
        res = run_basket_window(ticks, {7: 1.0}, entry, entry + 1800, engine_factory=seeded_engine_factory(s * 1000))
        if res[7]["net_return"] is not None:
            seed_sweep.append(res[7]["net_return"])
    entries = [(ticks[i].timestamp, {7: 1.0}) for i in range(2, 100)]
    panel = [r["net_return"] for r in panel_forward_returns(ticks, entries, 1800) if r["net_return"] is not None]
    assert len(seed_sweep) > 40 and len(panel) > 80
    assert abs(np.mean(panel) - np.mean(seed_sweep)) < 0.0015


def test_seeded_engine_factory_gives_distinct_seeds():
    f = seeded_engine_factory(100, capital=1.0)
    seeds = [f().realism_rng_seed for _ in range(5)]
    assert seeds == [101, 102, 103, 104, 105]


def test_engine_warns_when_many_few_fill_runs_share_a_seed(monkeypatch):
    monkeypatch.setattr(engine_mod, "_SEED_FEW_FILL_RUNS", {})
    monkeypatch.setattr(engine_mod, "_SEED_WARNED", set())
    ticks = flat_ticks(n=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(engine_mod.SAME_SEED_WARN_AFTER - 1):
            BacktestEngine(capital=2.0, realism_rng_seed=42).run(ticks, _one_buy())
    with pytest.warns(RuntimeWarning, match="realism_rng_seed=42"):
        BacktestEngine(capital=2.0, realism_rng_seed=42).run(ticks, _one_buy())


def _one_buy():
    class OneBuy:
        done = False

        def on_tick(self, tick, positions, capital, portfolio_value):
            if not self.done:
                self.done = True
                return [Order(7, "buy", tao_amount=0.05)]
            return []
    return OneBuy()


def test_engine_does_not_warn_for_distinct_seeds(monkeypatch):
    monkeypatch.setattr(engine_mod, "_SEED_FEW_FILL_RUNS", {})
    monkeypatch.setattr(engine_mod, "_SEED_WARNED", set())
    ticks = flat_ticks(n=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for s in range(engine_mod.SAME_SEED_WARN_AFTER + 10):
            BacktestEngine(capital=2.0, realism_rng_seed=s).run(ticks, _one_buy())


# ---------- trap 2: reserves versus price ----------


def test_rescale_preserves_k_and_hits_target_price():
    tao, alpha = rescale_reserves_to_price(5000.0, 400_000.0, 0.0125 * 1.2)
    assert math.isclose(tao * alpha, 5000.0 * 400_000.0, rel_tol=1e-12)
    assert math.isclose(tao / alpha, 0.0125 * 1.2, rel_tol=1e-12)


def test_reserve_price_gap_summarizes_consistency():
    clean = reserve_price_gap(flat_ticks(gap=0.0))
    stale = reserve_price_gap(flat_ticks(gap=-0.2))
    assert clean["median_abs_gap"] < 1e-9 and clean["share_over_5pct"] == 0.0
    assert stale["median_abs_gap"] > 0.15 and stale["share_over_5pct"] == 1.0


def test_engine_flags_orders_on_inconsistent_reserves():
    bad = BacktestEngine(capital=5.0).run(flat_ticks(n=15, gap=-0.2), BuyEveryTick(7))
    assert bad.orders_checked >= 10 and bad.orders_on_inconsistent_reserves == bad.orders_checked
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        good = BacktestEngine(capital=5.0, realism_rng_seed=11).run(flat_ticks(n=15, gap=0.0), BuyEveryTick(7))
    assert good.orders_on_inconsistent_reserves == 0


def test_engine_warns_when_most_orders_sit_on_stale_reserves():
    with pytest.warns(RuntimeWarning, match="reserves"):
        BacktestEngine(capital=5.0, realism_rng_seed=3).run(flat_ticks(n=15, gap=-0.2), BuyEveryTick(7))


def _write_parquet_fixture(tmp_path, jump=0.25):
    """One subnet, daily reserve snapshot at 00:00 consistent with price 0.01, hourly closes jump by `jump` later that day."""
    day0 = pd.Timestamp("2026-01-10 00:00", tz="UTC")
    hours = pd.date_range(day0 + pd.Timedelta(hours=1), periods=10, freq="h")
    close = [0.01] * 3 + [0.01 * (1 + jump)] * 7
    ohlcv = pd.DataFrame({"time": hours, "open": close, "high": close, "low": close, "close": close,
                          "volume": 1.0, "n_trades": 3, "net_flow_tao": 0.0, "netuid": 9})
    pool = pd.DataFrame({"timestamp": [day0, day0 + pd.Timedelta(days=1)], "netuid": 9,
                         "total_tao": [5000e9, 5100e9], "alpha_in_pool": [500_000e9, 500_000e9],
                         "startup_mode": False})
    o, p = tmp_path / "ohlcv.parquet", tmp_path / "pool.parquet"
    ohlcv.to_parquet(o)
    pool.to_parquet(p)
    return o, p


@pytest.fixture
def parquet_mod(monkeypatch):
    pytest.importorskip("pyarrow")
    import bt_trading_tools.data.ticks as m
    monkeypatch.setattr(m, "_PARQUET_WARNED", False)
    return m


def test_load_parquet_ticks_defaults_are_the_historical_behavior_and_warn_once(tmp_path, parquet_mod):
    o, p = _write_parquet_fixture(tmp_path)
    with pytest.warns(RuntimeWarning, match="known issues"):
        ticks = parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, stamp_identity=False)
    late = ticks[-1].subnets[9]
    assert late.signals["reserve_gap_raw"] == pytest.approx(1 / 1.25 - 1, rel=1e-6)    # daily reserves left stale (implied price 20% low)
    assert reserve_price_gap(ticks)["max_abs_gap"] > 0.15
    assert late.signals["reserve_age_s"] > 0
    with warnings.catch_warnings():                                                    # second call in the same process: silent
        warnings.simplefilter("error", RuntimeWarning)
        parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, stamp_identity=False)


def test_load_parquet_ticks_rescale_is_opt_in_and_matches_price_at_constant_k(tmp_path, parquet_mod):
    o, p = _write_parquet_fixture(tmp_path)
    with pytest.warns(RuntimeWarning):
        ticks = parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, stamp_identity=False, reserves="rescale")
    late = ticks[-1].subnets[9]
    assert math.isclose(late.tao_pool / late.alpha_pool, late.price, rel_tol=1e-9)
    assert math.isclose(late.tao_pool * late.alpha_pool, 5000e0 * 500_000e0, rel_tol=1e-9)
    assert reserve_price_gap(ticks)["max_abs_gap"] < 1e-9


def test_bar_label_end_stamps_ticks_with_the_bar_end(tmp_path, parquet_mod):
    """OHLCV time is the bar START; its close is the price at the bar END. 'end' moves the label so the close is known at its timestamp."""
    o, p = _write_parquet_fixture(tmp_path)
    ohlcv = pd.read_parquet(o)
    first_bar = int(pd.Timestamp(ohlcv.time.min()).timestamp())
    with pytest.warns(RuntimeWarning):
        start = parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, stamp_identity=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        end = parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, stamp_identity=False, bar_label="end")
    assert start[0].timestamp == first_bar
    assert end[0].timestamp == first_bar + 3600
    assert [t.subnets[9].price for t in start] == [t.subnets[9].price for t in end]          # same closes, later labels
    assert end[-1].timestamp == start[-1].timestamp + 3600


def test_load_parquet_ticks_rejects_unknown_options(tmp_path, parquet_mod):
    o, p = _write_parquet_fixture(tmp_path)
    with pytest.raises(ValueError):
        parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, reserves="bogus")
    with pytest.raises(ValueError):
        parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p, bar_label="middle")


def test_explicit_reregistrations_stamp_at_event_time_not_floored_to_midnight(tmp_path, parquet_mod):
    o, p = _write_parquet_fixture(tmp_path)
    event = pd.Timestamp("2026-01-10 06:30", tz="UTC")
    ticks = parquet_mod.load_parquet_ticks("2026-01-10", "2026-01-11", ohlcv_parquet=o, pool_parquet=p,
                                           reregistrations={9: [event]})
    gens = {t.timestamp: t.subnets[9].generation for t in ticks}
    assert gens[int(pd.Timestamp("2026-01-10 05:00", tz="UTC").timestamp())] == 0      # before the event: still the old subnet
    assert gens[int(pd.Timestamp("2026-01-10 07:00", tz="UTC").timestamp())] == 1      # after it: new subnet


def test_reregistrations_from_csvs_finds_a_staked_alpha_collapse(tmp_path):
    from bt_trading_tools.utils.lifecycle import reregistrations_from_csvs
    t = pd.date_range("2026-01-10", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": t, "netuid": 7, "alpha_in": [100, 100, 100, 50, 50, 50],
                       "alpha_out": [1000, 1000, 1000, 10, 10, 10]})
    f = tmp_path / "snap.csv"
    df.to_csv(f, index=False)
    ev = reregistrations_from_csvs(f)
    assert list(ev) == [7] and ev[7][0] == t[3]
