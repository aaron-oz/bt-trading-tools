"""Price-outlier guard in load_parquet_ticks / rescale_tick_reserves, and the
rebirth-day case under exact re-registration events (docs/known_traps.md).

With reserves rescaled to the hourly close, a close printed by a different pool
(the new subnet's bootstrap pool right after a netuid re-registration, paired
with the old subnet's deep daily snapshot) becomes a deep pool the engine fills
and marks at. The fixtures mirror netuid 97 on 2026-03-13 UTC: old pool implied
0.00266 TAO per alpha, deregistered at 09:19, new subnet's hour closes at 40.09.
"""
import math
import warnings

import pandas as pd
import pytest

from bt_trading_tools.backtest import BacktestEngine, Order, SubnetTick, TickData
from bt_trading_tools.data import (
    DEFAULT_OUTLIER_MAX_RATIO,
    load_parquet_ticks,
    rescale_tick_reserves,
)

pytest.importorskip("pyarrow")

EVENT = pd.Timestamp("2026-03-13 09:19:36", tz="UTC")   # netuid 97's deregistration block

DAY0 = pd.Timestamp("2026-03-12 23:59:48", tz="UTC")
OLD_TAO, OLD_ALPHA = 3645.0, 1_369_120.0            # implied 0.002662, the netuid-97 snapshot
NEW_TAO, NEW_ALPHA = 0.2553, 0.2577                 # next day's tiny new pool, implied 0.99


def _fixture(tmp_path, with_root=True, rebirth_snapshot=True, extra_hours=0):
    """Hourly closes for netuid 97 on 2026-03-13 (and optionally netuid 0), daily snapshots."""
    hours = pd.date_range("2026-03-13 00:00", periods=24 + extra_hours, freq="h", tz="UTC")
    close = []
    for h in hours:
        if h.day == 13 and h.hour < 9:
            close.append(0.00265)                  # old subnet, consistent with the snapshot
        elif h.day == 13:
            close.append(40.09 if h.hour == 9 else 2.2)   # new subnet's bootstrap pool
        else:
            close.append(1.0)                       # next day: consistent with the new snapshot
    rows = [dict(time=h, open=c, high=c, low=c, close=c, volume=1.0, n_trades=3, net_flow_tao=0.0, netuid=97)
            for h, c in zip(hours, close)]
    if with_root:
        rows += [dict(time=h, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, n_trades=1,
                      net_flow_tao=0.0, netuid=0) for h in hours]
    ohlcv = pd.DataFrame(rows)
    snaps = [dict(timestamp=DAY0 - pd.Timedelta(days=1), netuid=97, total_tao=OLD_TAO * 1e9,
                  alpha_in_pool=OLD_ALPHA * 1e9, alpha_staked=5e15, startup_mode=False),
             dict(timestamp=DAY0, netuid=97, total_tao=OLD_TAO * 1e9, alpha_in_pool=OLD_ALPHA * 1e9,
                  alpha_staked=5e15, startup_mode=False)]
    if rebirth_snapshot:
        snaps.append(dict(timestamp=DAY0 + pd.Timedelta(days=1), netuid=97, total_tao=NEW_TAO * 1e9,
                          alpha_in_pool=NEW_ALPHA * 1e9, alpha_staked=1e8, startup_mode=False))
    if with_root:   # root: price fixed at 1.0, reserves imply about 4 (as in pool_history)
        snaps += [dict(timestamp=DAY0 + pd.Timedelta(days=d), netuid=0, total_tao=4e18, alpha_in_pool=1e18,
                       alpha_staked=1e18, startup_mode=False) for d in (-1, 0, 1)]
    pool = pd.DataFrame(snaps)
    o, p = tmp_path / "ohlcv.parquet", tmp_path / "pool.parquet"
    ohlcv.to_parquet(o)
    pool.to_parquet(p)
    return o, p


@pytest.fixture(autouse=True)
def _quiet_once_warning(monkeypatch):
    """Silence main's once-per-process 'known issues' RuntimeWarning so each test sees only its own warnings."""
    import bt_trading_tools.data.ticks as m
    monkeypatch.setattr(m, "_PARQUET_WARNED", True)


def _load(o, p, end="2026-03-14 23:00", **kw):
    kw.setdefault("reserves", "rescale")
    return load_parquet_ticks("2026-03-13", end, ohlcv_parquet=o, pool_parquet=p, **kw)


def _cells(ticks, netuid):
    return {t.timestamp: t.subnets[netuid] for t in ticks if netuid in t.subnets}


def test_rescale_drops_netuid97_like_close_and_warns(tmp_path):
    """Rescale path, guard on by default. The re-registration is not visible in the pool file (no next-day snapshot), so only
    the ratio test can catch the new subnet's closes paired with the old snapshot."""
    o, p = _fixture(tmp_path, rebirth_snapshot=False)
    report = []
    with pytest.warns(UserWarning, match=r"dropped 15 of 48 subnet-ticks \(1 subnets\)"):
        ticks = _load(o, p, end="2026-03-13 23:00", outlier_report=report, stamp_identity=False)
    assert DEFAULT_OUTLIER_MAX_RATIO == 3.0
    cells = _cells(ticks, 97)
    hour9 = int(pd.Timestamp("2026-03-13 09:00", tz="UTC").timestamp())
    assert hour9 not in cells                                       # the 40.09 close is gone
    assert len(report) == 15 and {r[1] for r in report} == {97}     # 09:00..23:00
    assert {r[4] for r in report} == {"ratio"}
    r9 = next(r for r in report if r[0] == hour9)
    assert r9[2] == pytest.approx(40.09) and r9[3] == pytest.approx(OLD_TAO / OLD_ALPHA)
    assert sorted(cells) == [hour9 - 3600 * k for k in range(9, 0, -1)]   # 00:00..08:00 kept
    for st in cells.values():
        assert abs(math.log(1.0 + st.signals["reserve_gap_raw"])) <= math.log(3.0)


def test_root_is_exempt(tmp_path):
    o, p = _fixture(tmp_path)
    with pytest.warns(UserWarning):
        ticks = _load(o, p)
    root = _cells(ticks, 0)
    assert len(root) == 24                                      # every root hour kept (ratio 0.25)
    assert all(st.price == 1.0 for st in root.values())


def test_guard_off_reproduces_the_trap(tmp_path):
    o, p = _fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ticks = _load(o, p, outlier_max_ratio=None)
    st = _cells(ticks, 97)[int(pd.Timestamp("2026-03-13 09:00", tz="UTC").timestamp())]
    # the old deep pool, moved to 40.09 TAO per alpha: about 4,620 TAO on the TAO side
    assert st.tao_pool / st.alpha_pool == pytest.approx(40.09)
    assert st.tao_pool == pytest.approx(math.sqrt(OLD_TAO * OLD_ALPHA * 40.09))
    assert st.tao_pool > OLD_TAO


def test_threshold_is_configurable_and_validated(tmp_path):
    o, p = _fixture(tmp_path, with_root=False)
    report = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _load(o, p, outlier_max_ratio=1e6, outlier_report=report)
    assert report == []                     # 40.09 / 0.00266 is about 15,000x: under 1e6, kept, no drops
    for bad in (1.0, 0.5, "auto"):
        with pytest.raises(ValueError):
            _load(o, p, outlier_max_ratio=bad)


def test_raw_default_keeps_historical_behavior_and_can_opt_in(tmp_path):
    """Main's default (reserves='raw') is unchanged: no ratio guard unless a threshold is passed."""
    o, p = _fixture(tmp_path, with_root=False)
    hour9 = int(pd.Timestamp("2026-03-13 09:00", tz="UTC").timestamp())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ticks = load_parquet_ticks("2026-03-13", "2026-03-14 23:00", ohlcv_parquet=o, pool_parquet=p)
    assert hour9 in _cells(ticks, 97)
    with pytest.warns(UserWarning, match="dropped"):
        ticks = _load(o, p, reserves="raw", outlier_max_ratio=3.0)
    assert hour9 not in _cells(ticks, 97)


def test_ratio_guard_is_causal(tmp_path):
    """Adding later data (more hours, a later snapshot) never changes which earlier hours are kept."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    o1, p1 = _fixture(tmp_path / "a", rebirth_snapshot=False)
    o2, p2 = _fixture(tmp_path / "b", extra_hours=24)
    with pytest.warns(UserWarning):
        short = _load(o1, p1, end="2026-03-13 23:00", stamp_identity=False)
    with pytest.warns(UserWarning):
        long = _load(o2, p2, stamp_identity=False)
    cut = int(pd.Timestamp("2026-03-13 23:00", tz="UTC").timestamp())
    a = {(t.timestamp, n) for t in short for n in t.subnets}
    b = {(t.timestamp, n) for t in long if t.timestamp <= cut for n in t.subnets}
    assert a == b



class _BuyThenHold:
    def __init__(self):
        self.done = False

    def on_tick(self, tick, positions, capital, portfolio_value):
        if not self.done and 97 in tick.subnets:
            self.done = True
            return [Order(97, "buy", tao_amount=1.0)]
        return []


def test_engine_no_longer_marks_at_the_bad_close(tmp_path):
    """End to end, stamping off so the rebirth guard in the engine is not what saves us:
    with the guard off, a 1 TAO position bought before the rebirth is marked at the
    40.09 pool on the same day; with the guard on it never sees that pool."""
    o, p = _fixture(tmp_path, with_root=False, rebirth_snapshot=False)
    end = "2026-03-13 23:00"

    def peak(ticks):
        eng = BacktestEngine(capital=10.0, yield_model=None)
        res = eng.run(ticks, _BuyThenHold())
        return max(e["total_equity"] for e in res.equity_curve)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = peak(_load(o, p, end=end, stamp_identity=False, outlier_max_ratio=None))
        good = peak(_load(o, p, end=end, stamp_identity=False))
    assert bad > 100.0          # 1 TAO of alpha bought at 0.00265 marked at 40.09
    assert good < 11.0          # capital 10 TAO, no absurd mark


# ---------- rebirth day: exact events (main's reregistrations=) instead of a pre-rebirth drop ----------
#
# The branch fix/trap-fixes-2026-10-08 had a default-on drop_pre_rebirth_reserves
# guard for the midnight-floored daily detector. These tests show that exact
# events cover the failure it targeted (a position opened earlier on the rebirth
# day marked at the new subnet), that the one remaining hole on the rescale path
# (the bar containing the event, under bar_label="start") is closed by the ratio
# guard, and that without exact events the gap remains unless the (now opt-in)
# drop_pre_rebirth_reserves is passed.


def _run(ticks):
    eng = BacktestEngine(capital=10.0, yield_model=None)
    res = eng.run(ticks, _BuyThenHold())
    return max(e["total_equity"] for e in res.equity_curve), res.trades


@pytest.mark.parametrize("reserves,bar_label", [("raw", "start"), ("raw", "end"), ("rescale", "end")])
def test_exact_event_closes_a_position_opened_earlier_on_the_rebirth_day(tmp_path, reserves, bar_label):
    """1 TAO bought at the first hour of the event day (old subnet) is refunded at the old
    pool when the event passes, and never marked at the new subnet."""
    o, p = _fixture(tmp_path, with_root=False, extra_hours=24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticks = _load(o, p, reserves=reserves, bar_label=bar_label, outlier_max_ratio=None,
                      reregistrations={97: [EVENT]})
    gens = {t.timestamp: t.subnets[97].generation for t in ticks if 97 in t.subnets}
    assert gens[min(gens)] == 0 and gens[max(gens)] == 1
    peak, trades = _run(ticks)
    assert peak < 11.0
    refunds = [t for t in trades if t.get("reason") == "dereg_refund"]
    assert len(refunds) == 1 and refunds[0]["tao_received"] == pytest.approx(1.0, rel=0.05)


def test_event_bar_under_start_label_is_the_remaining_hole_and_the_ratio_guard_closes_it(tmp_path):
    """bar_label='start': the 09:00 bar is labeled before the 09:19 event, so it is stamped as
    the old subnet, but its close (40.09) is the new pool's. Rescaled, the position is marked
    and refunded at that close. The ratio guard (default on when rescaling) removes the hour."""
    o, p = _fixture(tmp_path, with_root=False, extra_hours=24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unguarded = _load(o, p, outlier_max_ratio=None, reregistrations={97: [EVENT]})
        guarded = _load(o, p, reregistrations={97: [EVENT]})
    hour9 = int(pd.Timestamp("2026-03-13 09:00", tz="UTC").timestamp())
    assert _cells(unguarded, 97)[hour9].generation == 0
    bad_peak, bad_trades = _run(unguarded)
    assert bad_peak > 100.0
    good_peak, good_trades = _run(guarded)
    assert good_peak < 11.0
    refunds = [t for t in good_trades if t.get("reason") == "dereg_refund"]
    assert len(refunds) == 1 and refunds[0]["tao_received"] == pytest.approx(1.0, rel=0.05)


def test_daily_detector_without_exact_events_still_misses_the_close(tmp_path):
    """Documents main's known limitation (no fix offered here): with the daily detector the
    event is floored to midnight, the 00:00 purchase is stamped as the NEW subnet, and the
    engine never closes it; it is valued at the new pool the next day."""
    o, p = _fixture(tmp_path, with_root=False, extra_hours=24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticks = _load(o, p, reserves="raw")
    gens = {t.timestamp: t.subnets[97].generation for t in ticks if 97 in t.subnets}
    assert set(gens.values()) == {1}
    _, trades = _run(ticks)
    assert not [t for t in trades if t.get("reason") == "dereg_refund"]


def test_opt_in_pre_rebirth_drop_closes_the_daily_detector_gap(tmp_path):
    """drop_pre_rebirth_reserves=True with the daily detector: the event day's hours paired with the
    dead subnet's snapshot are dropped (all 24 here, including 00:00..08:00 whose closes agree with
    that snapshot, so the ratio test alone keeps them). One day of lookahead, as the stamping."""
    o, p = _fixture(tmp_path, extra_hours=24)
    report = []
    with pytest.warns(UserWarning, match=r"dropped 24 of 96 subnet-hours \(1 subnets\) whose reserve snapshot"):
        ticks = _load(o, p, reserves="raw", drop_pre_rebirth_reserves=True, outlier_report=report)
    assert [r[4] for r in report] == ["pre_rebirth_reserves"] * 24
    cells = _cells(ticks, 97)
    day14 = int(pd.Timestamp("2026-03-14 00:00", tz="UTC").timestamp())
    assert sorted(cells) == [day14 + 3600 * k for k in range(24)]
    assert {st.generation for st in cells.values()} == {1}
    assert len(_cells(ticks, 0)) == 48                               # root untouched


def test_pre_rebirth_drop_with_exact_events_cuts_at_the_event_time(tmp_path):
    """With exact events the cut is the event time, not midnight: the old subnet's hours before 09:19
    stay (generation 0), the new subnet's hours that day on the dead snapshot go."""
    o, p = _fixture(tmp_path, with_root=False, extra_hours=24)
    report = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticks = _load(o, p, reserves="raw", reregistrations={97: [EVENT]},
                      drop_pre_rebirth_reserves=True, outlier_report=report)
    hours = sorted(_cells(ticks, 97))
    day13 = int(pd.Timestamp("2026-03-13 00:00", tz="UTC").timestamp())
    assert [h for h in hours if h < day13 + 86400] == [day13 + 3600 * k for k in range(10)]   # 00:00..09:00
    assert len(report) == 14 and {r[4] for r in report} == {"pre_rebirth_reserves"}            # 10:00..23:00


def test_pre_rebirth_drop_is_off_by_default(tmp_path):
    o, p = _fixture(tmp_path, with_root=False, extra_hours=24)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ticks = _load(o, p, reserves="raw")
    assert len(_cells(ticks, 97)) == 48


# ---------- custom builders ----------


def _builder_ticks():
    t0 = 1_773_360_000
    old = lambda price: SubnetTick(netuid=97, price=price, tao_pool=OLD_TAO, alpha_pool=OLD_ALPHA, signals={})
    root = lambda: SubnetTick(netuid=0, price=1.0, tao_pool=4.0, alpha_pool=1.0, signals={})
    return [TickData(timestamp=t0, subnets={97: old(0.00265), 0: root()}),
            TickData(timestamp=t0 + 3600, subnets={97: old(40.09), 0: root()}),
            TickData(timestamp=t0 + 7200, subnets={97: old(0.0070), 0: root()})]   # 2.6x: an ordinary surge, kept


def test_rescale_tick_reserves_drops_outliers_by_default():
    ticks = _builder_ticks()
    report = []
    with pytest.warns(UserWarning, match=r"rescale_tick_reserves: dropped 1 of 6"):
        rescale_tick_reserves(ticks, outlier_report=report)
    assert 97 not in ticks[1].subnets and 0 in ticks[1].subnets       # root exempt
    assert report == [(ticks[1].timestamp, 97, 40.09, pytest.approx(OLD_TAO / OLD_ALPHA), "ratio")]
    surge = ticks[2].subnets[97]
    assert surge.tao_pool / surge.alpha_pool == pytest.approx(0.0070)  # kept and rescaled
    assert ticks[0].subnets[0].tao_pool / ticks[0].subnets[0].alpha_pool == pytest.approx(1.0)


def test_rescale_tick_reserves_guard_off_keeps_everything():
    ticks = _builder_ticks()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rescale_tick_reserves(ticks, outlier_max_ratio=None)
    assert ticks[1].subnets[97].tao_pool / ticks[1].subnets[97].alpha_pool == pytest.approx(40.09)
    with pytest.raises(ValueError):
        rescale_tick_reserves(_builder_ticks(), outlier_max_ratio=1.0)
