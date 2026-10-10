"""Plausibility rule and generation awareness for the duck-typed yield models.

``AlphaYieldModel.rate`` rejects provider rates that are negative, non-finite
or above MAX_PLAUSIBLE_RATE_PER_DAY. Before 2026-10-08 the duck-typed models
the engine also accepts (``HistoricalSubnetYieldModel``, ``HistoricalYieldModel``)
either did not apply that rule or applied it silently, and both were keyed by
bare netuid, so accrual could run across a netuid rebirth.
"""

import math
import warnings

import pandas as pd
import pytest

from bt_trading_tools.alpha_yield import (
    MAX_PLAUSIBLE_RATE_PER_DAY,
    HistoricalSubnetYieldModel,
    generation_end,
    implausible_rate,
    normalize_reregistrations,
)
from bt_trading_tools.alpha_yield_history import DAY_S, HistoricalYieldModel

D0 = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())


def _csv(tmp_path, rows):
    p = tmp_path / "apy.csv"
    pd.DataFrame(rows, columns=["netuid", "date", "gross_apy"]).to_csv(p, index=False)
    return p


def _days(n):
    return [str((pd.Timestamp("2026-07-01") + pd.Timedelta(days=i)).date()) for i in range(n)]


def test_implausible_rate_rule_matches_provider_rule():
    assert implausible_rate(-1e-9)
    assert implausible_rate(float("nan"))
    assert implausible_rate(float("inf"))
    assert implausible_rate(MAX_PLAUSIBLE_RATE_PER_DAY * 1.0001)
    assert not implausible_rate(MAX_PLAUSIBLE_RATE_PER_DAY)
    assert not implausible_rate(0.0)
    assert not implausible_rate(0.003)


def test_subnet_model_rejects_high_rate_day_with_warning(tmp_path):
    d = _days(5)
    # 0.365 APY -> 0.001/day; day 2 is 58.5 APY (about 0.16/day), as netuid 49
    # on 2025-11-21 in one research CSV.
    rows = [(49, d[i], 0.365) for i in range(5)]
    rows[2] = (49, d[2], 58.5)
    with pytest.warns(UserWarning, match="plausibility ceiling"):
        m = HistoricalSubnetYieldModel(_csv(tmp_path, rows))
    assert m.rejected_rate_rows == 1
    assert m.rejected_examples[0][0] == 49
    # held from just after day 0's row to the end of day 4: rows 1..4 count,
    # row 2 is rejected to 0, so 3 days at 0.001.
    got = m.accrued_yield(49, 100.0, D0 + 3600, D0 + 4 * DAY_S + 3600)
    assert got == pytest.approx(100.0 * 0.001 * 3)


def test_subnet_model_negative_and_inf_rows(tmp_path):
    d = _days(3)
    rows = [(1, d[0], 0.365), (1, d[1], -5.0), (1, d[2], float("inf"))]
    with pytest.warns(UserWarning):
        m = HistoricalSubnetYieldModel(_csv(tmp_path, rows))
    # negative rows were already dropped before; only inf is newly counted
    assert m.rejected_rate_rows == 1
    assert m.accrued_yield(1, 10.0, D0 - 1, D0 + 3 * DAY_S) == pytest.approx(10.0 * 0.001)
    assert math.isfinite(m.accrued_yield(1, 10.0, D0 - 1, D0 + 3 * DAY_S))


def test_subnet_model_clean_csv_no_warning(tmp_path):
    d = _days(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = HistoricalSubnetYieldModel(_csv(tmp_path, [(1, x, 0.365) for x in d]))
    assert m.rejected_rate_rows == 0


def test_subnet_model_stops_at_rebirth(tmp_path):
    d = _days(10)
    rows = [(7, x, 0.365) for x in d]
    rebirth = pd.Timestamp("2026-07-05 08:41", tz="UTC")
    m_plain = HistoricalSubnetYieldModel(_csv(tmp_path, rows))
    m_gen = HistoricalSubnetYieldModel(_csv(tmp_path, rows), reregistrations={7: [rebirth]})
    entry, now = D0 + 3600, D0 + 9 * DAY_S + 3600
    # bare netuid: rows dated 07-02..07-10 -> 9 days
    assert m_plain.accrued_yield(7, 100.0, entry, now) == pytest.approx(100 * 0.001 * 9)
    # generation-aware: rows 07-02..07-04 only (the rebirth day's row is excluded)
    assert m_gen.accrued_yield(7, 100.0, entry, now) == pytest.approx(100 * 0.001 * 3)
    # a position opened after the rebirth accrues normally
    e2 = D0 + 5 * DAY_S + 3600
    assert m_gen.accrued_yield(7, 100.0, e2, now) == pytest.approx(m_plain.accrued_yield(7, 100.0, e2, now))
    # other netuids untouched
    assert m_gen.accrued_yield(8, 100.0, entry, now) == 0.0


def _table(rates, netuid=1):
    return pd.DataFrame({
        "netuid": [netuid] * len(rates),
        "date": pd.to_datetime([D0 + i * DAY_S for i in range(len(rates))], unit="s", utc=True),
        "rate_per_day": rates,
    })


def test_history_model_counts_rejected_rows_with_warning():
    with pytest.warns(UserWarning, match="HistoricalYieldModel"):
        m = HistoricalYieldModel(_table([0.001, 242.0, 0.001, float("nan")]))
    assert m.rejected_rate_rows == 2
    assert m.rate_on(1, D0 + DAY_S + 10) == 0.0


def test_history_model_stops_at_rebirth_both_conventions():
    rates = [0.001] * 5 + [0.015] * 5     # the new subnet has a higher rate
    reregs = {1: [pd.Timestamp(D0 + 5 * DAY_S + 3600, unit="s", tz="UTC")]}
    entry, now = D0 + 3600, D0 + 8 * DAY_S
    for conv in ("sale_rate", "integrated"):
        plain = HistoricalYieldModel(_table(rates), convention=conv)
        gen = HistoricalYieldModel(_table(rates), convention=conv, reregistrations=reregs)
        got = gen.accrued_yield(1, 100.0, entry, now)
        # accrual stops at the rebirth (5 days exactly) at the old rate
        assert got == pytest.approx(100.0 * 0.001 * 5.0), conv
        assert plain.accrued_yield(1, 100.0, entry, now) > got


def test_reregistration_helpers():
    r = normalize_reregistrations({3: [pd.Timestamp("2026-07-02", tz="UTC"), 100.0]})
    assert r == {3: [100.0, float(pd.Timestamp("2026-07-02", tz="UTC").timestamp())]}
    assert normalize_reregistrations(None) == {}
    assert generation_end(r, 3, 50.0, 200.0) == 100.0
    assert generation_end(r, 3, 100.0, 200.0) == 200.0      # entered at the rebirth: new generation
    assert generation_end(r, 4, 50.0, 200.0) == 200.0


def test_engine_uses_clamped_subnet_model(tmp_path):
    """End to end: the engine credits zero for a rejected day, not 0.16/day."""
    from bt_trading_tools.backtest import BacktestEngine
    from bt_trading_tools.backtest.types import Order, SubnetTick, TickData
    from bt_trading_tools.execution import RealismConfig

    d = _days(4)
    rows = [(5, x, 0.365) for x in d]
    rows[2] = (5, d[2], 58.5)
    with pytest.warns(UserWarning):
        model = HistoricalSubnetYieldModel(_csv(tmp_path, rows))

    ticks = [TickData(timestamp=D0 + i * DAY_S + 3600, subnets={
        5: SubnetTick(5, 0.001, 1000.0, 1_000_000.0, generation=0)}) for i in range(4)]

    class Script:
        def __init__(self):
            self.i = 0

        def on_tick(self, tick, positions, capital, pv):
            self.i += 1
            if self.i == 1:
                return [Order(5, "buy", tao_amount=5.0)]
            if self.i == 4:
                return [Order(5, "sell")]
            return []

    e = BacktestEngine(capital=10.0, swap_fee_rate=0.0, gas_fee_tao=0.0, uses_proxy=False,
                       realism_config=RealismConfig(enabled=False), yield_model=model)
    r = e.run(ticks, Script())
    sell = [t for t in r.trades if t.get("exit_time") is not None][0]
    alpha_bought = sell["alpha_qty"] - sell["alpha_yield_accrued"]
    # rows dated days 1..3 accrue; day 2 is rejected -> 2 days at 0.001/day
    assert sell["alpha_yield_accrued"] == pytest.approx(alpha_bought * 0.001 * 2.0, rel=1e-6)
