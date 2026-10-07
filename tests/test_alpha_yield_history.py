"""Point-in-time alpha yield rates from the validator yield history."""

import unittest

import pandas as pd

from bt_trading_tools.alpha_yield_history import (
    DAY_S, HistoricalYieldModel, build_rate_table)
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

D0 = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp())     # midnight UTC


def row(date, netuid, hk, stake, apy, uptime=1.0, ts=None):
    return {"date": date, "netuid": netuid, "hotkey": hk, "stake_rao": stake,
            "thirty_day_apy": apy, "thirty_day_epoch_participation": uptime,
            "timestamp": ts or f"{date}T12:00:00Z"}


def frame(rows):
    return pd.DataFrame(rows)


class TestRateTable(unittest.TestCase):
    def test_stake_weighted_mean_over_passing_validators_divided_by_365(self):
        h = frame([
            row("2026-07-01", 1, "A", 100, 0.50),
            row("2026-07-01", 1, "B", 300, 0.10),
            row("2026-07-01", 1, "C", 1, 9.00),                  # stake share < 2%: excluded
            row("2026-07-01", 1, "D", 100, 8.00, uptime=0.5),    # uptime < 0.95: excluded
        ])
        t = build_rate_table(h)
        self.assertEqual(len(t), 1)
        # shares are against the TOTAL recorded stake, so C (1/501) is below 2%
        self.assertAlmostEqual(t.iloc[0]["rate_per_day"], (100 * 0.50 + 300 * 0.10) / 400 / 365.0, places=12)

    def test_top_k_keeps_the_highest_apy(self):
        h = frame([row("2026-07-01", 1, "A", 100, 0.50), row("2026-07-01", 1, "B", 100, 0.20),
                   row("2026-07-01", 1, "C", 100, 0.10)])
        self.assertAlmostEqual(build_rate_table(h, top_k=1).iloc[0]["rate_per_day"], 0.50 / 365.0, places=12)
        self.assertAlmostEqual(build_rate_table(h, top_k=2).iloc[0]["rate_per_day"], 0.35 / 365.0, places=12)

    def test_last_snapshot_of_the_day_is_used(self):
        h = frame([row("2026-07-01", 1, "A", 100, 0.10, ts="2026-07-01T01:00:00Z"),
                   row("2026-07-01", 1, "A", 100, 0.30, ts="2026-07-01T20:00:00Z")])
        self.assertAlmostEqual(build_rate_table(h).iloc[0]["rate_per_day"], 0.30 / 365.0, places=12)

    def test_rows_with_missing_apy_or_nonpositive_stake_are_dropped(self):
        h = frame([row("2026-07-01", 1, "A", 100, 0.20), row("2026-07-01", 1, "B", 100, float("nan")),
                   row("2026-07-01", 1, "C", 0, 0.90)])
        self.assertAlmostEqual(build_rate_table(h).iloc[0]["rate_per_day"], 0.20 / 365.0, places=12)


def model(rates_by_day, **kw):
    """rates_by_day: {day_offset: rate_per_day} for netuid 1."""
    t = pd.DataFrame([{"netuid": 1, "date": pd.Timestamp(D0 + d * DAY_S, unit="s", tz="UTC"),
                       "rate_per_day": r} for d, r in rates_by_day.items()])
    return HistoricalYieldModel(t, **kw)


class TestModel(unittest.TestCase):
    def test_as_of_lookup_never_looks_ahead(self):
        m = model({0: 0.001, 2: 0.003})
        self.assertEqual(m.rate_on(1, D0 - 1), 0.0)                      # before any history
        self.assertEqual(m.rate_on(1, D0 + 0.5 * DAY_S), 0.001)
        self.assertEqual(m.rate_on(1, D0 + 1.9 * DAY_S), 0.001)          # day 1 has none: day 0's
        self.assertEqual(m.rate_on(1, D0 + 2.1 * DAY_S), 0.003)
        self.assertEqual(m.rate_on(99, D0), 0.0)                         # unknown subnet

    def test_lag_uses_an_earlier_days_rate(self):
        m = model({0: 0.001, 1: 0.002}, lag_days=1)
        self.assertEqual(m.rate_on(1, D0 + 1.5 * DAY_S), 0.001)

    def test_sale_rate_applies_the_rate_at_the_sale_over_the_whole_hold(self):
        m = model({0: 0.001, 1: 0.003})
        got = m.accrued_yield(1, 1000.0, D0 + 0.5 * DAY_S, D0 + 1.5 * DAY_S)
        self.assertAlmostEqual(got, 1000.0 * 0.003 * 1.0, places=9)

    def test_integrated_sums_each_days_own_rate(self):
        m = model({0: 0.001, 1: 0.003}, convention="integrated")
        got = m.accrued_yield(1, 1000.0, D0 + 0.5 * DAY_S, D0 + 1.5 * DAY_S)
        self.assertAlmostEqual(got, 1000.0 * (0.001 * 0.5 + 0.003 * 0.5), places=9)

    def test_implausible_rate_is_zero_and_nonpositive_inputs_are_zero(self):
        m = model({0: 0.5})                                              # 50%/day: a data error
        self.assertEqual(m.accrued_yield(1, 1000.0, D0, D0 + DAY_S), 0.0)
        m2 = model({0: 0.001})
        self.assertEqual(m2.accrued_yield(1, 0.0, D0, D0 + DAY_S), 0.0)
        self.assertEqual(m2.accrued_yield(1, 1000.0, D0 + DAY_S, D0), 0.0)

    def test_bad_convention_raises(self):
        with self.assertRaises(ValueError):
            model({0: 0.001}, convention="average")


class TestEngineIntegration(unittest.TestCase):
    def test_engine_credits_the_historical_rate_on_a_sale(self):
        rate = 0.002
        m = model({0: rate, 1: rate, 2: rate})
        ticks = [TickData(timestamp=D0 + i * DAY_S, subnets={
            1: SubnetTick(1, 0.001, 1000.0, 1_000_000.0, generation=0)}) for i in range(3)]

        class Script:
            def __init__(self):
                self.i = 0

            def on_tick(self, tick, positions, capital, pv):
                self.i += 1
                if self.i == 1:
                    return [Order(1, "buy", tao_amount=5.0)]
                if self.i == 3:
                    return [Order(1, "sell")]
                return []

        e = BacktestEngine(capital=10.0, swap_fee_rate=0.0, gas_fee_tao=0.0, uses_proxy=False,
                           realism_config=RealismConfig(enabled=False), yield_model=m)
        r = e.run(ticks, Script())
        sell = [t for t in r.trades if t.get("exit_time") is not None][0]
        alpha_bought = sell["alpha_qty"] - sell["alpha_yield_accrued"]
        self.assertAlmostEqual(sell["alpha_yield_accrued"], alpha_bought * rate * 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
