"""Path-dependent yield accrual (2026-10-10).

Before: yield = alpha x rate(now) x (now - anchor), so the rate at the sale or
mark was applied to the whole hold and a falling rate marked down yield
already earned. Now each tick credits the interval since the previous tick at
that interval's rate (``ledger.accrue_yield``); ``yield_accrual="anchor"``
reproduces the old formula.
"""

import unittest

import pandas as pd

from bt_trading_tools import ledger
from bt_trading_tools.alpha_yield_history import HistoricalYieldModel
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, Position, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

DAY = 86400


class StepRateModel:
    """Duck-typed model with the sale-rate convention of ``AlphaYieldModel``:
    the rate in force at ``now`` times the whole interval. ``steps`` is a list
    of (from_ts, rate_per_day)."""

    def __init__(self, steps):
        self.steps = sorted(steps)

    def rate_at(self, ts):
        r = 0.0
        for t, v in self.steps:
            if ts >= t:
                r = v
        return r

    def accrued_yield(self, netuid, alpha_qty, entry_time, now):
        if now <= entry_time or alpha_qty <= 0:
            return 0.0
        return alpha_qty * self.rate_at(now) * (now - entry_time) / DAY


def engine(model, mode, **kw):
    kw.setdefault("capital", 10.0)
    kw.setdefault("swap_fee_rate", 0.0)
    kw.setdefault("gas_fee_tao", 0.0)
    kw.setdefault("uses_proxy", False)
    kw.setdefault("realism_config", RealismConfig(enabled=False))
    return BacktestEngine(yield_model=model, yield_accrual=mode, **kw)


def tick(t, tao=100.0, alpha=100_000.0, gen=0, netuid=7):
    return TickData(timestamp=t, subnets={
        netuid: SubnetTick(netuid, tao / alpha, tao, alpha, generation=gen)})


class BuyThenHold:
    def __init__(self, sell_at=None):
        self.i, self.sell_at = 0, sell_at

    def on_tick(self, tick_, positions, capital, pv):
        i, self.i = self.i, self.i + 1
        if i == 0:
            return [Order(netuid=7, side="buy", tao_amount=1.0)]
        if self.sell_at is not None and i == self.sell_at:
            return [Order(netuid=7, side="sell")]
        return []


def _run(model, mode, n_days=15, sell_at=None, step=DAY):
    ticks = [tick(t) for t in range(0, n_days * DAY + 1, step)]
    return engine(model, mode).run(ticks, BuyThenHold(sell_at))


class TestLedgerAccrual(unittest.TestCase):
    def pos(self, **kw):
        base = dict(netuid=7, entry_price=0.001, alpha_qty=1000.0, tao_cost=1.0,
                    entry_time=0, yield_anchor_time=0, yield_accrued=0.0,
                    yield_accrued_to=0)
        base.update(kw)
        return Position(**base)

    def test_constant_rate_path_equals_anchor(self):
        m = StepRateModel([(0, 0.01)])
        p = self.pos()
        for t in range(3600, 10 * DAY + 1, 3600):
            ledger.accrue_yield(p, t, m.accrued_yield)
        legacy = m.accrued_yield(7, 1000.0, 0, 10 * DAY)
        self.assertAlmostEqual(ledger.accrued_yield(p, 10 * DAY, m.accrued_yield),
                               legacy, places=9)
        self.assertAlmostEqual(legacy, 100.0, places=9)

    def test_falling_rate_does_not_mark_down_past_accrual(self):
        m = StepRateModel([(0, 0.01), (10 * DAY, 0.001)])
        p = self.pos()
        seen = []
        for t in range(DAY, 15 * DAY + 1, DAY):
            ledger.accrue_yield(p, t, m.accrued_yield)
            seen.append(ledger.accrued_yield(p, t, m.accrued_yield))
        self.assertEqual(seen, sorted(seen))                     # never falls
        # Day 10's interval ends at 10*DAY, where the new rate is in force
        # (rate at the interval end, as the sale-rate convention reads it).
        self.assertAlmostEqual(seen[-1], 1000 * (0.01 * 9 + 0.001 * 6), places=9)
        legacy = m.accrued_yield(7, 1000.0, 0, 15 * DAY)
        self.assertAlmostEqual(legacy, 1000 * 0.001 * 15, places=9)

    def test_never_accrued_position_starts_at_old_formula(self):
        """State-file migration: a position with no accrual fields reports
        the old formula's value at its first accrual (no jump at restart)."""
        m = StepRateModel([(0, 0.01), (5 * DAY, 0.002)])
        p = self.pos(yield_accrued_to=None)
        old_value = m.accrued_yield(7, 1000.0, 0, 8 * DAY)
        credited = ledger.accrue_yield(p, 8 * DAY, m.accrued_yield)
        self.assertAlmostEqual(credited, old_value, places=9)
        self.assertAlmostEqual(p.yield_accrued, old_value, places=9)
        self.assertEqual(p.yield_accrued_to, 8 * DAY)

    def test_fold_resets_accumulator(self):
        m = StepRateModel([(0, 0.01)])
        p = self.pos()
        ledger.accrue_yield(p, 2 * DAY, m.accrued_yield)
        folded = ledger.fold_yield(p, 3 * DAY, m.accrued_yield)
        self.assertAlmostEqual(folded, 30.0, places=9)
        self.assertAlmostEqual(p.alpha_qty, 1030.0, places=9)
        self.assertEqual((p.yield_accrued, p.yield_accrued_to, p.yield_anchor_time),
                         (0.0, 3 * DAY, 3 * DAY))

    def test_dereg_close_before_accrued_to_removes_overrun(self):
        m = StepRateModel([(0, 0.01)])
        positions = {7: self.pos()}
        ledger.accrue_yield(positions[7], 10 * DAY, m.accrued_yield)
        rec = ledger.close_deregistered(positions, 7, 100.0, 100_000.0, 8 * DAY,
                                        yield_fn=m.accrued_yield)
        self.assertAlmostEqual(rec["alpha_yield_accrued"], 80.0, places=9)

    def test_rebirth_cutoff_holds_for_short_intervals(self):
        """HistoricalYieldModel with reregistrations: intervals after the
        rebirth earn nothing, although each interval starts after it."""
        days = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
        table = pd.DataFrame({"netuid": 7, "date": days, "rate_per_day": 0.01})
        t0 = int(days[0].timestamp())
        rebirth = t0 + 10 * DAY + 3600
        m = HistoricalYieldModel(table, reregistrations={7: [rebirth]})
        p = self.pos(entry_time=t0, yield_anchor_time=t0, yield_accrued_to=t0)

        def fn(n, q, a, b, origin=None):
            return m.accrued_yield(n, q, a, b, generation_origin=origin)
        for t in range(t0 + 3600, t0 + 15 * DAY + 1, 3600):
            ledger.accrue_yield(p, t, fn)
        legacy = m.accrued_yield(7, 1000.0, t0, t0 + 15 * DAY)
        self.assertAlmostEqual(p.yield_accrued, legacy, places=6)
        self.assertLess(p.yield_accrued, 1000 * 0.01 * 10.1)


class TestEngineModes(unittest.TestCase):
    def test_constant_rate_engine_path_equals_anchor(self):
        m = StepRateModel([(0, 0.005)])
        a = _run(m, "anchor", sell_at=12)
        b = _run(m, "path", sell_at=12)
        sa = [t for t in a.trades if t.get("tao_received")]
        sb = [t for t in b.trades if t.get("tao_received")]
        self.assertAlmostEqual(sa[0]["alpha_yield_accrued"], sb[0]["alpha_yield_accrued"], places=9)
        for x, y in zip(a.equity_curve, b.equity_curve):
            self.assertAlmostEqual(x["total_equity"], y["total_equity"], places=6)

    def test_falling_rate_engine(self):
        m = StepRateModel([(0, 0.01), (10 * DAY, 0.001)])
        a = _run(m, "anchor", sell_at=15)
        b = _run(m, "path", sell_at=15)
        sa = [t for t in a.trades if t.get("tao_received")][0]
        sb = [t for t in b.trades if t.get("tao_received")][0]
        ya, yb = sa["alpha_yield_accrued"], sb["alpha_yield_accrued"]
        q = sb["alpha_qty"] - yb                              # alpha bought
        self.assertAlmostEqual(sa["alpha_qty"] - ya, q, places=6)
        self.assertAlmostEqual(ya, q * 0.001 * 15, places=6)
        self.assertAlmostEqual(yb, q * (0.01 * 9 + 0.001 * 6), places=6)
        # Path-mode equity never drops from the rate cut at day 10.
        eq = [p["total_equity"] for p in b.equity_curve]
        self.assertGreaterEqual(eq[10], eq[9])
        eq_a = [p["total_equity"] for p in a.equity_curve]
        self.assertLess(eq_a[10], eq_a[9])                    # the old revaluation

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            engine(StepRateModel([]), "sale")


if __name__ == "__main__":
    unittest.main()
