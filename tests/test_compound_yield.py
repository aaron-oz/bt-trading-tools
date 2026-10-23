"""Compounding yield accrual (2026-10-22).

Simple accrual between folds was 9% to 13% below the chain's compounding of
the same realized rates on 178-day untouched holds (alpha-trading
docs/bagbot_live_alignment_2026_10_21.org). Default now compounds:
alpha x (exp(sum of r x dt) - 1), independent of the tick cadence.
``compound=False`` / ``yield_accrual="path_simple"`` keep the old numbers.
"""

import math
import unittest

from bt_trading_tools import ledger
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, Position, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

DAY = 86400


class StepRateModel:
    """Rate per day that steps at given times; linear in alpha."""

    def __init__(self, steps):
        self.steps = sorted(steps)

    def rate_at(self, t):
        r = 0.0
        for t0, v in self.steps:
            if t >= t0:
                r = v
        return r

    def accrued_yield(self, netuid, alpha_qty, entry_time, now):
        if now <= entry_time:
            return 0.0
        pts = [entry_time] + [t for t, _ in self.steps if entry_time < t < now] + [now]
        return alpha_qty * sum(self.rate_at(a) * (b - a) / DAY for a, b in zip(pts, pts[1:]))


def pos(**kw):
    base = dict(netuid=7, entry_price=0.001, alpha_qty=1000.0, tao_cost=1.0,
                entry_time=0, yield_anchor_time=0, yield_accrued=0.0, yield_accrued_to=0)
    base.update(kw)
    return Position(**base)


class TestLedgerCompounding(unittest.TestCase):
    def test_constant_rate_matches_compound_formula(self):
        m = StepRateModel([(0, 0.0015)])
        for step in (900, 3600, DAY, 7 * DAY):          # cadence must not matter
            p = pos()
            for t in range(step, 178 * DAY + 1, step):
                ledger.accrue_yield(p, t, m.accrued_yield)
            want = 1000.0 * math.expm1(0.0015 * 178)
            self.assertAlmostEqual(ledger.accrued_yield(p, 178 * DAY, m.accrued_yield),
                                   want, places=6, msg=f"step {step}")
            # daily compounding is 0.085% lower here; the chain compounds per
            # epoch (about 20 per day), about 20x closer to continuous.
            daily = 1000.0 * ((1 + 0.0015) ** 178 - 1)
            self.assertLess(abs(want / daily - 1), 1e-3)
            epoch = 1000.0 * ((1 + 0.0015 / 20) ** (178 * 20) - 1)
            self.assertLess(abs(want / epoch - 1), 1e-4)

    def test_simple_mode_reproduces_old_numbers(self):
        m = StepRateModel([(0, 0.0015)])
        p = pos()
        for t in range(3600, 178 * DAY + 1, 3600):
            ledger.accrue_yield(p, t, m.accrued_yield, compound=False)
        self.assertAlmostEqual(p.yield_accrued, 1000.0 * 0.0015 * 178, places=6)

    def test_rate_path_compounds_on_summed_rate(self):
        m = StepRateModel([(0, 0.01), (5 * DAY, 0.002), (9 * DAY, 0.004)])
        p = pos()
        for t in range(3600, 12 * DAY + 1, 3600):
            ledger.accrue_yield(p, t, m.accrued_yield)
        f = 0.01 * 5 + 0.002 * 4 + 0.004 * 3
        self.assertAlmostEqual(p.yield_accrued, 1000.0 * math.expm1(f), places=6)
        # never-accrued position (anchor path, one interval) gives the same total
        q = pos(yield_accrued_to=None)
        self.assertAlmostEqual(ledger.accrued_yield(q, 12 * DAY, m.accrued_yield),
                               1000.0 * math.expm1(f), places=6)

    def test_fold_then_continue_equals_unbroken_compounding(self):
        m = StepRateModel([(0, 0.01)])
        p = pos()
        ledger.accrue_yield(p, 3 * DAY, m.accrued_yield)
        ledger.fold_yield(p, 4 * DAY, m.accrued_yield)
        ledger.accrue_yield(p, 10 * DAY, m.accrued_yield)
        total = p.alpha_qty + ledger.accrued_yield(p, 10 * DAY, m.accrued_yield)
        self.assertAlmostEqual(total, 1000.0 * math.exp(0.10), places=6)

    def test_overrun_removed_by_discounting(self):
        m = StepRateModel([(0, 0.01)])
        positions = {7: pos()}
        ledger.accrue_yield(positions[7], 10 * DAY, m.accrued_yield)
        rec = ledger.close_deregistered(positions, 7, 100.0, 100_000.0, 8 * DAY,
                                        yield_fn=m.accrued_yield)
        self.assertAlmostEqual(rec["alpha_yield_accrued"], 1000.0 * math.expm1(0.08), places=6)


def _tick(t):
    return TickData(timestamp=t, subnets={7: SubnetTick(7, 0.001, 100.0, 100_000.0, generation=0)})


class _BuyHoldSell:
    def __init__(self, sell_at):
        self.i, self.sell_at = 0, sell_at

    def on_tick(self, tick_, positions, capital, pv):
        i, self.i = self.i, self.i + 1
        if i == 0:
            return [Order(netuid=7, side="buy", tao_amount=1.0)]
        if i == self.sell_at:
            return [Order(netuid=7, side="sell")]
        return []


def _engine_run(mode, step=3600, days=30):
    m = StepRateModel([(0, 0.004), (10 * DAY, 0.001)])
    eng = BacktestEngine(capital=10.0, yield_model=m, yield_accrual=mode, swap_fee_rate=0.0,
                         gas_fee_tao=0.0, uses_proxy=False,
                         realism_config=RealismConfig(enabled=False))
    ticks = [_tick(t) for t in range(0, days * DAY + 1, step)]
    res = eng.run(ticks, _BuyHoldSell(sell_at=days * DAY // step))
    sell = [t for t in res.trades if t.get("tao_received")][0]
    return sell["alpha_qty"] - sell["alpha_yield_accrued"], sell["alpha_yield_accrued"]


class TestEngineCompounding(unittest.TestCase):
    def test_engine_default_compounds_and_path_simple_does_not(self):
        f = 0.004 * 10 + 0.001 * 20
        q, y = _engine_run("path")
        self.assertAlmostEqual(y, q * math.expm1(f), places=6)
        q2, y2 = _engine_run("path_simple")
        self.assertAlmostEqual(q2, q, places=9)
        self.assertAlmostEqual(y2, q * f, places=6)
        qa, ya = _engine_run("anchor")                  # old formula, simple
        self.assertAlmostEqual(ya, q * f, places=6)

    def test_engine_cadence_invariant(self):
        _, y_hour = _engine_run("path", step=3600)
        _, y_day = _engine_run("path", step=DAY)
        self.assertAlmostEqual(y_hour, y_day, places=6)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine(yield_accrual="compound")


if __name__ == "__main__":
    unittest.main()
