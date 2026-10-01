"""BacktestEngine behavior after the 2026-10-01 switch to the shared ledger."""

import unittest
import warnings

from bt_trading_tools import ledger
from bt_trading_tools.alpha_yield import AlphaYieldModel, ZeroYieldProvider
from bt_trading_tools.amm import amm_buy
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

DAY = 86400


def engine(**kw):
    kw.setdefault("capital", 100.0)
    kw.setdefault("swap_fee_rate", 0.0)
    kw.setdefault("gas_fee_tao", 0.0)
    kw.setdefault("uses_proxy", False)
    kw.setdefault("realism_config", RealismConfig(enabled=False))
    kw.setdefault("yield_model", AlphaYieldModel(ZeroYieldProvider()))
    return BacktestEngine(**kw)


class Script:
    """Strategy that emits a fixed order list per tick index."""

    def __init__(self, plan):
        self.plan, self.i = plan, 0

    def on_tick(self, tick, positions, capital, pv):
        orders = self.plan.get(self.i, [])
        self.i += 1
        return orders


def tick(t, **subnets):
    return TickData(timestamp=t, subnets={
        int(k[1:]): SubnetTick(int(k[1:]), tp / ap, tp, ap, generation=g)
        for k, (tp, ap, g) in subnets.items()})


class TestRebirth(unittest.TestCase):
    def test_position_closed_at_refund_not_sold_into_new_pool(self):
        # Old subnet: 100 TAO / 100k alpha (spot 0.001). New subnet in the
        # same slot: 1000 TAO / 100k alpha (spot 0.01), a 10x price.
        ticks = [
            tick(0, n7=(100.0, 100_000.0, 0)),
            tick(DAY, n7=(100.0, 100_000.0, 0)),
            tick(2 * DAY, n7=(1000.0, 100_000.0, 1)),
            tick(3 * DAY, n7=(1000.0, 100_000.0, 1)),
        ]
        strat = Script({0: [Order(7, "buy", tao_amount=1.0)],
                        3: [Order(7, "sell")]})
        r = engine().run(ticks, strat)
        sells = [t for t in r.trades if t.get("exit_time") is not None]
        self.assertEqual([t["reason"] for t in sells], ["dereg_refund"])
        alpha, _, _ = amm_buy(1.0, 100.0, 100_000.0)
        refund = alpha * 100.0 / 100_000.0     # spot at the old subnet
        self.assertAlmostEqual(sells[0]["tao_received"], refund, places=9)
        self.assertLess(sells[0]["tao_received"], 1.0)
        self.assertTrue(r.identity_guard_active)

    def test_without_generations_engine_warns(self):
        ticks = [TickData(0, {1: SubnetTick(1, 0.01, 100.0, 10_000.0)})]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = engine().run(ticks, Script({}))
        self.assertTrue(any("generation" in str(x.message) for x in w))
        self.assertFalse(r.identity_guard_active)


class TestNoDefaultCap(unittest.TestCase):
    def test_buy_not_truncated_by_default(self):
        # 20 TAO into a 100-TAO pool: 20% of depth. Old default capped at 5.
        ticks = [tick(0, n3=(100.0, 10_000.0, 0)), tick(DAY, n3=(100.0, 10_000.0, 0))]
        r = engine().run(ticks, Script({0: [Order(3, "buy", tao_amount=20.0)]}))
        self.assertAlmostEqual(r.equity_curve[0]["capital"], 80.0)

    def test_explicit_cap_still_applies(self):
        ticks = [tick(0, n3=(100.0, 10_000.0, 0)), tick(DAY, n3=(100.0, 10_000.0, 0))]
        r = engine(max_pool_pct=0.05).run(
            ticks, Script({0: [Order(3, "buy", tao_amount=20.0)]}))
        self.assertAlmostEqual(r.equity_curve[0]["capital"], 95.0)

    def test_capped_sell_keeps_remainder(self):
        # The pool shrinks after entry (the case the old engine got wrong):
        # a 1% cap on 2,000 pool alpha allows 20 alpha per sell, far less than
        # the ~99 held. The old engine booked the full cost and deleted the
        # position; the remainder must now stay open and be closed later.
        ticks = [tick(0, n3=(100.0, 10_000.0, 0)),
                 tick(DAY, n3=(20.0, 2_000.0, 0)),
                 tick(2 * DAY, n3=(20.0, 2_000.0, 0))]
        strat = Script({0: [Order(3, "buy", tao_amount=1.0)],
                        1: [Order(3, "sell", reason="exit")]})
        r = engine(max_pool_pct=0.01).run(ticks, strat)
        sells = [t for t in r.trades if t.get("exit_time") is not None]
        self.assertEqual(sells[0]["reason"], "exit")
        self.assertAlmostEqual(sells[0]["alpha_qty"], 20.0)
        self.assertFalse(sells[0]["position_closed"])
        self.assertGreater(sells[0]["alpha_remaining"], 70.0)
        self.assertEqual(sells[-1]["reason"], "end_of_data")
        total_cost = sum(t["tao_cost"] for t in sells)
        self.assertAlmostEqual(total_cost, 1.0, places=9)


class TestValuation(unittest.TestCase):
    def test_mark_is_liquidation_value(self):
        ticks = [tick(0, n2=(100.0, 10_000.0, 0)), tick(DAY, n2=(100.0, 10_000.0, 0))]
        r = engine().run(ticks, Script({0: [Order(2, "buy", tao_amount=10.0)]}))
        # Valued against the TICK's pool state: backtest ticks (like paper
        # snapshots) do not include the bot's own trades in pool state.
        alpha, _, _ = amm_buy(10.0, 100.0, 10_000.0)
        expected = 90.0 + ledger.liquidation_value(alpha, 100.0, 10_000.0, 0.0)
        self.assertAlmostEqual(r.equity_curve[0]["total_equity"], expected, places=5)  # curve rounds to 6 dp
        self.assertLess(r.equity_curve[0]["total_equity"], 100.0)

    def test_absent_subnet_marked_at_last_seen_not_cost(self):
        ticks = [tick(0, n2=(100.0, 10_000.0, 0), n9=(50.0, 50.0, 0)),
                 tick(DAY, n9=(50.0, 50.0, 0))]
        r = engine().run(ticks, Script({0: [Order(2, "buy", tao_amount=10.0)]}))
        self.assertAlmostEqual(r.equity_curve[1]["total_equity"],
                               r.equity_curve[0]["total_equity"], places=9)
        self.assertLess(r.equity_curve[1]["total_equity"], 100.0)


if __name__ == "__main__":
    unittest.main()
