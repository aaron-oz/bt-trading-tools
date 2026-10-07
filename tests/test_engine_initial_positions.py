"""BacktestEngine.run(initial_positions=...): start from a book that is not flat."""

import unittest

from bt_trading_tools.alpha_yield import AlphaYieldModel, ZeroYieldProvider
from bt_trading_tools.amm import amm_sell
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, Position, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

DAY = 86400


def engine(**kw):
    kw.setdefault("capital", 10.0)
    kw.setdefault("swap_fee_rate", 0.0)
    kw.setdefault("gas_fee_tao", 0.0)
    kw.setdefault("uses_proxy", False)
    kw.setdefault("realism_config", RealismConfig(enabled=False))
    kw.setdefault("yield_model", AlphaYieldModel(ZeroYieldProvider()))
    return BacktestEngine(**kw)


def tick(t, tao, alpha, gen=0, netuid=7):
    return TickData(timestamp=t, subnets={
        netuid: SubnetTick(netuid, tao / alpha, tao, alpha, generation=gen)})


class Script:
    def __init__(self, plan=None):
        self.plan, self.i, self.seen = plan or {}, 0, []

    def on_tick(self, tick_, positions, capital, pv):
        self.seen.append((dict(positions), capital, pv))
        orders = self.plan.get(self.i, [])
        self.i += 1
        return orders


def position(**kw):
    base = dict(netuid=7, entry_price=0.001, alpha_qty=1000.0, tao_cost=1.0,
                entry_time=-5 * DAY, generation=0)
    base.update(kw)
    return Position(**base)


class TestInitialPositions(unittest.TestCase):
    def test_default_is_flat_and_unchanged(self):
        strat = Script()
        r = engine().run([tick(0, 100.0, 100_000.0), tick(DAY, 100.0, 100_000.0)], strat)
        self.assertEqual(strat.seen[0][0], {})
        self.assertEqual(r.positions_at_end, {})

    def test_starting_equity_is_cash_plus_liquidation_value(self):
        strat = Script()
        ticks = [tick(0, 100.0, 100_000.0), tick(DAY, 100.0, 100_000.0)]
        r = engine().run(ticks, strat, initial_positions={7: position()})
        tao_out, _, _ = amm_sell(1000.0, 100.0, 100_000.0)
        self.assertAlmostEqual(r.equity_curve[0]["total_equity"], 10.0 + tao_out, places=5)
        self.assertEqual(list(strat.seen[0][0]), [7])            # the strategy sees it
        self.assertAlmostEqual(strat.seen[0][1], 10.0)           # cost is not deducted from cash

    def test_the_injected_position_can_be_sold_and_is_closed_at_the_end_otherwise(self):
        ticks = [tick(0, 100.0, 100_000.0), tick(DAY, 100.0, 100_000.0), tick(2 * DAY, 100.0, 100_000.0)]
        r = engine().run(ticks, Script({1: [Order(7, "sell")]}), initial_positions={7: position()})
        sells = [t for t in r.trades if t.get("exit_time") is not None]
        self.assertEqual([t["reason"] for t in sells], [""])
        self.assertEqual(r.positions_at_end, {})
        r2 = engine().run(ticks, Script(), initial_positions={7: position()})
        self.assertEqual([t["reason"] for t in r2.trades if t.get("exit_time") is not None],
                         ["end_of_data"])

    def test_inputs_are_copied_not_mutated(self):
        p = position()
        engine().run([tick(0, 100.0, 100_000.0), tick(DAY, 100.0, 100_000.0)],
                     Script({0: [Order(7, "sell")]}), initial_positions={7: p})
        self.assertEqual((p.alpha_qty, p.tao_cost), (1000.0, 1.0))

    def test_rebirth_guard_protects_an_injected_position(self):
        # The old subnet (generation 0) is replaced by generation 1 at 10x the price.
        ticks = [tick(0, 100.0, 100_000.0, 0), tick(DAY, 1000.0, 100_000.0, 1),
                 tick(2 * DAY, 1000.0, 100_000.0, 1)]
        r = engine().run(ticks, Script(), initial_positions={7: position(generation=0)})
        sells = [t for t in r.trades if t.get("exit_time") is not None]
        self.assertEqual([t["reason"] for t in sells], ["dereg_refund"])
        self.assertAlmostEqual(sells[0]["tao_received"], 1000.0 * 100.0 / 100_000.0, places=6)


if __name__ == "__main__":
    unittest.main()
