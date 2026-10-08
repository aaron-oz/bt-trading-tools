"""Order.decision_tao_pool / decision_alpha_pool: decide on one pool, fill on another.

Paper bots that size an order on a stale data-feed pool and execute against the
live pool fail the order when the live fill is worse than the decision fill by
more than the rate tolerance (PaperBotBase.adjust_action_with_live_pool, then
RealismSimulator.check_rate_tolerance). These tests pin the engine's mirror of
that path.
"""

import unittest

from bt_trading_tools.alpha_yield import AlphaYieldModel, ZeroYieldProvider
from bt_trading_tools.backtest import BacktestEngine
from bt_trading_tools.backtest.types import Order, SubnetTick, TickData
from bt_trading_tools.execution import RealismConfig

DAY = 86400


def engine(realism=None, **kw):
    kw.setdefault("capital", 10.0)
    kw.setdefault("yield_model", AlphaYieldModel(ZeroYieldProvider()))
    # Realism on, but only the layers under test: no random rejects.
    rc = realism or RealismConfig(buy_failure_rate=0.0, sell_failure_rate=0.0)
    return BacktestEngine(realism_config=rc, **kw)


def tick(t, tao, alpha, netuid=7):
    return TickData(timestamp=t, subnets={netuid: SubnetTick(netuid, tao / alpha, tao, alpha, generation=0)})


class Once:
    """Places the orders on the first tick and records the book it sees after."""

    def __init__(self, orders):
        self.orders, self.done, self.book = orders, False, {}

    def on_tick(self, tick_, positions, capital, pv):
        if self.done:
            self.book = dict(positions)
            return []
        self.done = True
        return self.orders


def held(strategy_runs):
    """The book after the first tick (the end-of-data sweep empties positions_at_end)."""
    return strategy_runs.book


class TestHybridDecisionPool(unittest.TestCase):
    def test_default_order_unchanged(self):
        o = Order(netuid=7, side="buy", tao_amount=1.0)
        self.assertIsNone(o.decision_tao_pool)
        st = Once([o]); r = engine().run([tick(0, 1000.0, 1_000_000.0), tick(DAY, 1000.0, 1_000_000.0)], st)
        self.assertIn(7, held(st))
        self.assertFalse([t for t in r.trades if t.get("status") == "failed"])

    def test_buy_fails_rate_tolerance_when_live_pool_is_much_worse(self):
        # Decision pool says price 0.001; the fill pool is 10% more expensive.
        # Base AMM impact of 1 TAO on 1000 TAO is ~0.1%, tolerance 2 pp: fails.
        o = Order(netuid=7, side="buy", tao_amount=1.0,
                  decision_tao_pool=1000.0, decision_alpha_pool=1_000_000.0)
        st = Once([o]); r = engine().run([tick(0, 1100.0, 1_000_000.0 / 1.1), tick(DAY, 1100.0, 1_000_000.0 / 1.1)], st)
        failed = [t for t in r.trades if t.get("status") == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["failure_reason"], "rate_tolerance")
        self.assertNotIn(7, held(st))

    def test_buy_fills_when_live_pool_is_close(self):
        # 0.5% worse than decided: within base + 2 pp.
        o = Order(netuid=7, side="buy", tao_amount=1.0,
                  decision_tao_pool=1000.0, decision_alpha_pool=1_000_000.0)
        st = Once([o]); engine().run([tick(0, 1002.5, 1_000_000.0 / 1.0025), tick(DAY, 1002.5, 1_000_000.0 / 1.0025)], st)
        self.assertIn(7, held(st))

    def test_buy_fills_when_live_pool_is_better(self):
        o = Order(netuid=7, side="buy", tao_amount=1.0,
                  decision_tao_pool=1000.0, decision_alpha_pool=1_000_000.0)
        st = Once([o]); engine().run([tick(0, 900.0, 1_000_000.0 / 0.9), tick(DAY, 900.0, 1_000_000.0 / 0.9)], st)
        self.assertIn(7, held(st))

    def test_hybrid_fill_skips_csv_slippage_noise(self):
        # With a decision pool equal to the fill pool, the alpha received must be
        # the exact AMM fill (no Gaussian noise), whatever the seed.
        got = set()
        for seed in range(5):
            o = Order(netuid=7, side="buy", tao_amount=1.0,
                      decision_tao_pool=1000.0, decision_alpha_pool=1_000_000.0)
            e = BacktestEngine(
                capital=10.0, yield_model=AlphaYieldModel(ZeroYieldProvider()),
                realism_config=RealismConfig(buy_failure_rate=0.0), realism_rng_seed=seed)
            st = Once([o]); e.run([tick(0, 1000.0, 1_000_000.0), tick(DAY, 1000.0, 1_000_000.0)], st)
            got.add(round(held(st)[7].alpha_qty, 9))
        self.assertEqual(len(got), 1)

    def test_plain_buy_still_gets_noise(self):
        got = set()
        for seed in range(5):
            o = Order(netuid=7, side="buy", tao_amount=1.0)
            e = BacktestEngine(capital=10.0, yield_model=AlphaYieldModel(ZeroYieldProvider()),
                               realism_config=RealismConfig(buy_failure_rate=0.0), realism_rng_seed=seed)
            st = Once([o]); e.run([tick(0, 1000.0, 1_000_000.0), tick(DAY, 1000.0, 1_000_000.0)], st)
            got.add(round(held(st)[7].alpha_qty, 9))
        self.assertGreater(len(got), 1)


if __name__ == "__main__":
    unittest.main()
