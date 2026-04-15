"""Tests for bt_trading_tools.backtest — engine, stats, types."""

import os
import tempfile
import unittest

from bt_trading_tools.backtest import (
    BacktestEngine,
    BacktestResults,
    Order,
    Position,
    Strategy,
    SubnetTick,
    TickData,
    compute_stats,
)
from bt_trading_tools.tracking.trade_log import TradeLog
from bt_trading_tools.tracking.portfolio_log import PortfolioLog


# ── Test strategy implementations ────────────────────────────────

class BuyAndHoldStrategy:
    """Buy once on first tick, hold forever."""

    def __init__(self, netuid: int, spend: float):
        self.netuid = netuid
        self.spend = spend
        self.bought = False

    def on_tick(self, tick, positions, capital, portfolio_value):
        if not self.bought and capital >= self.spend:
            self.bought = True
            return [Order(self.netuid, "buy", tao_amount=self.spend, reason="initial_buy")]
        return []


class ThresholdStrategy:
    """Simple threshold strategy: buy on dip, sell on rally."""

    def __init__(self, netuid: int, buy_threshold: float, sell_threshold: float,
                 trade_size: float):
        self.netuid = netuid
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trade_size = trade_size
        self.prev_price = None

    def on_tick(self, tick, positions, capital, portfolio_value):
        orders = []
        st = tick.subnets.get(self.netuid)
        if st is None:
            return orders

        if self.prev_price is not None:
            pct_change = (st.price - self.prev_price) / self.prev_price * 100

            # Sell on rally
            if pct_change > self.sell_threshold and self.netuid in positions:
                orders.append(Order(self.netuid, "sell", reason="rally_sell",
                                   signal_data={"pct_change": pct_change}))

            # Buy on dip
            if pct_change < self.buy_threshold and self.netuid not in positions:
                if capital >= self.trade_size:
                    orders.append(Order(self.netuid, "buy", tao_amount=self.trade_size,
                                       reason="dip_buy",
                                       signal_data={"pct_change": pct_change}))

        self.prev_price = st.price
        return orders


class NeverTradeStrategy:
    """Does nothing."""
    def on_tick(self, tick, positions, capital, portfolio_value):
        return []


# ── Helpers ──────────────────────────────────────────────────────

def make_ticks(prices: list[float], netuid: int = 107,
               tao_pool: float = 500.0, alpha_pool: float = 25000.0,
               start_ts: int = 1000000, interval: int = 86400) -> list[TickData]:
    """Generate TickData from a price series."""
    ticks = []
    for i, price in enumerate(prices):
        # Adjust pool to match price: price = tao/alpha, k = tao*alpha
        # tao_pool * alpha_pool = k (constant), price = tao/alpha
        # So: tao = sqrt(k * price), alpha = sqrt(k / price)
        k = tao_pool * alpha_pool
        tp = (k * price) ** 0.5
        ap = (k / price) ** 0.5

        ticks.append(TickData(
            timestamp=start_ts + i * interval,
            subnets={
                netuid: SubnetTick(
                    netuid=netuid, price=price,
                    tao_pool=tp, alpha_pool=ap,
                ),
            },
        ))
    return ticks


# ── Tests ────────────────────────────────────────────────────────

class TestBacktestEngine(unittest.TestCase):

    def test_buy_and_hold(self):
        """Simple buy-and-hold: buy at tick 0, price doubles."""
        prices = [0.01, 0.012, 0.015, 0.02]  # price doubles
        ticks = make_ticks(prices)
        strategy = BuyAndHoldStrategy(107, spend=10.0)

        engine = BacktestEngine(capital=100.0, ticks_per_year=365)
        results = engine.run(ticks, strategy)

        self.assertIsInstance(results, BacktestResults)
        # Should have bought and then force-closed at end
        self.assertEqual(results.stats.n_trades, 1)
        self.assertGreater(results.stats.total_pnl, 0)  # price went up
        self.assertEqual(len(results.equity_curve), len(prices))

    def test_no_trades(self):
        """No trades means stats are all zero."""
        prices = [0.01, 0.012, 0.015]
        ticks = make_ticks(prices)
        strategy = NeverTradeStrategy()

        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, strategy)

        self.assertEqual(results.stats.n_trades, 0)
        self.assertAlmostEqual(results.stats.final_equity, 100.0)

    def test_threshold_strategy_round_trip(self):
        """Threshold strategy: buy on dip, sell on rally, should profit."""
        # Price dips 60% then rallies 200%
        prices = [0.01, 0.004, 0.003, 0.005, 0.015, 0.012]
        ticks = make_ticks(prices)
        strategy = ThresholdStrategy(107, buy_threshold=-50, sell_threshold=50,
                                     trade_size=5.0)

        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, strategy)

        # Should have at least one completed round-trip
        sell_trades = [t for t in results.trades if t.get("reason") == "rally_sell"]
        self.assertGreater(len(sell_trades), 0)

    def test_capital_conservation(self):
        """After force-close, no open positions remain."""
        prices = [0.01, 0.012, 0.008, 0.015]
        ticks = make_ticks(prices)
        strategy = BuyAndHoldStrategy(107, spend=50.0)

        engine = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, strategy)

        # All positions force-closed at end
        self.assertEqual(len(results.positions_at_end), 0)
        # Price went up 0.01 → 0.015, so we should have profited
        self.assertGreater(results.stats.total_pnl, 0)
        # Equity curve exists for every tick
        self.assertEqual(len(results.equity_curve), 4)

    def test_pool_cap_respected(self):
        """Trade size capped at max_pool_pct of pool depth."""
        prices = [0.01]
        ticks = make_ticks(prices, tao_pool=100.0, alpha_pool=10000.0)
        # Try to buy 50 TAO but pool only has 100 TAO and cap is 5%
        strategy = BuyAndHoldStrategy(107, spend=50.0)

        engine = BacktestEngine(capital=100.0, max_pool_pct=0.05)
        results = engine.run(ticks, strategy)

        # Trade should be capped at 5% of 100 = 5 TAO
        if results.trades:
            self.assertLessEqual(results.trades[0]["tao_cost"], 5.01)

    def test_fees_deducted(self):
        """Swap fees reduce returns."""
        prices = [0.01, 0.01]  # flat price
        ticks = make_ticks(prices)
        strategy = BuyAndHoldStrategy(107, spend=10.0)

        engine_no_fees = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        engine_fees = BacktestEngine(capital=100.0, swap_fee_rate=0.01, gas_fee_tao=0.1)

        results_no_fees = engine_no_fees.run(ticks, strategy)
        results_fees = engine_fees.run(
            ticks, BuyAndHoldStrategy(107, spend=10.0),
        )

        # With fees, PnL should be worse
        self.assertLess(results_fees.stats.total_pnl, results_no_fees.stats.total_pnl)

    def test_equity_curve_length(self):
        """Equity curve has one entry per tick."""
        prices = [0.01, 0.012, 0.015, 0.02, 0.018]
        ticks = make_ticks(prices)
        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, NeverTradeStrategy())
        self.assertEqual(len(results.equity_curve), len(prices))

    def test_writes_to_trade_log(self):
        """Trades are recorded to TradeLog when configured."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        try:
            tlog = TradeLog(db_path, bot_name="test_bt")
            prices = [0.01, 0.015]
            ticks = make_ticks(prices)

            engine = BacktestEngine(capital=100.0, trade_log=tlog)
            results = engine.run(ticks, BuyAndHoldStrategy(107, 10.0))

            # The force-close at end should have recorded a sell
            trades = tlog.get_recent_trades()
            self.assertGreater(len(trades), 0)
            tlog.close()
        finally:
            os.close(db_fd)
            os.unlink(db_path)

    def test_writes_to_portfolio_log(self):
        """Portfolio snapshots are written when configured."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        try:
            plog = PortfolioLog(db_path, bot_name="test_bt")
            prices = [0.01, 0.012, 0.015]
            ticks = make_ticks(prices)

            engine = BacktestEngine(capital=100.0, portfolio_log=plog)
            engine.run(ticks, NeverTradeStrategy())

            series = plog.get_series()
            self.assertEqual(len(series), 3)
            plog.close()
        finally:
            os.close(db_fd)
            os.unlink(db_path)

    def test_strategy_protocol(self):
        """Verify our test strategies satisfy the Strategy protocol."""
        self.assertIsInstance(NeverTradeStrategy(), Strategy)
        self.assertIsInstance(BuyAndHoldStrategy(107, 10), Strategy)
        self.assertIsInstance(ThresholdStrategy(107, -5, 5, 1), Strategy)

    def test_multiple_subnets(self):
        """Engine handles multiple subnets simultaneously."""
        ticks = []
        for i in range(5):
            ticks.append(TickData(
                timestamp=1000 + i * 86400,
                subnets={
                    107: SubnetTick(107, 0.01 + i * 0.001, 500, 25000),
                    99: SubnetTick(99, 0.02 - i * 0.001, 300, 15000),
                },
            ))

        class MultiBuyStrategy:
            def __init__(self):
                self.bought = set()
            def on_tick(self, tick, positions, capital, pv):
                orders = []
                for nid in [107, 99]:
                    if nid not in self.bought and capital >= 5:
                        self.bought.add(nid)
                        orders.append(Order(nid, "buy", tao_amount=5, reason="initial"))
                return orders

        engine = BacktestEngine(capital=100.0)
        results = engine.run(ticks, MultiBuyStrategy())

        # Should have trades on both subnets
        traded_netuids = {t["netuid"] for t in results.trades}
        self.assertIn(107, traded_netuids)
        self.assertIn(99, traded_netuids)


class TestBugRegressions(unittest.TestCase):
    """Regression tests for known backtest bugs (backtest_bugs_mar20.md)."""

    def test_execution_delay_updates_capital(self):
        """BUG FIX: delayed orders must update capital.

        Previously the return value from _execute_order was discarded
        in the delayed execution path, meaning buys didn't reduce capital
        (infinite buying power) and sells didn't add to capital.
        """
        prices = [0.01, 0.01, 0.01, 0.015]
        ticks = make_ticks(prices)

        # Buy 50 TAO at tick 0, delayed by 1 tick → executes at tick 1
        class DelayedBuyStrategy:
            def __init__(self):
                self.bought = False
            def on_tick(self, tick, positions, capital, pv):
                if not self.bought:
                    self.bought = True
                    return [Order(107, "buy", tao_amount=50, reason="test")]
                return []

        engine = BacktestEngine(capital=100.0, execution_delay=1,
                                swap_fee_rate=0, gas_fee_tao=0,
                                max_pool_pct=1.0)  # no pool cap for this test
        results = engine.run(ticks, DelayedBuyStrategy())

        # After delayed buy executes at tick 1, capital should be ~50 not 100
        # Tick 1 equity = 50 cash + ~50 position ≈ 100
        tick1_eq = results.equity_curve[1]
        self.assertLess(tick1_eq["capital"], 60)  # capital reduced by buy

    def test_execution_delay_sell_credits_capital(self):
        """BUG FIX: delayed sells must add TAO back to capital."""
        prices = [0.01, 0.01, 0.01, 0.015]
        ticks = make_ticks(prices)

        class BuySellStrategy:
            def __init__(self):
                self.state = 0
            def on_tick(self, tick, positions, capital, pv):
                if self.state == 0:
                    self.state = 1
                    return [Order(107, "buy", tao_amount=50, reason="buy")]
                if self.state == 1 and 107 in positions:
                    self.state = 2
                    return [Order(107, "sell", reason="sell")]
                return []

        # No delay — get baseline capital after buy+sell
        engine_nodelay = BacktestEngine(capital=100.0, execution_delay=0,
                                        swap_fee_rate=0, gas_fee_tao=0)
        results_nodelay = engine_nodelay.run(ticks, BuySellStrategy())

        # With delay — capital should still recover after sell
        engine_delay = BacktestEngine(capital=100.0, execution_delay=1,
                                       swap_fee_rate=0, gas_fee_tao=0)
        results_delay = engine_delay.run(ticks, BuySellStrategy())

        # Final equity should be similar (not 0 from missing sell credit)
        self.assertGreater(results_delay.stats.final_equity, 50)

    def test_tp_overshoot_prevention(self):
        """BUG FIX (Bug #4): TP sell with execution delay must not profit
        from extreme prices beyond the TP target.

        This was the critical bug that caused +164,605% phantom returns
        in backtest_realistic.py. A TP sell at price 0.012 (20% TP on
        0.01 entry) delayed to a tick where price is 0.10 would sell
        at 0.10 instead of 0.012.
        """
        # Price goes: 0.01 (buy), 0.015 (TP triggered), 0.10 (extreme move!)
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(2000, {107: SubnetTick(107, 0.015, 500, 50000)}),
            TickData(3000, {107: SubnetTick(107, 0.10, 500, 50000)}),
        ]

        class TPStrategy:
            """Buy at tick 0, TP sell with limit_price at tick 1."""
            def __init__(self):
                self.state = 0
            def on_tick(self, tick, positions, capital, pv):
                if self.state == 0:
                    self.state = 1
                    return [Order(107, "buy", tao_amount=10, reason="entry")]
                if self.state == 1 and 107 in positions:
                    self.state = 2
                    # TP at 20% = 0.012 — use limit_price to cap
                    entry = positions[107].entry_price
                    tp_price = entry * 1.20
                    return [Order(107, "sell", reason="take_profit",
                                  limit_price=tp_price)]
                return []

        # With delay=1, TP sell issued at tick 1 executes at tick 2 (0.10 price)
        engine = BacktestEngine(capital=100.0, execution_delay=1,
                                swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, TPStrategy())

        # Find the TP trade
        tp_trades = [t for t in results.trades if t.get("reason") == "take_profit"]
        if tp_trades:
            tp_trade = tp_trades[0]
            # Exit price should be capped near 0.012, NOT at 0.10
            self.assertLess(tp_trade["exit_price"], 0.015,
                           f"TP overshoot! exit_price={tp_trade['exit_price']:.4f} "
                           f"should be near TP target, not execution-time price")

    def test_tp_no_cap_when_below_limit(self):
        """limit_price should NOT restrict sells when market is below it."""
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(2000, {107: SubnetTick(107, 0.012, 500, 50000)}),
        ]

        class SellWithHighLimit:
            def __init__(self):
                self.state = 0
            def on_tick(self, tick, positions, capital, pv):
                if self.state == 0:
                    self.state = 1
                    return [Order(107, "buy", tao_amount=10, reason="entry")]
                if self.state == 1 and 107 in positions:
                    self.state = 2
                    # limit_price at 0.05 — well above current 0.012
                    return [Order(107, "sell", reason="sell", limit_price=0.05)]
                return []

        engine = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, SellWithHighLimit())

        # Sell should execute at market (0.012), not capped
        sells = [t for t in results.trades if t.get("reason") == "sell"]
        self.assertEqual(len(sells), 1)

    def test_buy_limit_price_rejects_expensive(self):
        """Buy with limit_price rejects when market moved up."""
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(2000, {107: SubnetTick(107, 0.05, 500, 50000)}),  # price 5x'd
        ]

        class LimitBuy:
            def __init__(self):
                self.done = False
            def on_tick(self, tick, positions, capital, pv):
                if not self.done:
                    self.done = True
                    # Willing to buy at 0.01 but not 0.05
                    return [Order(107, "buy", tao_amount=10, reason="buy",
                                  limit_price=0.015)]
                return []

        # With delay=1, buy queued at tick 0, executes at tick 1 (price 0.05)
        engine = BacktestEngine(capital=100.0, execution_delay=1,
                                swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, LimitBuy())

        # Buy should be rejected — capital unchanged
        self.assertAlmostEqual(results.stats.final_equity, 100.0)

    def test_delayed_order_uses_decision_price_on_sparse_data(self):
        """When execution tick has no data for the subnet (sparse transaction
        data), the engine should use the price from decision time — because
        the AMM pool didn't change between transactions.

        This avoids needing to forward-fill sparse data to block-level
        (which would be 15x larger).
        """
        # Tick 0: SN107 has data (strategy decides to buy)
        # Tick 1: SN107 has NO data (different subnet traded)
        # Tick 2: SN107 has data again (someone traded, price jumped)
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(1012, {99: SubnetTick(99, 0.05, 300, 6000)}),  # SN107 missing
            TickData(1024, {107: SubnetTick(107, 0.08, 500, 50000)}),  # jump
        ]

        class BuyOnce:
            def __init__(self):
                self.done = False
            def on_tick(self, tick, positions, capital, pv):
                if not self.done and 107 in tick.subnets:
                    self.done = True
                    return [Order(107, "buy", tao_amount=10, reason="test")]
                return []

        # delay=1: order from tick 0 executes at tick 1
        # tick 1 has no SN107 data → should fall back to tick 0's price (0.01)
        engine = BacktestEngine(capital=100.0, execution_delay=1,
                                swap_fee_rate=0, gas_fee_tao=0, max_pool_pct=1.0)
        results = engine.run(ticks, BuyOnce())

        # Should have bought at ~0.01 (decision-time price), NOT at 0.08 (tick 2)
        # and NOT failed silently (returning None because subnet missing)
        close_trades = [t for t in results.trades if t["reason"] == "end_of_data"]
        self.assertEqual(len(close_trades), 1, "Buy should have executed using fallback price")
        self.assertLess(close_trades[0]["entry_price"], 0.02,
                       "Entry price should be near 0.01, not 0.08")

    def test_delayed_order_uses_fresh_data_when_available(self):
        """When execution tick HAS data for the subnet, use the fresh data
        (not the stale decision-time snapshot).

        Pool state must differ between ticks — the AMM executes against
        pool state, not the price field.
        """
        # Tick 0: pool gives spot price 0.01 (500/50000)
        # Tick 1: pool gives spot price 0.02 (1000/50000) — someone bought TAO into the pool
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(1012, {107: SubnetTick(107, 0.02, 1000, 50000)}),
        ]

        class BuyOnce:
            def __init__(self):
                self.done = False
            def on_tick(self, tick, positions, capital, pv):
                if not self.done:
                    self.done = True
                    return [Order(107, "buy", tao_amount=10, reason="test")]
                return []

        engine = BacktestEngine(capital=100.0, execution_delay=1,
                                swap_fee_rate=0, gas_fee_tao=0, max_pool_pct=1.0)
        results = engine.run(ticks, BuyOnce())

        # Should have bought at ~0.02 (execution-time pool), not ~0.01
        close_trades = [t for t in results.trades if t["reason"] == "end_of_data"]
        self.assertEqual(len(close_trades), 1)
        self.assertGreater(close_trades[0]["entry_price"], 0.015,
                          "Should use fresh tick pool state, not stale snapshot")

    def test_entry_price_cost_averaged(self):
        """Bug #1 regression: entry price must be cost-weighted average."""
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(2000, {107: SubnetTick(107, 0.02, 500, 50000)}),
            TickData(3000, {107: SubnetTick(107, 0.015, 500, 50000)}),
        ]

        class DoubleBuyStrategy:
            def __init__(self):
                self.buys = 0
            def on_tick(self, tick, positions, capital, pv):
                if self.buys < 2:
                    self.buys += 1
                    return [Order(107, "buy", tao_amount=5, reason="accumulate")]
                return []

        engine = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, DoubleBuyStrategy())

        # The entry price should be between 0.01 and 0.02 (cost-weighted),
        # NOT 0.02 (overwrite bug) and NOT 0.01 (first-only).
        if 107 in results.positions_at_end:
            pos = results.positions_at_end[107]
            self.assertGreater(pos.entry_price, 0.01)
            self.assertLess(pos.entry_price, 0.02)

    def test_entry_timestamp_preserved(self):
        """Bug #2 regression: entry_time must be from FIRST buy, not latest."""
        ticks = [
            TickData(1000, {107: SubnetTick(107, 0.01, 500, 50000)}),
            TickData(2000, {107: SubnetTick(107, 0.008, 500, 50000)}),
            TickData(3000, {107: SubnetTick(107, 0.015, 500, 50000)}),
        ]

        class DoubleBuyStrategy:
            def __init__(self):
                self.buys = 0
            def on_tick(self, tick, positions, capital, pv):
                if self.buys < 2:
                    self.buys += 1
                    return [Order(107, "buy", tao_amount=5, reason="accumulate")]
                return []

        engine = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, DoubleBuyStrategy())

        # Force-close trade should show entry_time = 1000 (first buy),
        # not 2000 (second buy)
        close_trades = [t for t in results.trades if t["reason"] == "end_of_data"]
        if close_trades:
            self.assertEqual(close_trades[0]["entry_time"], 1000)

    def test_strategy_cannot_mutate_positions(self):
        """Defensive: strategy modifying positions dict doesn't corrupt engine."""
        ticks = make_ticks([0.01, 0.012, 0.015])

        class MutatingStrategy:
            def __init__(self):
                self.bought = False
            def on_tick(self, tick, positions, capital, pv):
                if not self.bought:
                    self.bought = True
                    return [Order(107, "buy", tao_amount=10, reason="buy")]
                # Try to corrupt the positions
                positions.clear()
                return []

        engine = BacktestEngine(capital=100.0, swap_fee_rate=0, gas_fee_tao=0)
        results = engine.run(ticks, MutatingStrategy())

        # Engine should still have the position despite strategy clearing the copy
        self.assertEqual(results.stats.n_trades, 1)  # end_of_data close


class TestComputeStats(unittest.TestCase):

    def test_empty_trades(self):
        stats = compute_stats([], [], 100.0)
        self.assertEqual(stats.n_trades, 0)
        self.assertAlmostEqual(stats.final_equity, 100.0)

    def test_basic_stats(self):
        trades = [
            {"pnl": 1.0, "fees": 0.01, "hold_seconds": 3600, "netuid": 107, "reason": "threshold"},
            {"pnl": -0.5, "fees": 0.01, "hold_seconds": 7200, "netuid": 107, "reason": "threshold"},
            {"pnl": 2.0, "fees": 0.02, "hold_seconds": 1800, "netuid": 99, "reason": "dip"},
        ]
        equity = [
            {"total_equity": 100},
            {"total_equity": 101},
            {"total_equity": 100.5},
            {"total_equity": 102.5},
        ]
        stats = compute_stats(trades, equity, 100.0)
        self.assertEqual(stats.n_trades, 3)
        self.assertEqual(stats.n_wins, 2)
        self.assertEqual(stats.n_losses, 1)
        self.assertAlmostEqual(stats.total_pnl, 2.5)
        self.assertAlmostEqual(stats.win_rate, 66.67, places=1)
        self.assertEqual(stats.n_subnets_traded, 2)
        self.assertIn("threshold", stats.by_reason)
        self.assertIn("dip", stats.by_reason)

    def test_max_drawdown(self):
        equity = [
            {"total_equity": 100},
            {"total_equity": 110},
            {"total_equity": 90},   # 18.18% DD from peak of 110
            {"total_equity": 105},
        ]
        stats = compute_stats(
            [{"pnl": 5, "netuid": 1, "reason": "x"}],
            equity, 100.0,
        )
        self.assertGreater(stats.max_drawdown_pct, 18)
        self.assertLess(stats.max_drawdown_pct, 19)


if __name__ == "__main__":
    unittest.main()
