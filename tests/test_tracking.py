"""Tests for bt_trading_tools.tracking — trade log, decision log, portfolio log, event log."""

import json
import os
import tempfile
import unittest

from bt_trading_tools.tracking.trade_log import TradeLog
from bt_trading_tools.tracking.decision_log import DecisionLog
from bt_trading_tools.tracking.portfolio_log import PortfolioLog
from bt_trading_tools.tracking.event_log import EventLog


class TestTradeLog(unittest.TestCase):

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.log = TradeLog(self.db_path, bot_name="test_bot")

    def tearDown(self):
        self.log.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_record_and_retrieve(self):
        rec = self.log.record_trade(
            "buy", netuid=107, tao_amount=0.5, alpha_amount=100,
            price=0.005, slippage=0.3, hotkey="5Gtest",
            reason="inventory_build", signal_data={"pct_change": -1.2},
            timestamp=1000,
        )
        self.assertEqual(rec.netuid, 107)
        self.assertEqual(rec.reason, "inventory_build")
        trades = self.log.get_recent_trades(limit=5)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].signal_data, {"pct_change": -1.2})

    def test_single_buy_cost_basis(self):
        self.log.record_trade(
            "buy", 10, tao_amount=1.0, alpha_amount=100,
            price=0.01, slippage=0, hotkey="h", timestamp=1000,
        )
        basis = self.log.get_cost_basis(10)
        self.assertAlmostEqual(basis["avg_buy_price"], 0.01)
        self.assertAlmostEqual(basis["total_tao_invested"], 1.0)
        self.assertAlmostEqual(basis["total_alpha_held"], 100)

    def test_weighted_average_two_buys(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("buy", 10, 2.0, 100, 0.02, 0, "h", timestamp=2000)
        basis = self.log.get_cost_basis(10)
        self.assertAlmostEqual(basis["avg_buy_price"], 3.0 / 200)
        self.assertAlmostEqual(basis["total_tao_invested"], 3.0)
        self.assertAlmostEqual(basis["total_alpha_held"], 200)

    def test_sell_reduces_basis(self):
        self.log.record_trade("buy", 10, 2.0, 200, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("sell", 10, 1.5, 100, 0.015, 0, "h", timestamp=2000)
        basis = self.log.get_cost_basis(10)
        # Sold 50%: invested reduced to 1.0, alpha to 100
        self.assertAlmostEqual(basis["total_alpha_held"], 100)
        self.assertAlmostEqual(basis["total_tao_invested"], 1.0)
        # P&L: received 1.5, cost was 1.0 → realized = 0.5
        self.assertAlmostEqual(basis["realized_pnl"], 0.5)

    def test_sell_all_zeros_out(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("sell", 10, 1.2, 100, 0.012, 0, "h", timestamp=2000)
        basis = self.log.get_cost_basis(10)
        self.assertAlmostEqual(basis["total_alpha_held"], 0.0)
        self.assertAlmostEqual(basis["total_tao_invested"], 0.0)
        self.assertAlmostEqual(basis["realized_pnl"], 0.2)

    def test_realized_pnl_loss(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("sell", 10, 0.8, 100, 0.008, 0, "h", timestamp=2000)
        basis = self.log.get_cost_basis(10)
        self.assertAlmostEqual(basis["realized_pnl"], -0.2)

    def test_no_trades_returns_none(self):
        self.assertIsNone(self.log.get_cost_basis(999))

    def test_multiple_subnets_independent(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("buy", 20, 2.0, 50, 0.04, 0, "h", timestamp=1000)
        b10 = self.log.get_cost_basis(10)
        b20 = self.log.get_cost_basis(20)
        self.assertAlmostEqual(b10["avg_buy_price"], 0.01)
        self.assertAlmostEqual(b20["avg_buy_price"], 0.04)

    def test_portfolio_summary(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.record_trade("buy", 20, 2.0, 50, 0.04, 0, "h", timestamp=1000)
        self.log.record_trade("sell", 10, 1.5, 100, 0.015, 0, "h", timestamp=2000)
        summary = self.log.get_portfolio_summary()
        self.assertAlmostEqual(summary["total_invested"], 2.0)  # 20 still open
        self.assertAlmostEqual(summary["total_received"], 1.5)
        self.assertAlmostEqual(summary["realized_pnl"], 0.5)

    def test_bot_name_isolation(self):
        """Different bot_name doesn't see other bot's trades."""
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        other = TradeLog(self.db_path, bot_name="other_bot")
        self.assertIsNone(other.get_cost_basis(10))
        other.close()

    def test_baseline_buy(self):
        self.log.insert_baseline_buy(10, 500, 0.02, "h", timestamp=500)
        basis = self.log.get_cost_basis(10)
        self.assertAlmostEqual(basis["total_alpha_held"], 500)
        self.assertAlmostEqual(basis["total_tao_invested"], 10.0)

    def test_snapshots_and_cleanup(self):
        self.log.record_snapshot(10, 100, 0.01, 0.8, timestamp=1000)
        self.log.record_snapshots_bulk([(10, 100, 0.015, 0.8), (20, 50, 0.04, 1.5)],
                                        timestamp=2000)
        # Cleanup with 0 days should remove everything
        self.log.cleanup_old_snapshots(max_age_days=0)

    def test_delete_subnet_trades(self):
        self.log.record_trade("buy", 10, 1.0, 100, 0.01, 0, "h", timestamp=1000)
        self.log.delete_subnet_trades(10)
        self.assertIsNone(self.log.get_cost_basis(10))

    def test_signal_data_preserved(self):
        self.log.record_trade(
            "buy", 10, 1.0, 100, 0.01, 0, "h",
            reason="rally_sell",
            signal_data={"pct_change": 8.4, "threshold": 7.0, "pool_tao": 150},
            timestamp=1000,
        )
        trades = self.log.get_recent_trades()
        self.assertEqual(trades[0].signal_data["pct_change"], 8.4)
        self.assertEqual(trades[0].reason, "rally_sell")


class TestDecisionLog(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "decisions.jsonl")
        self.dlog = DecisionLog(self.path, bot_name="test_bot")

    def tearDown(self):
        self.dlog.close()
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.tmpdir)

    def test_record_and_iterate(self):
        self.dlog.record_tick(
            tick=1, portfolio_value=50.0, cash=10.0, n_positions=3,
            decisions=[
                {"netuid": 107, "action": "hold", "reason": "below_threshold"},
                {"netuid": 99, "action": "sell", "reason": "rally_sell", "pct_change": 8.4},
            ],
            timestamp=1000,
        )
        self.dlog.record_tick(
            tick=2, portfolio_value=50.5, cash=9.5, n_positions=4,
            decisions=[],
            timestamp=1012,
        )
        self.dlog.close()

        ticks = list(DecisionLog.iter_ticks(self.path))
        self.assertEqual(len(ticks), 2)
        self.assertEqual(ticks[0]["tick"], 1)
        self.assertEqual(len(ticks[0]["decisions"]), 2)

    def test_find_tick(self):
        self.dlog.record_tick(tick=1, portfolio_value=50, cash=10, n_positions=0,
                              decisions=[], timestamp=1000)
        self.dlog.record_tick(tick=2, portfolio_value=51, cash=10, n_positions=0,
                              decisions=[], timestamp=1012)
        self.dlog.close()

        found = DecisionLog.find_tick(self.path, tick=2)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["pv"], 51.0)

    def test_find_trades_for_netuid(self):
        self.dlog.record_tick(
            tick=1, portfolio_value=50, cash=10, n_positions=0,
            decisions=[{"netuid": 107, "action": "buy", "reason": "reload"}],
            timestamp=1000,
        )
        self.dlog.record_tick(
            tick=2, portfolio_value=50, cash=10, n_positions=1,
            decisions=[{"netuid": 107, "action": "hold", "reason": "no_signal"}],
            timestamp=1012,
        )
        self.dlog.close()

        results = DecisionLog.find_trades_for_netuid(self.path, netuid=107)
        # Only tick 1 had an action (buy), tick 2 was hold
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tick"], 1)

    def test_bot_filter(self):
        self.dlog.record_tick(tick=1, portfolio_value=50, cash=10, n_positions=0,
                              decisions=[], timestamp=1000)
        self.dlog.close()

        ticks = list(DecisionLog.iter_ticks(self.path, bot_name="wrong_bot"))
        self.assertEqual(len(ticks), 0)

    def test_market_snapshot(self):
        self.dlog.record_tick(
            tick=1, portfolio_value=50, cash=10, n_positions=0,
            decisions=[],
            market_snapshot={"universe_size": 15, "block": 12345},
            timestamp=1000,
        )
        self.dlog.close()
        ticks = list(DecisionLog.iter_ticks(self.path))
        self.assertEqual(ticks[0]["market"]["universe_size"], 15)


class TestPortfolioLog(unittest.TestCase):

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.plog = PortfolioLog(self.db_path, bot_name="test_bot")

    def tearDown(self):
        self.plog.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_record_and_get_series(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 1, timestamp=1000)
        self.plog.record(51.0, 9.0, 42.0, 4, 0, timestamp=1012)
        rows = self.plog.get_series()
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0][1], 50.0)  # total_value
        self.assertAlmostEqual(rows[1][1], 51.0)

    def test_get_latest(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 0, timestamp=1000)
        self.plog.record(55.0, 12.0, 43.0, 4, 0, timestamp=2000)
        latest = self.plog.get_latest()
        self.assertAlmostEqual(latest[1], 55.0)

    def test_get_series_with_since(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 0, timestamp=1000)
        self.plog.record(51.0, 9.0, 42.0, 4, 0, timestamp=2000)
        self.plog.record(52.0, 8.0, 44.0, 5, 0, timestamp=3000)
        rows = self.plog.get_series(since=1500)
        self.assertEqual(len(rows), 2)

    def test_get_all_bots(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 0, timestamp=1000)
        other = PortfolioLog(self.db_path, bot_name="other_bot")
        other.record(100.0, 20.0, 80.0, 5, 0, timestamp=1000)
        bots = self.plog.get_all_bots()
        self.assertIn("test_bot", bots)
        self.assertIn("other_bot", bots)
        other.close()

    def test_export_csv(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 0, timestamp=1000)
        csv_path = self.db_path + ".csv"
        self.plog.export_csv(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # header + 1 row
        self.assertIn("total_value", lines[0])
        os.unlink(csv_path)

    def test_bot_isolation(self):
        self.plog.record(50.0, 10.0, 40.0, 3, 0, timestamp=1000)
        other = PortfolioLog(self.db_path, bot_name="other_bot")
        rows = other.get_series()
        self.assertEqual(len(rows), 0)
        other.close()


class TestEventLog(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "events.jsonl")
        self.elog = EventLog(self.path, bot_name="test_bot")

    def tearDown(self):
        self.elog.close()
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.tmpdir)

    def test_levels(self):
        self.elog.info("bot_start", detail={"version": "1.0"}, timestamp=1000)
        self.elog.trade("rally_sell", netuid=107, tao=0.5, pnl=0.12, timestamp=1012)
        self.elog.warning("reconnect", detail={"attempt": 2}, timestamp=1024)
        self.elog.error("subtensor_timeout", detail={"wait_s": 8}, timestamp=1036)
        self.elog.close()

        events = list(EventLog.iter_events(self.path))
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["level"], "info")
        self.assertEqual(events[1]["level"], "trade")
        self.assertEqual(events[1]["detail"]["netuid"], 107)
        self.assertEqual(events[2]["level"], "warning")
        self.assertEqual(events[3]["level"], "error")

    def test_filter_by_level(self):
        self.elog.info("start", timestamp=1000)
        self.elog.trade("buy", netuid=10, tao=1.0, timestamp=1012)
        self.elog.error("crash", timestamp=1024)
        self.elog.close()

        trades = list(EventLog.iter_events(self.path, level="trade"))
        self.assertEqual(len(trades), 1)
        errors = list(EventLog.iter_events(self.path, level="error"))
        self.assertEqual(len(errors), 1)

    def test_filter_by_bot(self):
        self.elog.info("start", timestamp=1000)
        self.elog.close()

        events = list(EventLog.iter_events(self.path, bot_name="wrong_bot"))
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
